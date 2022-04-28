#include "huffman.h"
#include "parallel.h"

/* Print program usage. */
void prog_usage(const char *prog) {
    printf("huffman: A simple text-based Huffman {en,de}coder.\n\n"
           "USAGE\n"
           "  %s (-e | -d) -i INPUT -o OUTPUT [-h]\n\n"
           "ARGUMENTS\n"
           "  -e  encode (default operation)\n"
           "  -d  decode\n"
           "  -l  use look-up table\n"
           "  -p  parallel decoding\n"
           "  -i  input file path\n"
           "  -o  output file path\n"
           "  -h  display program usage\n",
           prog);
}

/* Exit on error. */
void err_exit(const char *message) {
    perror(message);
    exit(EXIT_FAILURE);
}

/*
 * Encode all the bytes input file, and write it to the
 * output file.
 *
 * The Huffman code for each byte is computed and the bits
 * are packed together until the size of a byte is reached.
 * The byte is then written to the output file.
 */
void encode(FILE *ifile, struct node *root, FILE *ofile, uint64_t *nr_rd_bytes,
            uint64_t *nr_wr_bytes) {
    uint8_t shift = 0, code = 0, *arr, chunk;
    uint32_t th;
    uint64_t nr_rbytes = 0, nr_wbytes = 0;
    int8_t i, off;
    int16_t file_ch;

    assert(root);
    assert(ifile);
    assert(ofile);
    assert(nr_rd_bytes);
    assert(nr_wr_bytes);

    th = tree_height(root);
    assert(th > 0);

    /* Temporary array to store the Huffman code. */
    arr = calloc(th, sizeof(uint8_t));
    assert(arr);

    while ((file_ch = fgetc(ifile)) != EOF) {
        /* Read a byte from the file. */
        chunk = (uint8_t)file_ch;

        /* Calculate the Huffman code. */
        off = huffman_code(chunk, root, arr);
        assert(off > 0);

        /*
         * Pack the bits into a byte by shifting left; this may
         * happen across iterations, until we reach a full byte.
         */
        for (i = 0; i < off; i++) {
            code = (code << 1) | arr[i];
            shift++;

            /* If we have a fully encoded byte, write it ot the file. */
            if (shift >= MAX_INT_BUF_BITS) {
                fputc(code, ofile);
                nr_wbytes++;

                /* Reset the temporary bit buffer and shift count. */
                code = 0;
                shift = 0;
            }
        }

        nr_rbytes++;
    }

    /*
     * Append the pseudo-EOF byte to indicate the end of the
     * encoded file. While decoding, we stop at this byte.
     */
    off = huffman_code(PSEUDO_NULL_BYTE, root, arr);
    for (i = 0; i < off; i++) {
        code = (code << 1) | arr[i];
        shift++;

        if (shift >= MAX_INT_BUF_BITS) {
            fputc(code, ofile);
            nr_wbytes++;

            code = 0;
            shift = 0;
        }
    }

    /*
     * If we previously ended with a bit offset that did not fully occupy
     * a byte, that byte must also be written to the file. We also shift
     * the remaining bits to the left for the same reason. Otherwise, this
     * byte would have stray "0"s from the most-significant-bit and will
     * result in a wrongly decoded value.
     */
    if (shift && shift < MAX_INT_BUF_BITS) {
        code <<= (MAX_INT_BUF_BITS - shift);
        fputc(code, ofile);
        nr_wbytes++;
    }

    *nr_rd_bytes = nr_rbytes;
    *nr_wr_bytes = nr_wbytes;

    free(arr);
}

/*
 * Decode all the bytes in the encoded file (until the pseudo-EOF
 * byte is reached) and write them to the output file. The return
 * value is the number of bytes decoded.
 *
 * This function should be called only after reading the file
 * headers (which contains the deflated Huffman tree), and the
 * file offset is at the start of the first decodable byte.
 *
 * Decoding is done by extracting bits from each byte read from
 * the file. After reaching a leaf node by walking the Huffman
 * tree (depending on: "0" - left, or "1" - right) from the bits.
 * The resultant byte obtained from the tree traversal is written
 * to the output file.
 */
void decode_with_tree(FILE *ifile, uint64_t nr_en_bytes, struct node *root,
                      FILE *ofile, uint64_t *nr_rd_bytes,
                      uint64_t *nr_wr_bytes) {
    uint8_t shift = 0, chunk, mask;
    uint64_t nr_rbytes = 0, nr_wbytes = 0;
    int16_t file_ch;
    struct node *branch;

    assert(root);
    assert(ifile);
    assert(ofile);
    assert(nr_rd_bytes);
    assert(nr_wr_bytes);

    branch = root;

    /*
     * Extract the most significant bit of the byte.
     * This represents the root node of the tree.
     */
    mask = 0x1U << (MAX_INT_BUF_BITS - 1);

    file_ch = fgetc(ifile);

    if (file_ch == EOF)
        goto ret;

    nr_rbytes++;

    while (1) {
        chunk = (uint8_t)file_ch << shift;
        shift++;

        /* Determine which branch to take depending on the extracted bit. */
        branch = (chunk & mask) ? branch->right : branch->left;

        /*
         * If we reached the leaf node, we have decoded a byte;
         * write it to the output file.
         */
        if (tree_leaf(branch)) {
            /* This marks the end of the decoded file. */
            if (branch->data.ch == PSEUDO_NULL_BYTE && nr_rbytes >= nr_en_bytes)
                break;

            fputc(branch->data.ch, ofile);
            nr_wbytes++;

            /* Reset the branch to the root for the next byte. */
            branch = root;
        }

        /*
         * If we have shifted all the bits, we should read
         * another byte from the file and start extracting
         * bits.
         */
        if (shift >= MAX_INT_BUF_BITS) {
            file_ch = fgetc(ifile);

            /* If there are no more bytes to read. */
            if (file_ch == EOF)
                break;

            nr_rbytes++;

            /* Reset the shift counter. */
            shift = 0;
        }
    }

ret:
    *nr_rd_bytes = nr_rbytes;
    *nr_wr_bytes = nr_wbytes;
}

/*
 * Decode all the bytes in the encoded file (until the pseudo-EOF
 * byte is reached) and write them to the output file. The return
 * value is the number of bytes decoded.
 *
 * Unlike "decode_with_tree()", this function uses a look-up table for
 * Huffman codes. Decoding is done by extracting bits from each byte read
 * from the file. After reaching the maximum shift offset (log2 of the
 * number of entries in the look-up table), the decoded byte present at
 * that index is written to the file.
 */
void decode_with_tab(FILE *ifile, uint64_t nr_en_bytes, struct lookup *tab,
                     uint32_t tab_sz, FILE *ofile, uint64_t *nr_rd_bytes,
                     uint64_t *nr_wr_bytes) {
    uint8_t mask, bshft = 0, cshft = 0, mshft = 0;
    uint64_t nr_rbytes = 0, nr_wbytes = 0;
    uint32_t chunk = 0;
    int16_t file_ch;
    struct lookup *ent = NULL;

    assert(tab);
    assert(tab_sz > 0 && tab_sz <= MAX_LOOKUP_TAB_LEN);
    assert(ifile);
    assert(ofile);
    assert(nr_rd_bytes);
    assert(nr_wr_bytes);

    mshft = logb2(tab_sz);
    mask = 0x1U << (MAX_INT_BUF_BITS - 1);

    file_ch = fgetc(ifile);

    if (file_ch == EOF)
        goto ret;

    nr_rbytes++;

    while (1) {
        /*
         * Read the most-significant-bit from the byte and
         * append it to "chunk". Do this until the maximum
         * shift count is reached.
         */
        chunk <<= 1;
        chunk |= ((file_ch << bshft) & mask) ? 0x1U : 0x0U;

        bshft++;
        cshft++;

        if (cshft >= mshft) {
            /*
             * Check the lookup table and see if it contains an
             * entry for the index pointed to by "chunk".
             */
            ent = lookup_table(tab, tab_sz, chunk);
            assert(ent);

            /* No more bytes to decode, */
            if (ent->ch == PSEUDO_NULL_BYTE && nr_rbytes >= nr_en_bytes)
                goto ret;

            /* Write the decoded byte to the file. */
            fputc(ent->ch, ofile);
            nr_wbytes++;

            /*
             * If the offset (of the code) is less than the shifted
             * count, discard the decoded bits with the mask and
             * reset "chunk" to the leftover bits. Also update the
             * chunk shift count to reflect the difference.
             */
            chunk = chunk & ~((~0x0UL) << (mshft - (ent->off)));
            cshft = mshft - ent->off;
        }

        /*
         * If we have shifted all the bits, we should read
         * another byte from the file and start extracting
         * bits.
         */
        if (bshft >= MAX_INT_BUF_BITS) {
            file_ch = fgetc(ifile);
            if (file_ch == EOF)
                break;

            nr_rbytes++;

            /* Reset the byte shift counter. */
            bshft = 0;
        }
    }

ret:
    *nr_rd_bytes = nr_rbytes;
    *nr_wr_bytes = nr_wbytes;
}

/* Wrapper for decoding. */
void decode(int16_t dev, int16_t with_tree, FILE *ifile, struct meta *fmeta,
            struct node *root, FILE *ofile, uint64_t *nr_rd_bytes,
            uint64_t *nr_wr_bytes) {

    struct lookup *tab = NULL;
    uint32_t tab_sz;

    assert(root);

    if (with_tree) {
        assert(dev == 0);
        decode_with_tree(ifile, fmeta->nr_enc_bytes, root, ofile, nr_rd_bytes,
                         nr_wr_bytes);
        return;
    }

    tab = make_lookup_table(root, &tab_sz);
    assert(tab);
    assert(tab_sz > 0 && tab_sz <= MAX_LOOKUP_TAB_LEN);

    if (dev)
        dev_trampoline(ifile, fmeta, tab, tab_sz, ofile, nr_rd_bytes,
                       nr_wr_bytes);
    else
        decode_with_tab(ifile, fmeta->nr_enc_bytes, tab, tab_sz, ofile,
                        nr_rd_bytes, nr_wr_bytes);

    free(tab);
}

/* All the things happen here. */
int main(int argc, char *argv[]) {
    uint8_t *tbuf = NULL;
    uint32_t map_sz;
    uint64_t nr_rbytes = 0, nr_wbytes = 0;
    int16_t arg, enc = 1, dev = 1, with_tree = 0;
    char *ifpath = NULL, *ofpath = NULL;
    FILE *ifile = NULL, *ofile = NULL;

    struct node *head = NULL;
    struct map *fmap = NULL;
    struct meta fmeta = {0, 0, 0, 0};

    while ((arg = getopt(argc, argv, "edpli:o:h?")) > 0) {
        switch (arg) {
        case 'e':
            enc = 1;
            break;
        case 'd':
            enc = 0;
            break;
        case 'i':
            ifpath = optarg;
            break;
        case 'o':
            ofpath = optarg;
            break;
        case 'p':
            dev = 1;
            break;
        case 'l':
            with_tree = 0;
            break;
        case 'h':
        case '?':
        default:
            prog_usage(argv[0]);
            return 1;
        }
    }

    if (!(ifile = fopen(ifpath, "rb")))
        err_exit("error: failed to open input file");

    if (!(ofile = fopen(ofpath, "wb")))
        err_exit("error: failed to open output file");

    /* Encoding. */
    if (enc) {
        /* Construct the map. */
        fmap = make_map(ifile, &map_sz);
        assert(fmap);

        /* Build the priority queue. */
        head = make_queue(fmap, map_sz);
        assert(head);

        /* Make the queue into a tree. */
        make_tree(&head);

        /* Rewind back to start encoding. */
        rewind(ifile);

        /*
         *
         *  +---------+-------------------- >  < +-------------------- >  < -+
         *  |  HEADER | TREE [0, 1, ... N]  >  < | ENC [0, 1, ..., B]  >  <  |
         *  +---------+-------------------- >  < +-------------------- >  < -+
         *
         *  - HEADER    File header containing the "map" struct.
         *  - TREE[]    Encoded Huffman tree.
         *  - ENC[]     Encoded bytes.
         */

        /* Write headers to the encoded output file. */
        fwrite(&fmeta, sizeof(struct meta), 1, ofile);

        /* Deflate the tree into a buffer, and write it to the file. */
        tbuf = encode_tree(head, &fmeta.nr_tree_bytes, &fmeta.tree_lb_sh_pos);
        assert(tbuf);

        fwrite(tbuf, sizeof(uint8_t), fmeta.nr_tree_bytes, ofile);

        /* Write the encoded bytes to the file. */
        encode(ifile, head, ofile, &fmeta.nr_src_bytes, &fmeta.nr_enc_bytes);

        /*
         * Rewind back to write the number of bytes in the source, and
         * the number encoded bytes to the output file header.
         */
        rewind(ofile);
        fwrite(&fmeta, sizeof(struct meta), 1, ofile);
    } else {
        /* Decoding. */

        /* Read the file headers. */
        fread(&fmeta, sizeof(struct meta), 1, ifile);
        assert(fmeta.nr_tree_bytes > 0);
        assert(fmeta.nr_src_bytes >= 0);
        assert(fmeta.nr_enc_bytes >= 0);

        /*
         * Read the tree from file into a temporary
         * buffer and inflate it.
         */
        tbuf = calloc(fmeta.nr_tree_bytes, sizeof(uint8_t));
        assert(tbuf);
        fread(tbuf, sizeof(uint8_t), fmeta.nr_tree_bytes, ifile);

        head = decode_tree(tbuf, fmeta.nr_tree_bytes, fmeta.tree_lb_sh_pos);
        assert(head);

        /* Decode the file and write to the output file. */
        decode(dev, with_tree, ifile, &fmeta, head, ofile, &nr_rbytes,
               &nr_wbytes);

        /* Check if the decode was successful. */
        assert(fmeta.nr_enc_bytes == nr_rbytes);
        assert(fmeta.nr_src_bytes == nr_wbytes);
    }

    nuke_tree(&head);
    free(tbuf);
    free(fmap);
    fclose(ifile);
    fclose(ofile);

    return 0;
}
