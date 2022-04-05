#include "huffman.h"

/* Print program usage. */
void prog_usage(const char *prog) {
    printf("huffman: A simple text-based Huffman {enc,dec}oder.\n\n"
           "USAGE\n"
           "  %s (-e | -d) -i INPUT -o OUTPUT [-h]\n\n"
           "ARGUMENTS\n"
           "  -e  encode (default operation)\n"
           "  -d  decode\n"
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

/* Return the Huffman code for a given byte. */
int8_t huffman_code(uint8_t ch, struct node *root, uint8_t *arr) {
    assert(root);
    assert(arr);

    int8_t off = -1;
    traverse_tree(ch, root, 0, arr, &off);

    return off;
}

/*
 * Encode all the bytes input file, and write it to the
 * output file. The return value is the number of bytes
 * written to the output file.
 *
 * The Huffman code for each byte is computed and the bits
 * are packed together until the size of a byte is reached.
 * The byte is then written to the output file.
 */
uint64_t encode(FILE *ifile, struct node *root, FILE *ofile) {
    uint8_t *arr, shift, chunk, code;
    uint32_t th;
    uint64_t nr_bytes = 0;
    int8_t i, off, file_ch;

    assert(root);
    assert(ifile);
    assert(ofile);

    th = tree_height(root);
    assert(th > 0);

    /* Temporary array to store the Huffman code. */
    arr = calloc(th, sizeof(uint8_t));
    shift = 0;

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
                nr_bytes++;

                /* Reset the temporary bit buffer and shift count. */
                code = 0;
                shift = 0;
            }
        }
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
            nr_bytes++;

            code = 0;
            shift = 0;
        }
    }

    free(arr);

    return nr_bytes;
}

/*
 * Decode all the bytes in the encoded file (until the pseudo-EOF
 * byte is reached) and write them to the output file. The return
 * value is the number of bytes decoded.
 *
 * This function should be called only after reading the file
 * headers (which contains the byte-to-frequency map), and the
 * file offset is at the start of the first decodable byte.
 *
 * Decoding is done by extracting bits from each byte read from
 * the file. After reaching a leaf node by walking the Huffman
 * tree (depending on the "0" -- left -- or "1" -- right--) from
 * the bits. The resultant byte obtained from the tree traversal
 * is written to the output file.
 */
uint64_t decode(FILE *ifile, struct node *root, FILE *ofile) {
    uint64_t nr_bytes = 0;
    uint8_t shift, chunk, mask;
    int16_t file_ch;
    struct node *branch;

    assert(root);
    assert(ifile);
    assert(ofile);

    shift = 0;
    branch = root;

    /*
     * Extract the most significant bit of the byte.
     * This represents the root node of the tree.
     */
    mask = 0x1U << (MAX_INT_BUF_BITS - 1);

    file_ch = fgetc(ifile);
    if (file_ch == EOF)
        return 1;

    // chunk = (uint8_t)file_ch;
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
            if (branch->data.ch == PSEUDO_NULL_BYTE)
                break;

            fputc(branch->data.ch, ofile);
            nr_bytes++;

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

            // chunk = (uint8_t)file_ch;

            /* Reset the shift counter. */
            shift = 0;
        }
    }

    return 0;
}

/* All the things happen here. */
int main(int argc, char *argv[]) {
    uint32_t map_sz, i;
    struct node *head = NULL;
    struct map *fmap = NULL;
    FILE *ifile, *ofile;
    char *ifpath, *ofpath;
    int16_t arg, enc = 1;

    while ((arg = getopt(argc, argv, "edi:o:h?")) > 0) {
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
        assert(map_sz > 0);

        /* Build the priority queue. */
        head = make_queue(fmap, map_sz);
        assert(head);

        /* Make the queue into a tree. */
        make_tree(&head);
        rewind(ifile);

        /*
         * Write headers to the encoded output file:
         *
         *  +----------+------------------ >   < +-------------------- >   < -+
         *  | map-size | map-entries [...] >...< | encoded-bytes [...] >...<  |
         *  +----------+------------------ >   < +-------------------- >   < -+
         *
         */
        fwrite(&map_sz, sizeof(uint32_t), 1, ofile);
        for (i = 0; i < map_sz; i++)
            fwrite(&fmap[i], sizeof(struct map), 1, ofile);

        /* Write the encoded bytes to the file. */
        encode(ifile, head, ofile);
    } else {
        /* Decoding. */

        /* Read the file headers. */
        fread(&map_sz, sizeof(uint32_t), 1, ifile);
        assert(map_sz > 0);

        fmap = calloc(map_sz, sizeof(struct map));
        for (i = 0; i < map_sz; i++)
            fread(&fmap[i], sizeof(struct map), 1, ifile);

        /* Build the queue, and the tree from the headers. */
        head = make_queue(fmap, map_sz);
        make_tree(&head);

        /* Decode the file and write to the output file. */
        decode(ifile, head, ofile);
    }

    /* The queue can be free-d instead too. */
    nuke_tree(head);

    free(fmap);
    fclose(ifile);
    fclose(ofile);

    return 0;
}
