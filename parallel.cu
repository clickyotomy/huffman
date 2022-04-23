#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "huffman.h"

__device__ int16_t dev_tree_buff_off_incr(uint8_t *sh) {
    if (*sh < MAX_INT_BUF_BITS - 0x1U) {
        *sh++;
        return 0; 
    } else {
        *sh = 0;
        return 1;
    }
}

__device__ uint8_t dev_tree_leaf(struct node *node) {
    return (!node->left && !node->right);
}

__device__ struct node *inflate_dev_tree(uint8_t *buf, uint16_t eoff,
                                         uint8_t esh, uint16_t *doff,
                                         uint8_t *dsh) {
    uint8_t sh = MAX_INT_BUF_BITS, bit, mask, i;
    struct node *n;

    if (*doff >= eoff)
        return NULL;

    cudaMalloc((void *)n, sizeof(struct node));
    mask = 0x1U << (MAX_INT_BUF_BITS - 1);

    if (*doff == (eoff - 1))
        sh = esh;

    bit = ((buf[*doff] << *dsh) & mask) > 0;
    printf("bit\n");
    *doff += dev_tree_buff_off_incr(dsh);
    printf("bit: %d\n", *doff);

    if (bit) {
        n->left = NULL;
        n->right = NULL;

        for (i = 0; i < sh; i++) {
            bit = ((buf[*doff] << *dsh) & mask) > 0;
            printf("loop: %d\n", i );
            *doff += dev_tree_buff_off_incr(dsh);

            n->data.ch <<= 1;
            n->data.ch = n->data.ch | bit;
        }

        printf("dec: %c\n", n->data.ch);
        return n;
    }

    n->data.ch = PSEUDO_TREE_BYTE;
    printf("inflate_dev_tree-r1\n");
    n->left = inflate_dev_tree(buf, eoff, esh, doff, dsh);
    printf("inflate_dev_tree-r2\n");
    n->right = inflate_dev_tree(buf, eoff, esh, doff, dsh);

    return n;
}

__global__ void kern_decode_tree(uint8_t *buf, uint16_t eoff, uint8_t esh,
                                 struct node **dev_tree) {
    uint8_t dsh = 0;
    uint16_t doff = 0;

    // cudaMalloc((void **)&dsh, sizeof(uint64_t));
    // cudaMalloc((void **)&doff, sizeof(uint64_t));
    // cudaMemset((void *)&dsh, 0, sizeof(uint64_t));
    // cudaMemset((void *)&doff, 0, sizeof(uint64_t));

    for (int i = 0; i < eoff; i++) {
        printf("tbuf[%d]: 0x%hx\n", i, buf[i]);
    }
    
    *dev_tree = inflate_dev_tree(buf, eoff, esh, &doff, &dsh);
}

__global__ void kern_decode(uint8_t *ifile, uint64_t nr_en_bytes,
                            struct node *root, uint8_t *ofile,
                            uint64_t *nr_rd_bytes, uint64_t *nr_wr_bytes) {
    uint8_t shift = 0, chunk, mask;
    uint64_t nr_rbytes = 0, nr_wbytes = 0;
    int16_t file_ch;
    struct node *branch;

    branch = root;
    mask = 0x1U << (MAX_INT_BUF_BITS - 1);
    file_ch = ifile[nr_rbytes];

    if (file_ch == EOF)
        goto ret;

    nr_rbytes++;

    while (1) {
        chunk = (uint8_t)file_ch << shift;
        shift++;

        branch = (chunk & mask) ? branch->right : branch->left;

        if (dev_tree_leaf(branch)) {
            if (branch->data.ch == PSEUDO_NULL_BYTE && nr_rbytes >= nr_en_bytes)
                break;

            ofile[nr_wbytes] = branch->data.ch;
            printf("byte: %c\n", ofile[nr_wbytes]);
            nr_wbytes++;

            branch = root;
        }

        if (shift >= MAX_INT_BUF_BITS) {
            nr_rbytes++;

            if (nr_rbytes >= nr_en_bytes)
                break;

            file_ch = ifile[nr_rbytes];
            shift = 0;
        }
    }

ret:
    *nr_rd_bytes = nr_rbytes;
    *nr_wr_bytes = nr_wbytes;
}

extern "C" void dev_trampoline(FILE *ifile, struct meta *fmeta, FILE *ofile,
                    uint64_t *nr_rd_bytes, uint64_t *nr_wr_bytes) {
    uint8_t *ibuf = NULL, *obuf = NULL, *tbuf = NULL, *dev_ibuf = NULL,
            *dev_obuf = NULL, *dev_tbuf = NULL;
    uint64_t nr_bits, *dev_nr_rd_bytes, *dev_nr_wr_bytes;
    struct node *dev_tree_root;

    assert(ifile);
    assert(fmeta);
    assert(ofile);

    assert(nr_rd_bytes);
    assert(nr_wr_bytes);

    nr_bits = (fmeta->nr_enc_bytes * MAX_INT_BUF_BITS) + fmeta->tree_lb_sh_pos;

    tbuf = (uint8_t *)calloc(fmeta->nr_tree_bytes, sizeof(uint8_t));
    ibuf = (uint8_t *)calloc(fmeta->nr_enc_bytes, sizeof(uint8_t));
    obuf = (uint8_t *)calloc(fmeta->nr_src_bytes, sizeof(uint8_t));

    assert(tbuf);
    assert(ibuf);
    assert(obuf);

    cudaMalloc((void **)&dev_tbuf, fmeta->nr_tree_bytes * sizeof(uint8_t));
    cudaMalloc((void **)&dev_ibuf, fmeta->nr_enc_bytes * sizeof(uint8_t));
    cudaMalloc((void **)&dev_obuf, fmeta->nr_src_bytes * sizeof(uint8_t));

    cudaMalloc((void **)&dev_tree_root, sizeof(struct node));
    cudaMalloc((void **)&dev_nr_rd_bytes, sizeof(uint64_t));
    cudaMalloc((void **)&dev_nr_wr_bytes, sizeof(uint64_t));

    fread(tbuf, sizeof(uint8_t), (size_t)fmeta->nr_tree_bytes, ifile);
    fread(ibuf, sizeof(uint8_t), (size_t)fmeta->nr_enc_bytes, ifile);

    cudaMemcpy(dev_tbuf, tbuf, fmeta->nr_tree_bytes,
               cudaMemcpyHostToDevice);

    printf("launch 1\n");
    kern_decode_tree<<<1, 1>>>(dev_tbuf, fmeta->nr_tree_bytes,
                               fmeta->tree_lb_sh_pos, &dev_tree_root);

    cudaMemcpy(dev_ibuf, ibuf, sizeof(uint8_t) * fmeta->nr_enc_bytes,
               cudaMemcpyHostToDevice);

    // free(tbuf);
    // free(ibuf);

    cudaDeviceSynchronize();

    printf("launch 2\n");
    kern_decode<<<1, 1>>>(dev_ibuf, fmeta->nr_enc_bytes, dev_tree_root,
                          dev_obuf, dev_nr_rd_bytes, dev_nr_wr_bytes);
    cudaDeviceSynchronize();

    cudaMemcpy(obuf, dev_obuf, sizeof(uint8_t) * fmeta->nr_enc_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(nr_rd_bytes, dev_nr_rd_bytes, sizeof(uint8_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(nr_wr_bytes, dev_nr_wr_bytes, sizeof(uint8_t),
               cudaMemcpyDeviceToHost);

    printf("nr_wr_bytes: %d, %d\n", fmeta->nr_src_bytes, nr_wr_bytes);
    fwrite(obuf, sizeof(uint8_t), (size_t)nr_wr_bytes, ofile);
}
