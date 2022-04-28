#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "huffman.h"

/* Helper to check and print runtime errors. */
// static inline cudaError_t chkCuda(cudaError_t result) {
//     if (result != cudaSuccess) {
//         fprintf(stderr, "CUDA Runtime Error: %s\n",
//         		cudaGetErrorString(result));
//         assert(result == cudaSuccess);
//     }
//     return result;
// }

__device__ uint8_t dev_log2(uint32_t n) {
    uint8_t ret = 0;

    while (n >>= 1) {
        ret++;
    }

    return ret;
}

__device__ struct lookup *dev_lookup(struct lookup *tab, uint32_t tab_sz,
                                     uint32_t idx) {
    if (idx < tab_sz)
        return &tab[idx];

    return NULL;
}

__global__ void kern_decode(uint8_t *ifile, uint64_t nr_en_bytes,
                            struct lookup *tab, uint32_t tab_sz, uint8_t *ofile,
                            uint64_t *nr_rd_bytes, uint64_t *nr_wr_bytes) {

    uint8_t file_ch, mask, bshft = 0, cshft = 0, mshft = 0;
    uint64_t nr_rbytes = 0, nr_wbytes = 0;
    uint32_t chunk = 0;
    struct lookup *ent = NULL;

    assert(tab);
    assert(tab_sz > 0 && tab_sz <= MAX_LOOKUP_TAB_LEN);
    assert(ifile);
    assert(ofile);
    assert(nr_rd_bytes);
    assert(nr_wr_bytes);

    mshft = dev_log2(tab_sz);
    mask = 0x1U << (MAX_INT_BUF_BITS - 1);

    file_ch = ifile[0];
    // printf("file_ch: 0x%x\n", file_ch);
    nr_rbytes++;

    while (1) {
        chunk <<= 1;
        chunk |= ((file_ch << bshft) & mask) ? 0x1U : 0x0U;

        bshft++;
        cshft++;

        if (cshft >= mshft) {
            ent = dev_lookup(tab, tab_sz, chunk);
            assert(ent);

            if (ent->ch == PSEUDO_NULL_BYTE && nr_rbytes >= nr_en_bytes)
                goto ret;

            ofile[nr_wbytes] = (uint8_t)ent->ch;
            nr_wbytes++;

            chunk = chunk & ~((~0x0UL) << (mshft - (ent->off)));
            cshft = mshft - ent->off;
        }

        if (bshft >= MAX_INT_BUF_BITS) {
            file_ch = ifile[nr_rbytes];
            nr_rbytes++;
            bshft = 0;
        }
    }

ret:
    nr_rd_bytes[0] = nr_rbytes;
    nr_wr_bytes[0] = nr_wbytes;
}

extern "C" void dev_trampoline(FILE *ifile, struct meta *fmeta,
                               struct lookup *tab, uint32_t tab_sz, FILE *ofile,
                               uint64_t *nr_rd_bytes, uint64_t *nr_wr_bytes) {

    uint8_t *ibuf = NULL, *obuf = NULL, *dev_ibuf = NULL, *dev_obuf = NULL;
    uint64_t /*nr_bits,*/ *dev_nr_rd_bytes, *dev_nr_wr_bytes;
    struct lookup *dev_tab;

    /* Standard asserts. */
    assert(fmeta);
    assert(tab);
    assert(tab_sz);
    assert(ifile);
    assert(ofile);
    assert(nr_rd_bytes);
    assert(nr_wr_bytes);

    /* Calculate the number of bits. */
    // nr_bits = (fmeta->nr_enc_bytes * MAX_INT_BUF_BITS) +
    // fmeta->tree_lb_sh_pos;

    /* Copy the table to the device. */
    cudaMalloc((void **)&dev_tab, sizeof(struct lookup) * tab_sz);
    cudaMemset(dev_tab, 0, sizeof(uint8_t) * tab_sz);
    assert(dev_tab);

    cudaMemcpy(dev_tab, tab, sizeof(struct lookup) * tab_sz,
               cudaMemcpyHostToDevice);

    /* Read the encoded file content into the host input file buffer. */
    ibuf = (uint8_t *)calloc(fmeta->nr_enc_bytes, sizeof(uint8_t));
    assert(ibuf);

    fread(ibuf, sizeof(uint8_t), (size_t)fmeta->nr_enc_bytes, ifile);

    cudaMalloc((void **)&dev_ibuf, sizeof(uint8_t) * fmeta->nr_enc_bytes);
    assert(dev_ibuf);

    cudaMemset(dev_ibuf, 0, sizeof(uint8_t) * fmeta->nr_enc_bytes);
    cudaMemcpy(dev_ibuf, ibuf, sizeof(uint8_t) * fmeta->nr_enc_bytes,
               cudaMemcpyHostToDevice);
    free(ibuf);

    /* Allocate memory for decoding. */
    cudaMalloc((void **)&dev_nr_rd_bytes, sizeof(uint64_t));
    assert(dev_nr_rd_bytes);
    cudaMemset(dev_nr_rd_bytes, 0, sizeof(uint64_t));

    cudaMalloc((void **)&dev_nr_wr_bytes, sizeof(uint64_t));
    assert(dev_nr_wr_bytes);
    cudaMemset(dev_nr_wr_bytes, 0, sizeof(uint64_t));

    cudaMalloc((void **)&dev_obuf, sizeof(uint8_t) * fmeta->nr_src_bytes);
    assert(dev_obuf);
    cudaMemset(dev_obuf, 0, sizeof(uint8_t) * fmeta->nr_src_bytes);

    /* Decode the file. */
    kern_decode<<<1, 1>>>(dev_ibuf, fmeta->nr_enc_bytes, dev_tab, tab_sz,
                          dev_obuf, dev_nr_rd_bytes, dev_nr_wr_bytes);
    cudaDeviceSynchronize();
    cudaFree(dev_ibuf);
    cudaFree(dev_tab);

    /* Copy the read/write stats. */
    cudaMemcpy(nr_rd_bytes, dev_nr_rd_bytes, sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(nr_wr_bytes, dev_nr_wr_bytes, sizeof(uint64_t),
               cudaMemcpyDeviceToHost);

    /* Copy the device output file buffer to host. */
    obuf = (uint8_t *)calloc(*nr_wr_bytes, sizeof(uint8_t));
    assert(obuf);
    cudaMemcpy(obuf, dev_obuf, sizeof(uint8_t) * (*nr_wr_bytes),
               cudaMemcpyDeviceToHost);

    /* Write the decoded content to the output file. */
    fwrite(obuf, sizeof(uint8_t), *nr_wr_bytes, ofile);
    free(obuf);
    cudaFree(dev_obuf);
}
