#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <stdbool.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "huffman.h"

/*
 * Division modulo operations are awfully slow in CUDA.
 * This is a handy trick if "b" is a power of 2. From
 * the "Division Modulo Operations" section here. [1]
 *
 * [1]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
 */
#define MOD(a, b) ((a) & (b)-1)

#define UPDIV(x, y) (((x) + ((y)-1)) / (y))

#define UNIT_SIZE 8
#define SUBSEQ_LENGTH 4096
#define THREADS_PER_BLOCK 128
#define SUBSEQ_CONV 32

__device__ bool dev_blocks_synchronized;

struct sync_point {
    uint64_t last_word_bit;
    uint64_t num_symbols;
    uint64_t out_pos;
    bool sync;
};

__device__ __forceinline__ uint8_t dev_log2(uint32_t n) {
    uint8_t ret = 0;

    while (n >>= 1) {
        ret++;
    }

    return ret;
}

__device__ __forceinline__ struct lookup *
dev_lookup(struct lookup *tab, uint32_t tab_sz, uint32_t idx) {
    if (idx < tab_sz)
        return &tab[idx];

    return NULL;
}

__device__ __forceinline__ uint64_t
decode_sym(uint64_t start_bit_off, struct lookup *tab, uint32_t tab_sz,
           const uint8_t *bytes, uint64_t end_bit_off, uint8_t *ret) {
    uint8_t byte, mask, bshft = 0, cshft = 0, mshft;
    uint64_t curr_bit_off, unit_bit_off, curr_unit, bit_count = 0;
    uint32_t chunk = 0;
    struct lookup *ent = NULL;

    assert(ret);

    /*
     * Copy the offset of the current bit, and calculate the
     * byte offset of the current unit, and the bit offset of
     * the current bit relative to the unit.
     */
    curr_bit_off = start_bit_off;
    curr_unit = start_bit_off / UNIT_SIZE;
    unit_bit_off = MOD(start_bit_off, UNIT_SIZE);

    mask = 0x1U << (UNIT_SIZE - 1);
    mshft = dev_log2(tab_sz);

    byte = bytes[curr_unit];
    bshft = unit_bit_off;
    curr_unit++;

    while (curr_bit_off + cshft < end_bit_off) {
        chunk <<= 1;
        chunk |= ((byte << bshft) & mask) ? 0x1U : 0x0U;

        bshft++;
        cshft++;

        if (cshft >= mshft) {
            ent = dev_lookup(tab, tab_sz, chunk);
            assert(ent);

            *ret = ent->ch;
            bit_count = ent->off;
            break;
        }

        /* Overflow into the next byte. */
        if (bshft >= UNIT_SIZE) {
            byte = bytes[curr_unit];
            curr_unit++;
            bshft = 0;
        }
    }

    return bit_count;
}

__device__ void decode_subsequence(struct lookup *tab, uint32_t tab_sz,
                                   const uint8_t *bit_string,
                                   uint64_t subseq_start,
                                   uint64_t total_nr_bits,
                                   uint64_t *out_num_symbols,
                                   uint64_t *out_last_offset, bool print) {

    /* start decoding from the start of the subsequence */
    uint64_t offset = subseq_start;
    uint64_t subseq_end = offset + SUBSEQ_LENGTH;
    subseq_end = subseq_end < total_nr_bits ? subseq_end : total_nr_bits;
    uint64_t prev_off = offset;
    uint64_t num_symbols = 0, num_bits = 0;
    uint8_t c;

    while (offset < subseq_end) {
        num_bits =
            decode_sym(offset, tab, tab_sz, bit_string, total_nr_bits, &c);

        if (num_bits == 0) {
            break;
        }

        if (print)
            printf("decode: %c\n", (char)c);

        prev_off = offset;
        offset += num_bits;
        num_symbols++;
    }

    *out_num_symbols = num_symbols;
    *out_last_offset = prev_off;
}

__device__ void
decode_subsequence_write(struct lookup *tab, uint32_t tab_sz,
                         const uint8_t *bit_string, uint64_t subseq_start,
                         uint64_t total_nr_bits, uint64_t *out_num_symbols,
                         uint64_t *out_last_offset, uint8_t *output) {

    /* start decoding from the start of the subsequence */
    uint64_t offset = subseq_start;
    uint64_t subseq_end = offset + SUBSEQ_LENGTH;
    subseq_end = subseq_end < total_nr_bits ? subseq_end : total_nr_bits;
    uint64_t prev_off = offset;
    uint64_t num_symbols = 0, num_bits = 0;
    uint8_t c;
    uint64_t out_pos = 0;

    while (offset < subseq_end) {
        num_bits =
            decode_sym(offset, tab, tab_sz, bit_string, total_nr_bits, &c);

        if (num_bits == 0) {
            break;
        }

        output[out_pos] = c;
        prev_off = offset;
        offset += num_bits;
        num_symbols++;
        out_pos++;
    }

    *out_num_symbols = num_symbols;
    *out_last_offset = prev_off;
}

__global__ void phase1_decode_subseq(uint64_t total_nr_bits,
                                     uint32_t total_num_subsequences,
                                     const uint8_t *bit_string,
                                     struct lookup *tab, uint32_t tab_sz,
                                     struct sync_point *sync_points) {

    const size_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    uint64_t current_subsequence = gid;
    uint64_t subsequence_offset = gid * SUBSEQ_LENGTH;
    uint64_t cur_pos = subsequence_offset;

    uint32_t subsequences_processed = 0;
    bool synchronised_flag = false;

    while (subsequences_processed < blockDim.x) {
        if (!synchronised_flag &&
            current_subsequence < total_num_subsequences) {
            uint64_t num_symbols;
            uint64_t last_word_bit;
            decode_subsequence(tab, tab_sz, bit_string, cur_pos, total_nr_bits,
                               &num_symbols, &last_word_bit, false);

            if (subsequences_processed == 0) {
                sync_points[current_subsequence].sync = 1;
                sync_points[current_subsequence].last_word_bit = last_word_bit;
                sync_points[current_subsequence].num_symbols = num_symbols;
            } else {
                struct sync_point sync_point = sync_points[current_subsequence];
                sync_point.num_symbols = num_symbols;

                // if sync point detected
                if (sync_point.last_word_bit == last_word_bit) {
                    sync_point.sync = 1;
                    synchronised_flag = true;
                } else {
                    // correct erroneous position data
                    sync_point.last_word_bit = last_word_bit;
                    sync_point.sync = 0;
                }
                sync_points[current_subsequence] = sync_point;
            }
            cur_pos = last_word_bit;
        }
        ++current_subsequence;
        ++subsequences_processed;

        __syncthreads();
    }
}

__global__ void phase2_synchronise_blocks(
    uint64_t total_nr_bits, uint64_t total_num_subsequences,
    uint64_t num_blocks, const uint8_t *bit_string, struct lookup *tab,
    uint32_t tab_sz, struct sync_point *sync_points, bool *block_synchronised) {

    const uint64_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    const uint64_t num_of_seams = num_blocks - 1;

    if (gid < num_of_seams) {
        // jump to first sequence of the block
        uint64_t current_subsequence = (gid + 1) * blockDim.x;

        // search for synchronised sequences at the end of previous block

        // number of symbols found in this subsequence
        uint64_t num_symbols = 0;
        uint64_t subsequences_processed = 0;
        bool synchronised_flag = false;

        while (subsequences_processed < blockDim.x) {
            if (current_subsequence < total_num_subsequences) {
                struct sync_point sync_point =
                    sync_points[current_subsequence - 1];
                uint64_t cur_pos = sync_point.last_word_bit;

                if (!synchronised_flag) {
                    uint64_t last_word_bit;
                    decode_subsequence(tab, tab_sz, bit_string, cur_pos,
                                       total_nr_bits, &num_symbols,
                                       &last_word_bit, false);
                    sync_point = sync_points[current_subsequence];
                    sync_point.num_symbols = num_symbols;

                    // if sync point detected
                    if (sync_point.last_word_bit == last_word_bit) {
                        sync_point.sync = 1;
                        block_synchronised[gid + 1] = true;
                        synchronised_flag = true;
                    } else {
                        // correct erroneous position data
                        sync_point.last_word_bit = last_word_bit;
                        sync_point.sync = 0;
                        block_synchronised[gid + 1] = false;
                    }
                    sync_points[current_subsequence] = sync_point;
                    cur_pos = last_word_bit;
                }
            }

            ++current_subsequence;
            ++subsequences_processed;

            __syncthreads();
        }
    }
}

__global__ void phase2_seq_sync_chk(bool *sequences_sync,
                                    size_t num_sequences) {
    const size_t num_blocks = num_sequences - 1;

    dev_blocks_synchronized = true;
    for (size_t i = 1; i < num_blocks; i++) {
        if (sequences_sync[i] == 0) {
            dev_blocks_synchronized = false;
            break;
        }
    }
}

__global__ void phase3_copy_to_num_symbols(uint64_t total_num_subsequences,
                                           const struct sync_point *sync_points,
                                           uint64_t *num_symbols) {

    uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid < total_num_subsequences) {
        num_symbols[gid] = sync_points[gid].num_symbols;
    }
}

__global__ void phase3_copy_to_sync_points(uint64_t total_num_subsequences,
                                           struct sync_point *sync_points,
                                           uint64_t *num_symbols) {

    uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid < total_num_subsequences) {
        sync_points[gid].out_pos = num_symbols[gid];
    }
}

/* Returns total number of characters in total */
uint64_t phase3(struct sync_point *sync_points, size_t num_subsequences,
                size_t num_sequences) {
    uint64_t *num_symbols;
    uint64_t total_num_chars;

    cudaMalloc((void **)&num_symbols, num_subsequences * sizeof(uint64_t));
    phase3_copy_to_num_symbols<<<num_sequences, THREADS_PER_BLOCK>>>(
        num_subsequences, sync_points, num_symbols);
    thrust::device_ptr<uint64_t> dev_ptr =
        thrust::device_pointer_cast(num_symbols);
    thrust::exclusive_scan(dev_ptr, dev_ptr + num_subsequences, dev_ptr);
    phase3_copy_to_sync_points<<<num_sequences, THREADS_PER_BLOCK>>>(
        num_subsequences, sync_points, num_symbols);

    cudaMemcpy(&total_num_chars, num_symbols + (num_subsequences - 1),
               sizeof(uint64_t), cudaMemcpyDeviceToHost);

    struct sync_point last_sync_point;
    cudaMemcpy(&last_sync_point, sync_points + (num_subsequences - 1),
               sizeof(struct sync_point), cudaMemcpyDeviceToHost);

    total_num_chars += last_sync_point.num_symbols;
    cudaFree(num_symbols);

    return total_num_chars;
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

__global__ void phase4_decode_write_output(uint64_t total_nr_bits,
                                           uint64_t total_num_subsequences,
                                           const uint8_t *bit_string,
                                           struct lookup *tab, uint32_t tab_sz,
                                           struct sync_point *sync_points,
                                           uint8_t *output_buf) {

    const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if (gid < total_num_subsequences) {
        uint32_t cur_pos = 0;
        uint32_t out_pos = 0;

        if (gid > 0) {
            struct sync_point sync_point = sync_points[gid - 1];
            cur_pos = sync_point.last_word_bit;
            out_pos = sync_point.out_pos;
        }

        uint64_t last_word_bit;
        uint64_t num_symbols;
        decode_subsequence_write(tab, tab_sz, bit_string, cur_pos,
                                 total_nr_bits, &num_symbols, &last_word_bit,
                                 output_buf + out_pos);
    }
}

extern "C" void dev_trampoline(FILE *ifile, struct meta *fmeta,
                               struct lookup *tab, uint32_t tab_sz, FILE *ofile,
                               uint64_t *nr_rd_bytes, uint64_t *nr_wr_bytes) {

    uint8_t *ibuf = NULL, *obuf = NULL, *dev_ibuf = NULL, *dev_obuf = NULL;
    uint64_t total_nr_bits, *dev_nr_rd_bytes, *dev_nr_wr_bytes;
    size_t num_subseq, num_sequences, conv = 0;
    struct lookup *dev_tab;
    struct sync_point *dev_sync_points;
    bool *blocks_synchronized, *dev_sequences_sync;
    cudaEvent_t start, stop;
    float p1, p2, p3;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    total_nr_bits = (fmeta->nr_enc_bytes * UNIT_SIZE);
    num_subseq = UPDIV(total_nr_bits, (SUBSEQ_LENGTH * UNIT_SIZE));
    num_sequences = UPDIV(num_subseq, THREADS_PER_BLOCK);

    /* Standard asserts. */
    assert(fmeta);
    assert(tab);
    assert(tab_sz);
    assert(ifile);
    assert(ofile);
    assert(nr_rd_bytes);
    assert(nr_wr_bytes);

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

    cudaMalloc((void **)&dev_sync_points,
               sizeof(struct sync_point) * num_subseq);
    assert(dev_sync_points);
    cudaMemset(dev_sync_points, 0, sizeof(struct sync_point) * num_subseq);

    /* Phase 1. */
    {
        cudaEventRecord(start);

        phase1_decode_subseq<<<num_sequences, THREADS_PER_BLOCK>>>(
            total_nr_bits, num_subseq, dev_ibuf, dev_tab, tab_sz,
            dev_sync_points);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&p1, start, stop);
    }

    // sequences_sync = (bool *)calloc(num_sequences, sizeof(bool));
    // assert(sequences_sync);

    cudaMalloc((void **)&dev_sequences_sync, sizeof(bool) * num_sequences);
    assert(dev_sequences_sync);

    /* Phase 2. */
    {
        cudaEventRecord(start);

        blocks_synchronized = (bool *)calloc(1, sizeof(bool));
        *blocks_synchronized = false;
        while (!*blocks_synchronized && conv < SUBSEQ_CONV) {
            phase2_synchronise_blocks<<<num_sequences, THREADS_PER_BLOCK>>>(
                total_nr_bits, num_subseq, num_sequences, dev_ibuf, dev_tab,
                tab_sz, dev_sync_points, dev_sequences_sync);
            cudaDeviceSynchronize();

            phase2_seq_sync_chk<<<1, 1>>>(dev_sequences_sync, num_sequences);
            cudaDeviceSynchronize();

            cudaMemcpyFromSymbol(blocks_synchronized, dev_blocks_synchronized,
                                 sizeof(bool), 0, cudaMemcpyDeviceToHost);
            conv++;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&p2, start, stop);
    }

    /* Phases 3. */
    {
        cudaEventRecord(start);

        phase3(dev_sync_points, num_subseq, num_sequences);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&p3, start, stop);
    }

    /* Phase 4: We will not be timing the file writes. */
    {
        phase4_decode_write_output<<<num_sequences, THREADS_PER_BLOCK>>>(
            total_nr_bits, num_subseq, dev_ibuf, dev_tab, tab_sz,
            dev_sync_points, dev_obuf);
        cudaDeviceSynchronize();

        cudaFree(dev_ibuf);
        cudaFree(dev_tab);

        /* Copy the device output file buffer to host. */
        obuf = (uint8_t *)calloc(fmeta->nr_src_bytes, sizeof(uint8_t));
        assert(obuf);
        cudaMemcpy(obuf, dev_obuf, sizeof(uint8_t) * fmeta->nr_src_bytes,
                   cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        for (uint64_t i = 0; i < fmeta->nr_src_bytes; i++) {
            fwrite(&obuf[i], sizeof(uint8_t), 1, ofile);
        }

        *nr_rd_bytes = fmeta->nr_enc_bytes;
        *nr_wr_bytes = fmeta->nr_src_bytes;

        free(obuf);
        cudaFree(dev_obuf);
    }

    printf("decode: %0.3fms {p1: %0.3fms, p2: %0.3fms, p3: %0.3fms}\n",
           p1 + p2 + p3, p1, p2, p3);
}
