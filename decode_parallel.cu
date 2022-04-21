#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include "decode_parallel.h"
#include "queue.h"
#include "tree.h"

/*
 * Division modulo operations are awfully slow in CUDA.
 * This is a handy trick if "b" is a power of 2.
 * Ref: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html,
 *      under "Division Modulo Operations"
 */
#define MOD(a, b) ((a) & (b)-1)


/* Bit string is on the device */
char * get_bit_string_device(FILE *ifile, uint64_t rd_bits) {
    assert(ifile);

    int num_bytes = (rd_bits + 0x7U) / 0x8U;

    // allocate memory to contain the whole file:
    char * buffer_host = (char*) malloc (sizeof(char)*num_bytes);
    // copy the file into the buffer:
    fread(buffer_host, 1, num_bytes, ifile);

    char * buffer_device;
    cudaMalloc(&buffer_device, num_bytes);
    cudaMemcpy(buffer_device, buffer_host, num_bytes, cudaMemcpyHostToDevice);

    free(buffer_host);
    return buffer_device;
}


void make_tree_arr_helper(tree_arr_node_t *tree_arr, node_t *head, int i) {
//    printf("helper: %d\n", i);
    tree_arr[i].ch = 0;

    if (tree_leaf(head)) {
        tree_arr[i].ch = head->data.ch;
        return;
    }
    make_tree_arr_helper(tree_arr, head->left, 2*i);
    make_tree_arr_helper(tree_arr, head->right, 2*i + 1);
}
/*
 * Build a Huffmann encoding tree from a priority queue.
 * This tree is on the device.
 */
tree_arr_node_t *make_tree_device(struct node *head, uint32_t tree_depth) {
    uint64_t array_size = 0x1U << (tree_depth);

    tree_arr_node_t * tree_arr_host = (tree_arr_node_t *) calloc(array_size, sizeof(tree_arr_node_t ));
    printf("size: %d\n", array_size);
    make_tree_arr_helper(tree_arr_host, head, 1);

    tree_arr_node_t * tree_arr_device;
    cudaMalloc(&tree_arr_device, array_size * sizeof(tree_arr_node_t ));
    cudaMemcpy(tree_arr_device, tree_arr_host, array_size * sizeof(tree_arr_node_t), cudaMemcpyHostToDevice);
    free(tree_arr_host);

    return tree_arr_device;
}

/* Return true if the node is the leaf of a tree. */
__device__ __inline__
bool is_tree_arr_leaf(tree_arr_node_t *arr) {
    assert(arr);
    return arr->ch != 0;
}

/*
 *
 *  Returns the number of bits taken to decode the current symbol
 */
#define UNIT_SIZE 8
__device__ __inline__
uint64_t decode_one_symb(uint64_t bit_offset,  tree_arr_node_t *decode_root, const char *bits, uint64_t end_bit_offset,
                         char *c){

    uint64_t cur_bit_offset = bit_offset;
    int i = 1;
    while (cur_bit_offset < end_bit_offset) {
        uint64_t cur_unit = cur_bit_offset / UNIT_SIZE;
        uint64_t cur_bit_in_unit = (UNIT_SIZE - 1) - (cur_bit_offset % UNIT_SIZE);
        uint64_t cur_bit = (bits[cur_unit] & (0x1U << cur_bit_in_unit));
        /* Determine which branch to take depending on the extracted bit. */
        i = cur_bit ? 2 * i + 1: 2 * i;

        /*
         * If we reached the leaf node, we have decoded a byte;
         * write it to the output file.
         */
        if (is_tree_arr_leaf(decode_root + i)) {
            *c = decode_root[i].ch;
            /* Let's skip out on the writing first */
//            fputc(branch->data.ch, ofile);
            break;
        }

        cur_bit_offset++;
    }

    return cur_bit_offset - bit_offset + 1;
}

__device__ __inline__
void decode_subsequence_write(tree_arr_node_t *decode_tree, const char *bit_string, uint64_t subseq_start, uint64_t total_nr_bits,
                        int *out_num_symbols, uint64_t *out_last_offset, char *output) {

    /* start decoding from the start of the subsequence */
    uint64_t offset = subseq_start;
    uint64_t subseq_end = offset + SUBSEQ_LENGTH;
    subseq_end = subseq_end < total_nr_bits ? subseq_end : total_nr_bits - 1;
    uint64_t prev_num_bits = 0;
    int num_symbols = 0;
    int out_pos = 0;

    while (offset < subseq_end) {
        char c;
        uint64_t num_bits = decode_one_symb(offset,  decode_tree, bit_string, total_nr_bits, &c);
        output[out_pos] = c;
        offset += num_bits;
        prev_num_bits = num_bits;
        num_symbols++;
        out_pos++;
    }

    *out_num_symbols = num_symbols;
    *out_last_offset = offset - prev_num_bits;

}

__device__ __inline__
void decode_subsequence(tree_arr_node_t *decode_tree, const char *bit_string, uint64_t subseq_start, uint64_t total_nr_bits,
                        int *out_num_symbols, uint64_t *out_last_offset, bool print) {

    /* start decoding from the start of the subsequence */
    uint64_t offset = subseq_start;
    uint64_t subseq_end = offset + SUBSEQ_LENGTH;
    subseq_end = subseq_end < total_nr_bits ? subseq_end : total_nr_bits;
    uint64_t prev_num_bits = 0;
    int num_symbols = 0;

    while (offset < subseq_end) {
        char c;
        uint64_t num_bits = decode_one_symb(offset,  decode_tree, bit_string, total_nr_bits, &c);
        if (print) {
            printf("%c\n", c);
        }
        offset += num_bits;
        prev_num_bits = num_bits;
        num_symbols++;
    }

    *out_num_symbols = num_symbols;
    *out_last_offset = offset - prev_num_bits;

}

__global__ void phase1_decode_subseq(
        uint64_t total_nr_bits,
        uint32_t total_num_subsequences,
        const char *bit_string,
        tree_arr_node_t *decode_tree,
        sync_point_t * sync_points) {

    const size_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    int current_subsequence = gid;
    int subsequence_offset = gid * SUBSEQ_LENGTH;
    int cur_pos = subsequence_offset;

    uint32_t subsequences_processed = 0;
    bool synchronised_flag = false;

    while(subsequences_processed < blockDim.x) {
        if (!synchronised_flag && current_subsequence < total_num_subsequences) {
            int num_symbols;
            uint64_t last_word_bit;
            decode_subsequence(decode_tree, bit_string, cur_pos, total_nr_bits, &num_symbols, &last_word_bit,
                               false);

            if (subsequences_processed == 0) {
                sync_points[current_subsequence].sync = 1;
                sync_points[current_subsequence].last_word_bit = last_word_bit;
                sync_points[current_subsequence].num_symbols = num_symbols;
            } else {
                sync_point_t sync_point = sync_points[current_subsequence];
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
        uint64_t total_nr_bits,
        uint64_t total_num_subsequences,
        int num_blocks,
        const char *bit_string,
        tree_arr_node_t *decode_tree,
        sync_point_t * sync_points,
        bool *block_synchronised) {

    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int num_of_seams = num_blocks - 1;

    if(gid < num_of_seams) {
        // jump to first sequence of the block
        int current_subsequence = (gid + 1) * blockDim.x;

        // search for synchronised sequences at the end of previous block

        // number of symbols found in this subsequence
        int num_symbols = 0;
        int subsequences_processed = 0;
        bool synchronised_flag = false;

        while(subsequences_processed < blockDim.x) {
            if(current_subsequence < total_num_subsequences) {
                sync_point sync_point = sync_points[current_subsequence - 1];
                int cur_pos = sync_point.last_word_bit;

                if (!synchronised_flag) {
                    uint64_t last_word_bit;
                    decode_subsequence(decode_tree, bit_string, cur_pos, total_nr_bits, &num_symbols, &last_word_bit, false);
                    sync_point = sync_points[current_subsequence];
                    sync_point.num_symbols = num_symbols;

                    // if sync point detected
                    if (sync_point.last_word_bit == last_word_bit) {
                        sync_point.sync = 1;
                        block_synchronised[gid + 1] = 1;
                        synchronised_flag = true;
                    } else {
                        // correct erroneous position data
                        sync_point.last_word_bit = last_word_bit;
                        sync_point.sync = 0;
                        block_synchronised[gid + 1] = 0;
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

/*
 * Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

/* Do the {up,down}-sweep inner loops of "exclusive_sum". */
__global__ void kern_sweep(bool up, int N, int twod, int twod1, int *data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    /* Equivalent to "i += twod1" from the loop in the sequential version. */
    if (index < N && (MOD(index, twod1) == 0)) {
        /* The "up-sweep" part. */
        if (up) {
            data[index + twod1 - 1] += data[index + twod - 1];
            return;
        }

        /* The "down-sweep" part. */
        int tmp = data[index + twod - 1];
        data[index + twod - 1] = data[index + twod1 - 1];
        data[index + twod1 - 1] += tmp;
    }
}

/* Set the element at the last index to 0. */
__global__ void kern_zero(int N, int *data) { data[N - 1] = 0; }

void exclusive_scan(int *device_data, int length) {
    int blocks, N;

    N = nextPow2(length);
    blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int twod = 1; twod < N; twod *= 2) {
        int twod1 = twod * 2;
        kern_sweep<<<blocks, THREADS_PER_BLOCK>>>(true, N, twod, twod1,
                                                device_data);
    }

    kern_zero<<<1, 1>>>(N, device_data);

    for (int twod = (N / 2); twod >= 1; twod /= 2) {
        int twod1 = twod * 2;
        kern_sweep<<<blocks, THREADS_PER_BLOCK>>>(false, N, twod, twod1,
                                                device_data);
    }
}

__global__ void phase3_copy_to_num_symbols(
        uint64_t total_num_subsequences,
        const sync_point_t *sync_points,
        int *num_symbols) {

    uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        num_symbols[gid] = sync_points[gid].num_symbols;
    }
}

__global__ void phase3_copy_to_sync_points(
        uint64_t total_num_subsequences,
        sync_point_t *sync_points,
        int *num_symbols) {

    uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        sync_points[gid].out_pos = num_symbols[gid];
    }
}

/* Returns total number of characters in total */
int phase3(sync_point_t *sync_points, size_t num_subsequences, size_t num_sequences) {
    int *num_symbols;
    cudaMalloc((void **)&num_symbols, num_subsequences * sizeof(int));
    phase3_copy_to_num_symbols<<<num_sequences, THREADS_PER_BLOCK>>>(num_subsequences, sync_points, num_symbols);
    thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(num_symbols);
    // exclusive_scan(num_symbols, num_subsequences);
    // thrust::device_ptr<int> dev_ptr(num_symbols);
    thrust::exclusive_scan(dev_ptr, dev_ptr + num_subsequences, dev_ptr);
    phase3_copy_to_sync_points<<<num_sequences, THREADS_PER_BLOCK>>>(num_subsequences, sync_points, num_symbols);
    int total_num_chars;
    cudaMemcpy(&total_num_chars, num_symbols + (num_subsequences - 1), sizeof(int), cudaMemcpyDeviceToHost);
    sync_point_t last_sync_point;
    cudaMemcpy(&last_sync_point, sync_points + (num_subsequences - 1), sizeof(sync_point_t), cudaMemcpyDeviceToHost);
    total_num_chars += last_sync_point.num_symbols;
    cudaFree(num_symbols);
    return total_num_chars;
}




__global__ void phase4_decode_write_output(uint64_t total_nr_bits, uint64_t total_num_subsequences,
                                           const char *bit_string, tree_arr_node_t *decode_tree,
                                           sync_point_t * sync_points, char *output_buf) {

    const uint32_t gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {
        uint32_t cur_pos = 0;
        uint32_t out_pos = 0;

        if(gid > 0) {
            sync_point_t sync_point = sync_points[gid - 1];
            cur_pos = sync_point.last_word_bit;
            out_pos = sync_point.out_pos;
        }

        uint64_t last_word_bit;
        int num_symbols;
        decode_subsequence_write(decode_tree, bit_string, cur_pos, total_nr_bits, &num_symbols, &last_word_bit,
                           output_buf + out_pos);
    }
}


/* We assume that the decode_table and bit_string is malloc-ed in cuda */
void decode_cuda(uint64_t total_nr_bits, const char *bit_string, tree_arr_node_t *decode_table, FILE *ofile) {

    printf("total_nr_bits %d\n", total_nr_bits);
    size_t num_subseq = updivide(total_nr_bits, SUBSEQ_LENGTH);
    size_t num_sequences = updivide(num_subseq, THREADS_PER_BLOCK);

    /* Phase 1 */
    sync_point_t *sync_points;
    cudaMalloc((void **)&sync_points, num_subseq * sizeof(sync_point_t));

    phase1_decode_subseq<<<num_sequences, THREADS_PER_BLOCK>>>(total_nr_bits, num_subseq, bit_string, decode_table, sync_points);
    cudaDeviceSynchronize();

    printf("creating blocks\n");
    /* Phase 2 */
    bool blocks_synchronized = false;
    bool *sequences_sync_device;
    cudaMalloc((void **)&sequences_sync_device, num_sequences * sizeof(bool));

    bool *sequences_sync_host = (bool *)calloc(num_sequences, sizeof(bool));
    while (!blocks_synchronized){
        phase2_synchronise_blocks<<<num_sequences, THREADS_PER_BLOCK>>>(total_nr_bits, num_subseq, num_sequences, bit_string, decode_table,
                                                                        sync_points, sequences_sync_device);
        cudaDeviceSynchronize();
        cudaMemcpy(sequences_sync_host, sequences_sync_device, num_sequences * sizeof(bool), cudaMemcpyDeviceToHost);
        const size_t num_blocks = num_sequences - 1;
        blocks_synchronized = true;
        for(size_t i = 1; i < num_blocks; i++) {
            if (i % 100 == 0) {
               printf("i: %d\n", i);
            }
            if(sequences_sync_host[i] == 0) {
                blocks_synchronized = false;
                break;
            }
        }
    }
    free(sequences_sync_host);
    printf("Exclusive scan\n");

    /* Exclusive scan to get total number of symbols */
    int total_num_chars = phase3(sync_points, num_subseq, num_sequences);
    char *output_buf_device;
    cudaMalloc((void **)&output_buf_device, total_num_chars * sizeof(char));
    printf("total_num_chars: %d\n", total_num_chars);

    /* Write into output */
    phase4_decode_write_output<<<num_sequences, THREADS_PER_BLOCK>>>(
            total_nr_bits, num_subseq, bit_string, decode_table, sync_points, output_buf_device);

    char *output_buf_host = (char *) calloc(total_num_chars + 1, sizeof(char));
    cudaMemcpy(output_buf_host, output_buf_device, total_num_chars * sizeof(char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    fwrite(output_buf_host, 1, total_num_chars, ofile);

}
