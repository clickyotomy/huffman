#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "decode_parallel.h"
#include "queue.h"
#include "tree.h"


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
                         bool print) {

    uint64_t cur_bit_offset = bit_offset;
    int i = 1;
    while (cur_bit_offset < end_bit_offset) {
        uint64_t cur_unit = cur_bit_offset / UNIT_SIZE;
        uint64_t cur_bit_in_unit = (UNIT_SIZE - 1) - (cur_bit_offset % UNIT_SIZE);
        uint64_t cur_bit = (bits[cur_unit] & (0x1U << cur_bit_in_unit));
        /* Determine which branch to take depending on the extracted bit. */
        i = cur_bit ? 2 * i : 2 * i + 1;

        /*
         * If we reached the leaf node, we have decoded a byte;
         * write it to the output file.
         */
        if (is_tree_arr_leaf(decode_root + i)) {
            if (print) {
                printf("%d %c\n", i, decode_root[i].ch);
            }
            /* Let's skip out on the writing first */
//            fputc(branch->data.ch, ofile);
            break;
        }

        cur_bit_offset++;
    }

    return cur_bit_offset - bit_offset;
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
        uint64_t num_bits = decode_one_symb(offset,  decode_tree, bit_string, total_nr_bits, print);
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


/* We assume that the decode_table and bit_string is malloc-ed in cuda */
void decode_cuda(uint64_t total_nr_bits, const char *bit_string, tree_arr_node_t *decode_table) {

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
}
