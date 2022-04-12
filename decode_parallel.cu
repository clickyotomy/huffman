#include "decode_table.h"


#define SUBSEQ_LENGTH 50
/*
 *  Returns the number of bits taken to decode the current symbol
 */
#define UNIT_SIZE 8
uint64_t decode_one_symb(uint64_t bit_offset,  node_t *decode_root, char *bits) {

    uint64_t cur_bit_offset = 0;

    node_t *branch;

    while (1) {

        uint64_t cur_unit = cur_bit_offset / UNIT_SIZE;
        uint64_t cur_bit_in_unit = cur_bit_offset % UNIT_SIZE;
        uint64_t cur_bit = (bits[cur_unit] & (0x1U << cur_bit_in_unit));
        /* Determine which branch to take depending on the extracted bit. */
        branch = cur_bit ? branch->right : branch->left;

        /*
         * If we reached the leaf node, we have decoded a byte;
         * write it to the output file.
         */
        if (tree_leaf(branch)) {
            /* Let's skip out on the writing first */
//            fputc(branch->data.ch, ofile);

            /* This marks the end of the decoded file. */
            if (branch->data.ch == PSEUDO_NULL_BYTE) {
                break;
        }

        cur_bit_offset++;
    }

    return cur_bit_offset - bit_offset;
}

__global__ void decode_subsequence(node_t *decode_tree, char *bit_string, uint64_t subseq_start,
                        int *out_num_symbols, uint64_t *out_last_offset) {

    /* start decoding from the start of the subsequence */
    uint64_t offset = subseq_start;
    uint64_t subseq_end = offset + SUBSEQ_LENGTH;
    uint64_t prev_num_bits = 0;
    int num_symbols = 0;

    while (offset < subseq_end) {
        uint64_t num_bits = decode_one_symb(offset,  decode_tree, char *bit_string);
        offset += num_bits;
        prev_num_bits = num_bits;
        num_symbols++;
    }

    *out_num_symbols = num_symbols;
    *out_last_offset = offset - prev_num_bits;

}


typedef struct sync_point {
    int last_word_bit;
    int num_symbols;
    bool sync;
} sync_point_t;

__global__ void phase1_decode_subseq(
        std::uint32_t total_num_subsequences,
        const char *bit_string,
        node_t *decode_table,
        sync_point_t * sync_points) {

    const int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid < total_num_subsequences) {

        int current_subsequence = gid;
        int subsequence_offset = gid * SUBSEQ_LENGTH;
        int cur_pos = subsequence_offset;

        std::uint32_t subsequences_processed = 0;
        bool synchronised_flag = false;

        while(subsequences_processed < blockDim.x) {

            if(!synchronised_flag && current_subsequence < total_num_subsequences) {
                int num_symbols, last_word_bit;
                decode_subsequence(decode_tree, bit_string, cur_pos, &num_symbols, &last_word_bit);

                if(subsequences_processed == 0) {
                    sync_points[current_subsequence] =
                            {last_word_bit, num_symbols, 1};
                } else {
                    uint4 sync_point = sync_points[current_subsequence];
                    sync_point.num_symbols = num_symbols;

                    // if sync point detected
                    if(sync_point.last_word_bit == last_word_bit) {
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
}


__global__ void phase2_synchronise_blocks(
        int num_blocks,
        const char *bit_string,
        node_t *decode_table,
        sync_point_t * sync_points,
        bool *block_synchronised) {

    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int num_of_seams = num_blocks - 1;

    if(gid < num_of_seams) {
        // jump to first sequence of the block
        int current_subsequence = (gid + 1) * blockDim.x;

        // search for synchronised sequences at the end of previous block
        sync_point sync_point = sync_points[current_subsequence - 1];

        int cur_pos = sync_point.last_word_bit;

        // number of symbols found in this subsequence
        int num_symbols = 0;
        int subsequences_processed = 0;
        bool synchronised_flag = false;

        while(subsequences_processed < blockDim.x) {

            if(!synchronised_flag) {
                int last_word_bit;
                decode_subsequence(decode_tree, bit_string, cur_pos, &num_symbols, &last_word_bit);
                sync_point = sync_points[current_subsequence];
                sync_point.num_symbols = num_symbols;

                // if sync point detected
                if(sync_point.last_word_bit == last_word_bit) {
                    sync_point.sync = 1;
                    block_synchronised[gid + 1] = 1;
                    synchronised_flag = true;
                } else {
                    // correct erroneous position data
                    sync_point.last_word_bit = last_word_bit;
                    sync_point.w = 0;
                    block_synchronised[gid + 1] = 0;
                }
                sync_points[current_subsequence] = sync_point;
            }
            cur_pos = last_word_bit;

            ++current_subsequence;
            ++subsequences_processed;

            __syncthreads();
        }
    }
}

#define updivide(x, y) (((x) + ((y)-1))/(y))
#define THREADS_PER_BLOCK (64)


/* We assume that the decode_table and bit_string is malloc-ed in cuda */
void decode_cuda(const char *bit_string, const node_t *decode_table) {

    size_t num_subseq = updivide(input_size, SUBSEQ_LENGTH);
    size_t num_sequences = updivide(num_subseq, threads_per_block);

    /* Phase 1 */
    sync_point_t *sync_points;
    cudaMalloc((void **)&sync_points, num_subseq * sizeof(sync_point_t));

    phase1_decode_subseq<<<num_sequences, threads_per_block>>>(num_subseq, bit_string, decode_table, sync_points);

    /* Phase 2 */
    bool blocks_synchronized = false;
    bool *sequences_sync;
    cudaMalloc((void **)&sequences_sync, num_sequences * sizeof(bool));

    while (!blocks_synchronized){
        phase2_synchronise_blocks<<<num_sequences, threads_per_block>>>(num_sequences, bit_string, decode_table,
                                                                        sync_points, sequences_sync);
        const size_t num_blocks = num_sequences - 1;
        blcok_synchronized = true;
        for(size_t i = 1; i < num_blocks; i++) {
            if(sequences_sync[i] == 0) {
                blocks_synchronized = false;
                break;
            }
        }
    }
}
