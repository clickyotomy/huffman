#ifndef __DECODE_PARALLEL_H__
#define __DECODE_PARALLEL_H__

#include <stdint.h>
#include <stdio.h>
#define SUBSEQ_LENGTH (5000)
#define updivide(x, y) (((x) + ((y)-1))/(y))
#define THREADS_PER_BLOCK (64)

typedef struct sync_point {
    int last_word_bit;
    int num_symbols;
    int out_pos;
    bool sync;
} sync_point_t;

typedef struct tree_arr_node {
    char ch;
} tree_arr_node_t;


/* Bit string is on the device */
char * get_bit_string_device(FILE *ifile, uint64_t rd_bits);

tree_arr_node_t *make_tree_device(struct node *head, uint32_t tree_depth);

/* We assume that the decode_table and bit_string is malloc-ed in cuda */
void decode_cuda(uint64_t total_nr_bits, const char *bit_string, tree_arr_node_t *decode_table, FILE *ofile);

#endif /* DEFINE __DECODE_PARALLEL_H__ */