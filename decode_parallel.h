#ifndef __DECODE_PARALLEL_H__
#define __DECODE_PARALLEL_H__

#include <stdint.h>
#include <stdio.h>
#define SUBSEQ_LENGTH 50
#define updivide(x, y) (((x) + ((y)-1))/(y))
#define THREADS_PER_BLOCK (64)

typedef struct sync_point {
    int last_word_bit;
    int num_symbols;
    bool sync;
} sync_point_t;


/* Bit string is on the device */
char * get_bit_string_device(FILE *ifile, uint64_t rd_bits);
/*
 * Build a Huffmann encoding tree from a priority queue.
 * This tree is on the device.
 */
void make_tree_device(struct node *head);
/* We assume that the decode_table and bit_string is malloc-ed in cuda */
void decode_cuda(uint64_t total_nr_bits, const char *bit_string, struct node*decode_table);

#endif /* DEFINE __DECODE_PARALLEL_H__ */