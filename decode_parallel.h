#ifndef __DECODE_PARALLEL_H__
#define __DECODE_PARALLEL_H__

#define SUBSEQ_LENGTH 50

/* Bit string is on the device */
char * get_bit_string_device(FILE *ifile, uint64_t rd_bits);
/*
 * Build a Huffmann encoding tree from a priority queue.
 * This tree is on the device.
 */
void make_tree_device(node_t **head);
/* We assume that the decode_table and bit_string is malloc-ed in cuda */
void decode_cuda(uint64_t total_nr_bits, const char *bit_string, const node_t *decode_table);

#endif /* DEFINE __DECODE_PARALLEL_H__ */