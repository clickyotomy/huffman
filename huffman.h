/*
 * huffman: A simple text-based Huffman {enc,dec}order.
 */

#ifndef __HUFFMAN_H__
#define __HUFFMAN_H__

#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Maximum size of a encoded (or decoded) byte. */
#define MAX_INT_BUF_BITS (0x8U)

/*
 * Denotes the end of file (EOF) for our encoding or
 * decoding scheme. This is not a "NULL" byte or EOF.
 */
#define PSEUDO_NULL_BYTE (MAX_HIST_TAB_LEN - 0x2U)

/* Stores the metadata for the encoded file/ */
typedef struct meta {
    uint32_t map_sz;     /* Size of the map. */
    uint32_t tree_depth; /* Height of the decode tree */
    uint64_t nr_bytes;   /* Number of encoded bytes (excluding headers). */
    uint64_t nr_bits;    /* Number of encoded bits (excluding headers). */
} meta_t;

/* Routines for maps. */
uint32_t table_size(uint32_t *);
uint32_t *make_table(FILE *);
struct map *make_map(FILE *, uint32_t *);

/* Routines for encoding and decoding. */
int8_t huffman_code(uint8_t, struct node *, uint8_t *);
uint64_t encode(FILE *, struct node *, FILE *);
uint64_t decode(FILE *, uint64_t, struct node *, FILE *);

/* Helper routines. */
void prog_usage(const char *);
void err_exit(const char *);

#endif /* DEFINE __HUFFMAN_H__ */
