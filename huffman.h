/*
 * huffman: A simple text-based Huffman {en,de}coder.
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

/* Maximum size of the histogram table. */
#define MAX_HIST_TAB_LEN (0x1U << 8)

/* Maximum size of a encoded (or decoded) byte. */
#define MAX_INT_BUF_BITS (0x8U)

#define MAX_LOOKUP_TAB_LEN (0x1LU << 24)
/*
 * Intermediate nodes in the tree are populated with
 * this byte to distinguish them from leaf nodes.
 */
#define PSEUDO_TREE_BYTE (MAX_HIST_TAB_LEN - 0x1U)

/*
 * Denotes the end of file (EOF) for our encoding or
 * decoding scheme. This is not a "NULL" byte or EOF.
 */
#define PSEUDO_NULL_BYTE (MAX_HIST_TAB_LEN - 0x2U)

/* Stores the metadata for the encoded file. */
struct meta {
    uint8_t tree_lb_sh_pos; /* Bit shift position of the last tree byte. */
    uint16_t nr_tree_bytes; /* Number of bytes in the encoded tree buffer. */
    uint64_t nr_src_bytes;  /* Number of bytes in the original file. */
    uint64_t nr_enc_bytes;  /* Number of encoded bytes (excluding headers). */
};

/* Maps a character to its frequency in the data. */
struct map {
    uint8_t ch;
    uint32_t freq;
};

/* Map a canonical Huffman code to a byte (and offset). */
struct lookup {
    uint8_t ch;
    uint8_t off;
};

/* Represents a node in the priority queue (or tree). */
struct node {
    struct node *next;  /* Next link in the linked list (queue). */
    struct node *right; /* Tree node to the right. */
    struct node *left;  /* Tree node to the left. */
    struct map data;    /* A mapping of byte to its frequency. */
};

/* Routines for trees. */
uint8_t logb2(uint32_t);
uint32_t tree_height(struct node *root);
uint8_t tree_leaf(struct node *node);
void traverse_tree(uint8_t, struct node *, int8_t, uint8_t *, int8_t *);
void make_tree(struct node **);
void nuke_tree(struct node **);
uint8_t *encode_tree(struct node *, uint16_t *, uint8_t *);
struct node *decode_tree(uint8_t *, uint16_t, uint8_t);
int8_t huffman_code(uint8_t, struct node *, uint8_t *);
struct lookup *make_lookup_table(struct node *, uint32_t *);
struct lookup *lookup_table(struct lookup *, uint32_t, uint32_t);
void print_huffman_codes(struct node *);
void print_huffman_table(struct lookup *, uint32_t);

/* Routines for queues. */
uint32_t queue_size(struct node *);
void enqueue(struct node **head, struct node *);
struct node *dequeue(struct node **);
void init_queue(struct node **);
struct node *make_queue(struct map *, uint32_t);
void nuke_queue(struct node **);

/* Routines for maps. */
uint32_t table_size(uint32_t *);
uint32_t *make_table(FILE *);
struct map *make_map(FILE *, uint32_t *);

/* Routines for encoding and decoding. */
void encode(FILE *, struct node *, FILE *, uint64_t *, uint64_t *);
void decode(int16_t, int16_t, FILE *, struct meta *, struct node *, FILE *,
            uint64_t *, uint64_t *);
void decode_with_tree(FILE *, uint64_t, struct node *, FILE *, uint64_t *,
                      uint64_t *);
void decode_with_tab(FILE *, uint64_t, struct lookup *, uint32_t, FILE *,
                     uint64_t *, uint64_t *);

/* Helper routines. */
void prog_usage(const char *);
void err_exit(const char *);

#endif /* DEFINE __HUFFMAN_H__ */
