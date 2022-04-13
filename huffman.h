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

/* Maximum size of the histogram table. */
#define MAX_HIST_TAB_LEN (0x1U << 8)

/* Maximum size of a encoded (or decoded) byte. */
#define MAX_INT_BUF_BITS (0x8U)

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

/* Stores the metadata for the encoded file/ */
typedef struct meta {
    uint32_t map_sz;   /* Size of the map. */
    uint64_t nr_bytes; /* Number of encoded bytes (excluding headers). */
    uint64_t nr_bits; /* Number of encoded bits (excluding headers). */
} meta_t;

/* Maps a character to its frequency in the data. */
typedef struct map {
    uint8_t ch;
    uint32_t freq;
} map_t;

/* Represents a node in the priority queue (or tree). */
typedef struct node {
    struct node *next;  /* Next link in the linked list (queue). */
    struct node *right; /* Tree node to the right. */
    struct node *left;  /* Tree node to the left. */
    struct map data;    /* A mapping of byte to its frequency. */
} node_t;

/* Routines for trees. */
uint32_t tree_height(struct node *root);
uint8_t tree_leaf(struct node *node);
void traverse_tree(uint8_t, struct node *, int8_t, uint8_t *, int8_t *);
void make_tree(struct node **);
void nuke_tree(struct node **);

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
int8_t huffman_code(uint8_t, struct node *, uint8_t *);
uint64_t encode(FILE *, struct node *, FILE *);
uint64_t decode(FILE *, uint64_t, struct node *, FILE *);

/* Helper routines. */
void prog_usage(const char *);
void err_exit(const char *);

#endif /* DEFINE __HUFFMAN_H__ */
