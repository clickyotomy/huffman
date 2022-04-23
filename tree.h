#ifndef __TREE_H__
#define __TREE_H__

#include <stdint.h>
/* Maximum size of the histogram table. */
#define MAX_HIST_TAB_LEN (0x1U << 8)

/*
 * Intermediate nodes in the tree are populated with
 * this byte to distinguish them from leaf nodes.
 */
#define PSEUDO_TREE_BYTE (MAX_HIST_TAB_LEN - 0x1U)

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

/* Return the height of the tree. */
uint32_t tree_height(struct node *root);
/* Return true if the node is the leaf of a tree. */
uint8_t tree_leaf(struct node *node);
void traverse_tree(uint8_t ch, struct node *root, int8_t off, uint8_t *arr,
                   int8_t *ret);
void make_tree(struct node **head);
void nuke_tree(struct node **root);

#endif /* DEFINE __TREE_H__ */