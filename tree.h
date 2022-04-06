#ifndef __TREE_H__
#define __TREE_H__

#include <cstdint>


/* Maps a character to its frequency in the data. */
struct map {
    uint8_t ch;
    uint32_t freq;
};

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

#endif /* DEFINE __TREE_H_ */

