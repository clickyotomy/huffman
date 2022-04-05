#include "huffman.h"

/* Return the height of the tree. */
uint32_t tree_height(struct node *root) {
    uint32_t lh, rh;

    if (!root)
        return 0;

    lh = tree_height(root->left);
    rh = tree_height(root->right);

    return (lh > rh ? lh : rh) + 1;
}

/* Return true if the node is the leaf of a tree. */
uint8_t tree_leaf(struct node *node) {
    assert(node);

    return (!node->left && !node->right);
}

/* Return the traversal path for a given character.
 *
 * In Huffman encoding, a "0" means a left, and a "1" means a
 * right. Starting at the root, find the path (in terms of "0"s
 * and "1"s) until a leaf node containing the character is
 * reached.
 */
void traverse_tree(uint8_t ch, struct node *root, int8_t off, uint8_t *arr,
                   int8_t *ret) {
    assert(root);

    if (tree_leaf(root) && root->data.ch == ch) {
        *ret = off;
        return;
    }

    if (*ret < 0 && root->left) {
        arr[off] = 0;
        traverse_tree(ch, root->left, off + 1, arr, ret);
    }

    if (*ret < 0 && root->right) {
        arr[off] = 1;
        traverse_tree(ch, root->right, off + 1, arr, ret);
    }
}

/* Build a Huffmann encoding tree from a priority queue.
 *
 * The tree is constructed by repeatedly dequeueing elements from
 * the queue (dequeueing is done from the head of the queue, and the
 * elements with the least frequency are removed first), and building
 * a tree such that the parent node's frequency is the sum of its
 * children. Thsis process continues until there is only one node
 * remaining in the queue. This node becomes the root of the tree.
 */
void make_tree(struct node **head) {
    assert(*head);

    struct node *lt, *rt, *up;

    while (queue_size(*head) > 1) {
        lt = dequeue(head);
        rt = dequeue(head);
        up = calloc(1, sizeof(struct node));

        up->data.ch = PSEUDO_TREE_BYTE;
        up->data.freq = (lt->data.freq + rt->data.freq);
        up->left = lt;
        up->right = rt;

        if (*head) {
            enqueue(head, up);
        } else {
            *head = up;
            init_queue(head);
        }
    }
}

/* Free all the nodes in the tree. */
void nuke_tree(struct node *root) {
    if (!root)
        return;

    nuke_tree(root->left);
    nuke_tree(root->right);

    free(root);
}
