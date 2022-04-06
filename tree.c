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

/*
 * Return the traversal path for a given character.
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

/*
 * Build a Huffmann encoding tree from a priority queue.
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
        up->data.ch = 0;
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
void nuke_tree(struct node **root) {
    struct node *n;

    assert(root);

    n = *root;
    if (!n)
        return;

    nuke_tree(&n->left);
    nuke_tree(&n->right);

    free(n);
}

static inline int16_t tree_buff_off_incr(uint8_t *sh) {
    assert(sh);
    return (*sh < MAX_INT_BUF_BITS - 0x1U) ? !(++(*sh)) : !(*sh = 0);
}

static void deflate_tree(struct node *root, uint8_t *buf, uint16_t *off,
                         uint8_t *sh) {
    uint8_t i, ch, mask, bit;
    uint16_t j;
    assert(root);
    assert(buf);

    mask = 0x1U << (MAX_INT_BUF_BITS - 1);

    if (tree_leaf(root)) {
        printf("off: %u, sh: %u, write: 1[%c]: ", *off, *sh,
               (char)root->data.ch);
        buf[*off] <<= 1;
        buf[*off] = buf[*off] | 0x1;
        *off += tree_buff_off_incr(sh);

        ch = root->data.ch;

        for (i = 0; i < MAX_INT_BUF_BITS; i++) {
            bit = ((ch << i) & mask) > 0;
            buf[*off] <<= 1;
            buf[*off] = buf[*off] | bit;
            *off += tree_buff_off_incr(sh);
            printf("%d: [%d:0x%x] ", ((ch << i) & mask) > 0, *off, buf[*off]);
        }
        printf(" ... ");
        for (j = 0; j < *off; j++)
            printf("0x%x ", buf[j]);
        printf("\n");
    } else {
        printf("off: %u, sh: %u, write: 0 ... ", *off, *sh);
        for (j = 0; j <= *off; j++)
            printf("0x%x ", buf[j]);
        printf("\n");

        buf[*off] <<= 1;
        *off += tree_buff_off_incr(sh);
        deflate_tree(root->left, buf, off, sh);
        deflate_tree(root->right, buf, off, sh);
    }
}

uint8_t *encode_tree(struct node *root, uint16_t *eoff, uint8_t *esh) {
    uint8_t sh, *buf;
    uint16_t off, i;

    buf = calloc(MAX_HIST_TAB_LEN * MAX_INT_BUF_BITS, sizeof(uint8_t));
    off = 0;
    sh = 0;

    deflate_tree(root, buf, &off, &sh);

    printf("tree_buf: ");
    for (i = 0; i <= off; i++) {
        printf("0x%hhx ", buf[i]);
    }
    printf("\n");

    if (sh && (sh < MAX_INT_BUF_BITS)) {
        buf[off] <<= (MAX_INT_BUF_BITS - sh);
    }

    printf("deflate: %u, %u\n", off, sh);
    printf("tree_buf: ");
    for (i = 0; i <= off; i++) {
        printf("0x%hhx ", buf[i]);
    }
    printf("\n");

    // decode_tree(buf, off, sh);
    *eoff = off + 1;
    *esh = sh + 1;
    return buf;
}

static struct node *inflate_tree(uint8_t *buf, uint16_t eoff, uint8_t esh,
                                 uint16_t *doff, uint8_t *dsh) {
    uint8_t bit, mask, sh, last, i;
    struct node *n;

    n = calloc(1, sizeof(struct node));
    assert(n);

    mask = 0x1U << (MAX_INT_BUF_BITS - 1);

    // printf("inflate_tree: eoff: %u doff: %u esh: %u, dsh: %u\n", eoff, *doff,
    // esh, *dsh);

    if (*doff >= eoff) {
        return NULL;
    }

    sh = MAX_INT_BUF_BITS;
    last = 0;
    if (*doff == (eoff - 1)) {
        printf("last byte\n");
        sh = esh;
        last = 1;
    }

    bit = ((buf[*doff] << *dsh) & mask) > 0;
    printf("off: %u, sh: %u, byte: 0x%x, bit: %u\n", *doff, *dsh, buf[*doff],
           bit);
    *doff += tree_buff_off_incr(dsh);

    if (bit) {
        n->left = NULL;
        n->right = NULL;

        printf("leaf: ");
        for (i = 0; i < sh; i++) {
            bit = ((buf[*doff] << *dsh) & mask) > 0;
            printf("%d", bit);
            n->data.ch <<= 1;
            n->data.ch = n->data.ch | bit;
            *doff += tree_buff_off_incr(dsh);
        }

        printf("\nret: %c\n", n->data.ch);
        return n;
    }

    n->data.ch = PSEUDO_TREE_BYTE;
    n->left = inflate_tree(buf, eoff, esh, doff, dsh);
    n->right = inflate_tree(buf, eoff, esh, doff, dsh);

    // printf("ret\n");
    return n;
}

struct node *decode_tree(uint8_t *buf, uint16_t eoff, uint8_t esh) {
    uint8_t dsh;
    uint16_t doff;

    dsh = 0;
    doff = 0;

    return inflate_tree(buf, eoff, esh, &doff, &dsh);
}
