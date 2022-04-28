#include "huffman.h"

uint8_t logb2(uint32_t n) {
    uint8_t ret = 0;

    while (n >>= 1) {
        ret++;
    }

    return ret;
}

/* Return the height of the tree. */
uint32_t tree_height(struct node *root) {
    uint32_t lh, rh;

    if (!root)
        return 0;

    lh = tree_height(root->left);
    rh = tree_height(root->right);

    return (lh > rh ? lh : rh) + 1;
}

/* Return the number of nodes in a tree. */
uint32_t tree_nodes(struct node *root) {
    if (!root)
        return 0;

    return 1 + tree_nodes(root->left) + tree_nodes(root->right);
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

/* Track the byte offset and shifts for the tree buffer. */
static inline int16_t tree_buff_off_incr(uint8_t *sh) {
    assert(sh);
    return (*sh < MAX_INT_BUF_BITS - 0x1U) ? !(++(*sh)) : !(*sh = 0);
}

/*
 * Deflate a Huffman tree into a buffer.
 *
 * The tree is encoded as follows:
 *
 *  +---+---------------+---+---+---+-------------+-- >  < --+
 *  | L | B [0, ..., 7] | L | L | L | [0, ..., 7] |   >  <   |
 *  +---+---------------+---+---+---+-------------+-- >  < --+
 *
 *  - L     A bit indicating whether a node is a leaf. If the node is a
 *          leaf, then it is followed by a sequence of "MAX_INT_BUF_BITS"
 *          representing the byte value at that leaf node. Otherwise, it
 *          is followed by a subsequent bit indicating the status of the
 *          next node.
 *  - B[]   The byte value of a leaf node.
 *
 * The encoded buffer along with the byte and shift offsets are set to
 * pointers in "buf", "off" and "sh" respectively.
 *
 */
static void deflate_tree(struct node *root, uint8_t *buf, uint16_t *off,
                         uint8_t *sh) {
    uint8_t i, ch, mask, bit;

    assert(root);
    assert(buf);

    mask = 0x1U << (MAX_INT_BUF_BITS - 1);

    if (tree_leaf(root)) {
        buf[*off] <<= 1;
        buf[*off] = buf[*off] | 0x1;
        *off += tree_buff_off_incr(sh);

        ch = root->data.ch;

        for (i = 0; i < MAX_INT_BUF_BITS; i++) {
            bit = ((ch << i) & mask) > 0;
            buf[*off] <<= 1;
            buf[*off] = buf[*off] | bit;
            *off += tree_buff_off_incr(sh);
        }

        return;
    }

    buf[*off] <<= 1;
    *off += tree_buff_off_incr(sh);
    deflate_tree(root->left, buf, off, sh);
    deflate_tree(root->right, buf, off, sh);
}

/* Wrapper for tree deflation. */
uint8_t *encode_tree(struct node *root, uint16_t *eoff, uint8_t *esh) {
    uint8_t sh = 0, *buf = NULL;
    uint16_t off = 0;

    /* Allocate a large enough array to hold the tree buffer. */
    buf = calloc(MAX_HIST_TAB_LEN * MAX_INT_BUF_BITS, sizeof(uint8_t));
    assert(buf);

    deflate_tree(root, buf, &off, &sh);

    /*
     * Shift any leftover bits to the left. Since the byte is read
     * from the right (during inflation), we want theencoded values
     * to be there.
     */
    if (sh && (sh < MAX_INT_BUF_BITS)) {
        buf[off] <<= (MAX_INT_BUF_BITS - sh);
    }

    *eoff = off + 1;
    *esh = sh + 1;
    return buf;
}

/*
 * Inflate the tree buffer into a Huffman tree.
 *
 * The most significant bit is checked to see if it is a leaf node. If that
 * is the case, then the next "MAX_INT_BUF_BITS" are read to get the byte
 * value at the leaf node. A new leaf node is created. Otherwise, it is an
 * intermediate node, and this function is called recursively until all the
 * leaf nodes are created. The last non-leaf node will be the root of the
 * tree.
 */
static struct node *inflate_tree(uint8_t *buf, uint16_t eoff, uint8_t esh,
                                 uint16_t *doff, uint8_t *dsh) {
    uint8_t sh = MAX_INT_BUF_BITS, bit, mask, i;
    struct node *n;

    /* No more bytes left to decode. */
    if (*doff >= eoff)
        return NULL;

    n = calloc(1, sizeof(struct node));
    assert(n);

    mask = 0x1U << (MAX_INT_BUF_BITS - 1);

    /* Only read the leftover bits, for the last byte. */
    if (*doff == (eoff - 1))
        sh = esh;

    bit = ((buf[*doff] << *dsh) & mask) > 0;
    *doff += tree_buff_off_incr(dsh);

    /* Leaf node. */
    if (bit) {
        n->left = NULL;
        n->right = NULL;

        for (i = 0; i < sh; i++) {
            bit = ((buf[*doff] << *dsh) & mask) > 0;
            *doff += tree_buff_off_incr(dsh);

            n->data.ch <<= 1;
            n->data.ch = n->data.ch | bit;
        }

        return n;
    }

    /* Intermediate (or root) node. */
    n->data.ch = PSEUDO_TREE_BYTE;
    n->left = inflate_tree(buf, eoff, esh, doff, dsh);
    n->right = inflate_tree(buf, eoff, esh, doff, dsh);

    return n;
}

/* Wrapper for tree inflation. */
struct node *decode_tree(uint8_t *buf, uint16_t eoff, uint8_t esh) {
    uint8_t dsh = 0;
    uint16_t doff = 0;

    return inflate_tree(buf, eoff, esh, &doff, &dsh);
}

/* Return the Huffman code for a given byte. */
int8_t huffman_code(uint8_t ch, struct node *root, uint8_t *arr) {
    assert(root);
    assert(arr);

    int8_t off = -1;
    traverse_tree(ch, root, 0, arr, &off);

    return off;
}

/* Construct a lookup table for Huffman codes from the tree. */
struct lookup *make_lookup_table(struct node *root, uint32_t *nr_tab_ents) {
    uint32_t i, j;
    int32_t off;
    uint8_t *arr;
    uint32_t th, nr_ents = 0;
    uint64_t tmp, diff;
    struct lookup *tab = NULL;

    assert(nr_tab_ents);

    th = tree_height(root);
    assert(th > 0);

    arr = calloc(th, sizeof(uint8_t));
    assert(arr);

    /* Calculate the number of entries needed for the lookup table. */
    for (i = 0; i < MAX_HIST_TAB_LEN; i++) {
        off = huffman_code((uint8_t)i, root, arr);

        /* If the byte exists in the tree. */
        if (off > 0) {
            /*
             * Calculate the number of "repeat" entries, i.e.,
             * entries with the same Huffman code prefix that
             * should be added to the table. This is obtained
             * by calculating the number of bits the current
             * code differs from the code of the byte with the
             * maximum bit string length.
             */
            diff = 0x1UL << (th - off - 0x1U);
            nr_ents += diff;
        }
    }

    assert(nr_ents <= MAX_LOOKUP_TAB_LEN);

    tab = calloc(nr_ents, sizeof(struct lookup));
    assert(tab);

    for (i = 0; i < MAX_HIST_TAB_LEN; i++) {
        tmp = 0;
        off = huffman_code((uint8_t)i, root, arr);

        if (off > 0) {
            for (j = 0; j < (uint16_t)off; j++) {
                tmp |= arr[j];
                tmp <<= 1;
            }

            /* Trim off the extra left shift. */
            tmp >>= 1;
            diff = (th - off) - 0x1U;

            if (diff > 0) {
                /* Shift the index value by the number of differing bits. */
                tmp <<= diff;

                /*
                 * Starting off with the inital index value, increment the
                 * index of the repeated entries until (2 ^ diff) entries
                 * have to be filled. For instance, if the code is "011",
                 * with length 3, and length of bit string for the byte with
                 * the maximum bit string length is 5, the bit difference is
                 * 5 - 3 = 2. So, we add repeat entries for the following
                 * indexes:
                 *
                 *  +------+------+-------+
                 *  | CODE | DIFF | INDEX |
                 *  +------+------+-------+
                 *  | 011  | 00   | 12    |
                 *  | 011  | 01   | 13    |
                 *  | 011  | 10   | 14    |
                 *  | 011  | 11   | 15    |
                 *  +------+------+-------+
                 *
                 * For a difference of 2 bits, a total for 2^2 = 4
                 * entries are repeated.
                 *
                 */
                for (j = 0; j < (0x1UL << diff); j++) {
                    tab[tmp] = (struct lookup){
                        .ch = (uint8_t)i,
                        .off = (uint8_t)off,
                    };

                    tmp++;
                }
            } else {
                tab[tmp] = (struct lookup){
                    .ch = (uint8_t)i,
                    .off = (uint8_t)off,
                };
            }
        }
    }

    free(arr);

    *nr_tab_ents = nr_ents;
    return tab;
}

struct lookup *lookup_table(struct lookup *tab, uint32_t tab_sz, uint32_t idx) {
    if (idx < tab_sz)
        return &tab[idx];

    return NULL;
}

/* Print all the Huffman codes in the tree. */
void print_huffman_codes(struct node *root) {
    uint16_t i, j;
    uint8_t *arr;
    uint32_t th;
    int8_t off;

    th = tree_height(root);
    assert(th > 0);

    arr = calloc(th, sizeof(uint8_t));
    assert(arr);

    printf("Huffman Codes (Maximum Length: %u)\n", th - 1);
    for (i = 0; i < MAX_HIST_TAB_LEN; i++) {
        off = huffman_code((char)i, root, arr);

        if (off > 0) {
            switch (i) {
            case 0:
                printf("byte: NUL, code: ");
                break;

            case '\n':
                printf("byte: EOL, code: ");
                break;

            case '\r':
                printf("byte: EOL, code: ");
                break;

            case '\t':
                printf("byte: TAB, code: ");
                break;

            case PSEUDO_NULL_BYTE:
                printf("byte: EOF, code: ");
                break;

            default:
                if (i >= 32 && i < 127)
                    printf("byte: \'%c\', code: ", (char)i);
                else
                    printf("byte: BIN, code: ");
                break;
            }

            for (j = 0; j < off; j++)
                printf("%d", arr[j]);

            printf("\n");
        }
    }

    free(arr);
}

void print_huffman_table(struct lookup *tab, uint32_t nr_tab_ents) {
    uint32_t i, j, bit, len;
    uint64_t tmp;

    len = logb2(nr_tab_ents);

    printf("Huffman Look-up Table (Entries: %u)\n", nr_tab_ents);
    for (i = 0; i < nr_tab_ents; i++) {
        printf("table[%04u]: off: %02u, ", i, tab[i].off);

        switch (tab[i].ch) {
        case 0:
            printf("byte: NUL, code: ");
            break;

        case '\n':
            printf("byte: EOL, code: ");
            break;

        case '\r':
            printf("byte: EOL, code: ");
            break;

        case '\t':
            printf("byte: TAB, code: ");
            break;

        case PSEUDO_NULL_BYTE:
            printf("byte: EOF, code: ");
            break;

        default:
            if (tab[i].ch >= 32 && tab[i].ch < 127)
                printf("byte: \'%c\', code: ", (char)tab[i].ch);
            else
                printf("byte: BIN, code: ");
            break;
        }

        tmp = i >> (len - tab[i].off);
        for (j = 0; j < tab[i].off; j++) {
            bit = tmp & (0x1UL << (tab[i].off - 1));
            printf("%u", bit >> (tab[i].off - 1));
            tmp <<= 1;
        }
        printf("\n");
    }
}
