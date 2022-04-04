#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

#define MAX_HIST_TAB_LEN (0x1U << 8)
#define MAX_INT_BUF_BITS (0x8U)

#define PSEUDO_TREE_BYTE (MAX_HIST_TAB_LEN - 0x1U)
#define PSEUDO_NULL_BYTE (MAX_HIST_TAB_LEN - 0x2U)

struct node {
    struct node *next;
    struct node *right;
    struct node *left;
    uint8_t ch;
    uint64_t freq;
};

void qinit(struct node **head) {
    assert(*head);
    (*head)->next = NULL;
}

void enqueue(struct node **head, struct node *new) {
    assert(*head);
    assert(new);

    struct node *temp, *prev;

    temp = *head;
    prev = NULL;

    while (temp) {
        if (new->freq < temp->freq) 
            break;

        prev = temp;
        temp = temp->next;
    }

    if (!prev) {
        new->next = *head;
        *head = new;
        return;
    }

    new->next = prev->next;
    prev->next = new;
}

struct node *dequeue(struct node **head) {
    struct node *ret = NULL;

    if (*head) {
        ret = *head;
        *head = (*head)->next;
    }

    return ret;
}

uint64_t qsize(struct node *head) {
    uint64_t count = 0;
    struct node *temp;
    if (head) {
        temp = head;
        while (temp) {
            temp = temp->next;
            count++;
        }
    }

    return count;
}

uint64_t *build_hist_tab(FILE *fp) {
    uint64_t *table;
    uint8_t rch;
    int16_t fch;

    assert(fp);

    table = calloc(MAX_HIST_TAB_LEN, sizeof(uint64_t));
    assert(table);

    table[PSEUDO_NULL_BYTE] = 1;

    while ((fch = fgetc(fp)) != EOF) {
        rch = (uint8_t)fch;
        table[rch]++;
    }

    return table;
}


struct node *build_queue(uint64_t *table) {
    uint64_t i;
    struct node *head = NULL, *temp;
    for (i = 0; i < MAX_HIST_TAB_LEN; i++) {
        if (table[i]) {
            if (!head) {
                head = calloc(1, sizeof(struct node));
                assert(head);

                head->ch = i;
                head->freq = table[i];
                qinit(&head);
            } else {
                temp = calloc(1, sizeof(struct node));
                assert(temp);
                temp->ch = i;
                temp->freq = table[i];
                enqueue(&head, temp);
            }
        }
    }

    return head;
}

void build_tree(struct node **head) {
    assert(*head);

    while(qsize(*head) > 1) {
        struct node *left = dequeue(head);
        struct node *right = dequeue(head);
        struct node *new = calloc(1, sizeof(struct node));

        new->ch = PSEUDO_TREE_BYTE;
        new->freq = left->freq + right->freq;
        new->left = left;
        new->right = right;

        if (*head) {
            enqueue(head, new);
        } else {
            *head = new;
            qinit(head);
        }    
    }
}

uint64_t tree_height(struct node *root) {
    uint64_t lh, rh;

    if (!root)
        return 0;

    lh = tree_height(root->left);
    rh = tree_height(root->right);

    return (lh > rh ? lh : rh) + 1;
}

uint8_t tree_leaf(struct node *root) {
    assert(root);

    return (!root->left && !root->right);
}

void traverse_tree(uint8_t ch, struct node *root, int8_t off,
                   uint8_t *arr, int8_t *ret) {
    assert(root);

    if (tree_leaf(root) && root->ch == ch) {
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

int8_t huffman_code(uint8_t ch, struct node *root, uint8_t *arr) {
    assert(root);
    assert(arr);

    int8_t off = -1;
    traverse_tree(ch, root, 0, arr, &off);

    return off;
}

uint64_t encode(FILE *fp, struct node *root, uint8_t **out) {
    uint8_t *arr, *buf, sh_off, rch;
    uint64_t bf_off;
    int8_t i, hc_off, fch;

    assert(root);
    assert(fp);

    arr = calloc(tree_height(root), sizeof(uint8_t));
    buf = calloc(20, sizeof(uint8_t));

    sh_off = 0;
    bf_off = 0;

    uint8_t tmp = 0;
    while ((fch = fgetc(fp)) != EOF) {
        rch = (uint8_t)fch;
        hc_off = huffman_code(rch, root, arr);
        assert(hc_off > 0);

        // printf("ch: \'%c\': ", rch);
        // for (i = 0; i < hc_off; i++) {
        //     printf("%u", arr[i]);
        // }
        // printf("\n");

        for (i = 0; i < hc_off; i++) {
            printf("%u", arr[i]);
            // buf[bf_off] = (buf[bf_off] << 1) | arr[i];
            tmp = (tmp << 1) | arr[i];
            sh_off++;

            if (sh_off >= MAX_INT_BUF_BITS) {
                printf("\n0x%x [%llu]\n", tmp, bf_off);
                buf[bf_off] = tmp;
                bf_off++;

                // buf = realloc(buf, bf_off * sizeof(uint8_t));
                // buf[bf_off] = 0x0U;
                tmp = 0;
                sh_off = 0;
                // printf(" ");
            }
        }
        memset(arr, 0, tree_height(root) * sizeof(uint8_t));
    }

    hc_off = huffman_code(PSEUDO_NULL_BYTE, root, arr);

    for (i = 0; i < hc_off; i++) {
        printf("%u", arr[i]);
        // buf[bf_off] = (buf[bf_off] << 1) | arr[i];
        tmp = (tmp << 1) | arr[i];
        sh_off++;

        // gets skipped
        if (sh_off >= MAX_INT_BUF_BITS) {
            printf("\n0x%x [%llu]\n", tmp, bf_off);
            buf[bf_off] = tmp;
            bf_off++;

            // buf = realloc(buf, bf_off * sizeof(uint8_t));
            assert(buf);
            buf[bf_off] = 0x0U;
            sh_off = 0;
            tmp = 0;
            // printf(" ");
        }
    }   

    printf("\n");
    if (sh_off && sh_off < MAX_INT_BUF_BITS) {
        printf("extra: %u\n", MAX_INT_BUF_BITS - sh_off);
        buf[bf_off] = tmp << (MAX_INT_BUF_BITS - sh_off);
    }

    *out = buf;
    return sh_off ? bf_off + 1: bf_off;
}

uint64_t decode(uint8_t *buf, uint64_t off, struct node *root, uint8_t **out) {
    assert(root);
    assert(buf);

    uint8_t *str;
    volatile uint8_t sh_off, chunk, mask;
    uint64_t bf_off, st_off;
    struct node *branch;

    str = calloc(64, sizeof(uint8_t));
    str[0] = 0x0;
    sh_off = 0;
    bf_off = 0;
    st_off = 0;
    branch = root;
    mask = 0x1U << (MAX_INT_BUF_BITS - 1);
    chunk = buf[bf_off];

    printf("0x%x\n", chunk);
    while (bf_off < off) {
        chunk = buf[bf_off] << sh_off;
        sh_off++;

        branch = (chunk & mask) ? branch->right : branch->left;
        printf("%d\n", (chunk & mask) ? 1 : 0);

        if (tree_leaf(branch)) {
            if (branch->ch == PSEUDO_NULL_BYTE) {
                printf("ch: EOF\n");
                str[st_off] = '\0';
                break;
            }

            printf("ch: '%c'\n", branch->ch);
            str[st_off] = branch->ch;

            st_off++;
            // printf("realloc: %s, %llu\n", str, st_off);
            // str = realloc(str, st_off * sizeof(uint8_t));
            assert(str);

            str[st_off] = 0x0;
            branch = root;
        }

        if (sh_off >= MAX_INT_BUF_BITS) {
            bf_off++;
            chunk = buf[bf_off];
            sh_off = 0;
            printf("0x%x\n", chunk);
        }
    }

    // if (str[st_off] != '\0') {
    //     str = realloc(str, (st_off + 1) * sizeof(char));
    //     str[st_off+1] = '\0';
    // }

    *out = str;
    return st_off;
}

int main(int argc, char *argv[]) {
    uint64_t *table, i, off;
    uint8_t *enc, *dec;
    struct node *head;
    FILE *in_file;

    if (argc < 2 || !(in_file = fopen(argv[1], "rb")))
        return 1;

    table = build_hist_tab(in_file);
    head = build_queue(table);
    build_tree(&head);
    rewind(in_file);

    for (i = 0; i < MAX_HIST_TAB_LEN; i++) {
        if (table[i]) {
            uint8_t *arr = calloc(tree_height(head) * 2, sizeof(uint8_t));
            if (i == PSEUDO_NULL_BYTE)
                printf("EOF: %4llu; code: ", table[i]);
            else if (i == '\n')
                printf("EOL: %4llu; code: ", table[i]);
            else
                printf("'%c': %4llu; code: ", (char)i, table[i]);
            int64_t hc_off = huffman_code((uint8_t)i, head, arr);
            assert(hc_off >= 0);

            for (int64_t j = 0; j < hc_off; j++) {
                printf("%u", arr[j]);
            }

            free(arr);
            printf("\n");
        }
    }

    off = encode(in_file, head, &enc);
    fclose(in_file);

    for (i = 0; i < off; i++) {
        printf("0x%x ", enc[i]);
    }
    printf("\n");

    off = decode(enc, off, head, &dec);

    for (i = 0; i < off; i++) {
        printf("%c", (char)dec[i]);
    }
    printf("\n");


    return 0;
}
