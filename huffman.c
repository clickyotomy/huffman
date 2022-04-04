#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>

#define MAX_HIST_TAB_LEN (0x1U << 8)
#define MAX_INT_BUF_BITS (0x8U)

#define PSEUDO_TREE_BYTE (MAX_HIST_TAB_LEN - 0x1U)
#define PSEUDO_NULL_BYTE (MAX_HIST_TAB_LEN - 0x2U)

struct meta {
    uint64_t map_sz;
    uint64_t tree_ht;
};

struct map {
    uint8_t ch;
    uint64_t freq;
};

struct node {
    struct node *next;
    struct node *right;
    struct node *left;
    struct map data;
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
        if (new->data.freq < temp->data.freq) 
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

uint64_t hist_tab_size(uint64_t *tab) {
    uint64_t size = 0, i;

    assert(tab);

    for (i = 0; i < MAX_HIST_TAB_LEN; i++)
        if (tab[i])
            size++;

    return size;
}

struct map *build_freq_map(uint64_t *tab, uint64_t sz) {
    uint64_t i, j;
    struct map *fm;

    assert(tab);

    fm = calloc(sz, sizeof(struct map));
    assert(fm);

    j = 0;
    for (i = 0; i < MAX_HIST_TAB_LEN; i++) {
        if (tab[i]) {
            fm[j].ch = (uint8_t)i;
            fm[j].freq = tab[i];
            j++;
        }
    }

    return fm;
}

struct node *build_queue(struct map *m, uint64_t sz) {
    uint64_t i;
    struct node *head = NULL, *temp;

    assert(m);

    for (i = 0; i < sz; i++) {
        if (!head) {
            head = calloc(1, sizeof(struct node));
            assert(head);

            head->data.ch = m[i].ch;
            head->data.freq = m[i].freq;
            qinit(&head);
        } else {
            temp = calloc(1, sizeof(struct node));
            assert(temp);
            temp->data.ch = m[i].ch;
            temp->data.freq = m[i].freq;
            enqueue(&head, temp);
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

        new->data.ch = PSEUDO_TREE_BYTE;
        new->data.freq = left->data.freq + right->data.freq;
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

int8_t huffman_code(uint8_t ch, struct node *root, uint8_t *arr) {
    assert(root);
    assert(arr);

    int8_t off = -1;
    traverse_tree(ch, root, 0, arr, &off);

    return off;
}

uint64_t encode(FILE *ifile, FILE *ofile, struct node *root) {
    uint8_t *arr, sh_off, rch;
    uint64_t bf_off;
    int8_t i, hc_off, fch;

    assert(root);
    assert(ifile);
    assert(ofile);

    arr = calloc(tree_height(root), sizeof(uint8_t));

    sh_off = 0;
    bf_off = 0;

    uint8_t tmp = 0;
    while ((fch = fgetc(ifile)) != EOF) {
        rch = (uint8_t)fch;
        hc_off = huffman_code(rch, root, arr);
        assert(hc_off > 0);

        // printf("ch: \'%c\': ", rch);
        // for (i = 0; i < hc_off; i++) {
        //     printf("%u", arr[i]);
        // }
        // printf("\n");

        for (i = 0; i < hc_off; i++) {
            // printf("%u", arr[i]);
            tmp = (tmp << 1) | arr[i];
            sh_off++;

            if (sh_off >= MAX_INT_BUF_BITS) {
                printf("\n0x%x\n", tmp);
                fputc(tmp, ofile);
                tmp = 0;
                sh_off = 0;
            }
        }
    }

    hc_off = huffman_code(PSEUDO_NULL_BYTE, root, arr);
    for (i = 0; i < hc_off; i++) {
        // printf("%u", arr[i]);
        tmp = (tmp << 1) | arr[i];
        sh_off++;

        if (sh_off >= MAX_INT_BUF_BITS) {
            // printf("\n0x%x\n", tmp);
            fputc(tmp, ofile);
            tmp = 0;
            sh_off = 0;
        }
    }   

    // printf("\n");
    if (sh_off && sh_off < MAX_INT_BUF_BITS) {
        printf("extra: %u\n", MAX_INT_BUF_BITS - sh_off);
        // buf[bf_off] = tmp << (MAX_INT_BUF_BITS - sh_off);
        fputc(tmp << (MAX_INT_BUF_BITS - sh_off), ofile);
    }
    return 0;
}

uint64_t decode(FILE *ifile, FILE *ofile, struct node *root) {
    uint8_t sh_off, chunk, mask;
    struct node *branch;
    int16_t fch;

    assert(root);
    assert(ifile);
    assert(ofile);

    sh_off = 0;
    branch = root;
    mask = 0x1U << (MAX_INT_BUF_BITS - 1);

    fch = fgetc(ifile);
    chunk = (uint8_t)fch << sh_off;
    printf("chunk: 0x%x\n", chunk);
    while (1) {
        chunk = (uint8_t)fch << sh_off;
        sh_off++;

        branch = (chunk & mask) ? branch->right : branch->left;
        printf("%d\n", (chunk & mask) ? 1 : 0);

        if (tree_leaf(branch)) {
            if (branch->data.ch == PSEUDO_NULL_BYTE) {
                printf("ch: EOF\n");
                break;
            }

            printf("ch: '%c'\n", branch->data.ch);
            fputc(branch->data.ch, ofile);

            branch = root;
        }

        if (sh_off >= MAX_INT_BUF_BITS) {
            fch = fgetc(ifile);
            if (fch == EOF) {
                printf("fch: EOF: 0x%x\n", fch);
                break;
            }
            chunk = (uint8_t)fch;
            sh_off = 0;
            printf("chunk: 0x%x\n", chunk);

            // if (chunk == PSEUDO_NULL_BYTE) {
            //     printf("break: 0x%x\n", chunk);
            //     break;
            // }
        }
    }

    // if (str[st_off] != '\0') {
    //     str = realloc(str, (st_off + 1) * sizeof(char));
    //     str[st_off+1] = '\0';
    // }

    return 0;
}

int main(int argc, char *argv[]) {
    uint64_t *table, msz, i;
    uint8_t en;
    struct node *head;
    struct map *fm;
    char *ifpath, *ofpath;
    FILE *ifile, *ofile;
    int16_t arg;

    while ((arg = getopt(argc, argv, "edi:o:")) > 0) {
        switch (arg) {
        case 'e':
            en = 1;
            break;
        case 'd':
            en = 0;
            break;
        case 'i':
            ifpath = optarg;
            break;
        case 'o':
            ofpath = optarg;
            break;
        case 'h':
        case '?':
        default:
            printf("TBD\n");
        }
    }

    if (!(ifile = fopen(ifpath, "rb")))
        return 1;

    if (!(ofile = fopen(ofpath, "wb")))
        return 1;

    if (en) {
        table = build_hist_tab(ifile);
        msz = hist_tab_size(table);
        fm = build_freq_map(table, msz);
        head = build_queue(fm, msz);
        build_tree(&head);
        rewind(ifile);

    for (i = 0; i < msz; i++) {
        uint8_t *arr = calloc(tree_height(head), sizeof(uint8_t));
        if (fm[i].ch == PSEUDO_NULL_BYTE)
            printf("EOF: %4llu; code: ", fm[i].freq);
        else if (fm[i].ch == '\n')
            printf("EOL: %4llu; code: ", fm[i].freq);
        else
            printf("'%c': %4llu; code: ", fm[i].ch, fm[i].freq);
        int64_t hc_off = huffman_code(fm[i].ch, head, arr);
        assert(hc_off > 0);

        for (int64_t j = 0; j < hc_off; j++) {
            printf("%u", arr[j]);
        }

        free(arr);
        printf("\n");
    }

        encode(ifile, ofile, head);

        fclose(ifile);
        fclose(ofile);

        FILE *t1file, *t2file;

        if (!(t1file = fopen(ofpath, "rb")))
            return 1;

        if (!(t2file = fopen("bar", "wb")))
            return 1;

        decode(t1file, t2file, head);
    } else {
        // off = decode(enc, off, head, &dec);

        // for (i = 0; i < off; i++) {
        //     printf("%c", (char)dec[i]);
        // }
        // printf("\n");
    }


    return 0;
}
