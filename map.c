#include "huffman.h"
#include "tree.h" // for defines

/*
 * Return the number of non-zero
 * frequency keys in the table.
 */
uint32_t table_size(uint32_t *tab) {
    uint32_t size = 0, i;

    assert(tab);

    for (i = 0; i < MAX_HIST_TAB_LEN; i++)
        if (tab[i])
            size++;

    return size;
}

/* Create a hash table.
 *
 * The integral value of a byte is used to index
 * into the table and pupulate its frequency.
 */
uint32_t *make_table(FILE *fp) {
    uint32_t *tab;
    uint8_t read_ch;
    int16_t file_ch;

    assert(fp);

    tab = (uint32_t *)calloc(MAX_HIST_TAB_LEN, sizeof(uint32_t));
    assert(tab);

    /* There should be only one pseudo-EOF byte in the table. */
    tab[PSEUDO_NULL_BYTE] = 1;

    while ((file_ch = fgetc(fp)) != EOF) {
        read_ch = (uint8_t)file_ch;
        tab[read_ch]++;
    }

    return tab;
}

/* Make a map (byte-to-frequency).
 *
 * The map is built using an already constructed hash table.
 * This is added as a header to the encoded file, and can be
 * used to contruct a Huffmann tree while decoding. The size
 * of the map is retuned via "ret".
 */
struct map *make_map(FILE *fp, uint32_t *ret) {
    uint32_t map_sz, i, j;
    uint32_t *tab = NULL;
    struct map *fmap = NULL;

    tab = make_table(fp);
    assert(tab);

    map_sz = table_size(tab);
    fmap = (map *)calloc(map_sz, sizeof(struct map));
    assert(fmap);

    j = 0;
    for (i = 0; i < MAX_HIST_TAB_LEN; i++) {
        if (tab[i]) {
            fmap[j].ch = (uint8_t)i;
            fmap[j].freq = tab[i];
            j++;
        }
    }

    free(tab);

    *ret = map_sz;
    return fmap;
}
