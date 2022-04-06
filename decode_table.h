#ifndef __DECODE_TABLE_H__
#define __DECODE_TABLE_H__

#include "tree.h"
#include <vector>
#include <string>
#include <unordered_map>


#define MAX_CODEWORD_LENGTH 23

typedef struct symbol {
    char character;
    int num_bits = 0;
} symbol_t;

using CodeWord = std::vector<bool>;
using DecodeTable = std::unordered_map<std::string, symbol_t>;


void get_decoder_table(node_t *decoder_tree_root, DecodeTable *decode_tab);


#endif /* DEFINE __DECODE_TABLE_H_ */

