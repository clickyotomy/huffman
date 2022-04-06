#include <iostream>
#include "decode_table.h"
#include <bitset>

void add_table_entry_permutations(DecodeTable *decode_table, CodeWord &c, symbol_t &s) {

    int shift = MAX_CODEWORD_LENGTH - s.num_bits;
    int num_entries = 1 << shift;
    std::string codeword_str = "";
    for (int i = 0; i < s.num_bits; i++) {
        codeword_str += std::to_string(c[i]);
    }


    std::cout << codeword_str << " :" << s.character << " " << s.num_bits << std::endl;
    for (int i = 0; i < num_entries; i++) {
        std::string offset = std::bitset<MAX_CODEWORD_LENGTH>(i).to_string();
        std::string offset_substr = offset.substr(s.num_bits, shift);
        decode_table->emplace(codeword_str + offset_substr, s);
    }
}

void get_decoder_table_helper(node_t *decoder_tree_root, DecodeTable *decode_table,
                              CodeWord codeword, symbol_t symb) {
    if (tree_leaf(decoder_tree_root)) {
        symbol_t s = symb;
        s.character = decoder_tree_root->data.ch;
//        printf("%d\n", symb.num_bits);
        add_table_entry_permutations(decode_table, codeword, s);
        return;
    }

    if (decoder_tree_root->left) {
        symbol_t s = symb;
        s.num_bits++;
        CodeWord c = codeword;
        get_decoder_table_helper(decoder_tree_root->left, decode_table, c, s);
    }

    if (decoder_tree_root->right) {
        symbol_t s = symb;
        CodeWord c = codeword;
//        s.num_bits++;
        c[s.num_bits++] = 1;
        get_decoder_table_helper(decoder_tree_root->right, decode_table, c, s);
    }
}

void get_decoder_table(node_t *decoder_tree_root, DecodeTable *decode_table) {
    std::vector<bool> c(MAX_CODEWORD_LENGTH, 0);
    symbol_t s;
    get_decoder_table_helper(decoder_tree_root, decode_table, c, s);
}