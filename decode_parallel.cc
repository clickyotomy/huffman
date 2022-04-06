

symbol_t symb decode_codeword(node_t *decode_tree, std::vector<bool> &bit_string, int ind) {

}

#define SUBSEQ_LENGTH 50
void decode_subsequence(node_t *decode_tree, std::vector<bool> &bit_string, int subseq_start,
                        int *out_num_symbols, int *out_last_offset) {

    /* start decoding from the start of the subsequence */
    int offset = subseq_start;
    int subseq_end = offset + SUBSEQ_LENGTH;
    int num_symbols = 0;
    int prev_num_bits = 0;

    while (offset < subseq_end) {
        symbol_t symb = decode_codeword(decode_tree, bit_string, offset);
        next_offset += symb.num_bits;
        prev_num_bits = symb.num_bits;
        num_symbols++;
    }
    *out_num_symbols = num_symbols;
    *out_last_offset = offset - prev_num_bits;

}
//
//    // current unit in this subsequence
//    std::uint32_t current_unit = 0;
//
//    // current bit position in unit
//    std::uint32_t at = start_bit;
//
//    // number of symbols found in this subsequence
//    std::uint32_t num_symbols_l = 0;
//
//    // perform overflow from previous subsequence
//    if(overflow && current_subsequence > 0) {
//
//        // shift to start
//        UNIT_TYPE copy_next = next;
//        copy_next >>= bits_in_unit - at;
//
//        next <<= at;
//        window <<= at;
//        window += copy_next;
//
//        // decode first symbol
//        std::uint32_t taken = table[(window & mask) >> shift].num_bits;
//
//        copy_next = next;
//        copy_next >>= bits_in_unit - taken;
//
//        next <<= taken;
//        window <<= taken;
//        at += taken;
//        window += copy_next;
//
//        // overflow
//        if(at > bits_in_unit) {
//            ++in_pos;
//            window = in_ptr[in_pos];
//            next = in_ptr[in_pos + 1];
//            at -= bits_in_unit;
//            window <<= at;
//            next <<= at;
//
//            copy_next = in_ptr[in_pos + 1];
//            copy_next >>= bits_in_unit - at;
//            window += copy_next;
//        }
//
//        else {
//            ++in_pos;
//            window = in_ptr[in_pos];
//            next = in_ptr[in_pos + 1];
//            at = 0;
//        }
//    }
//
//    while(current_unit < subsequence_size) {
//
//        while(at < bits_in_unit) {
//            const cuhd::CUHDCodetableItemSingle hit =
//                    table[(window & mask) >> shift];
//
//            // decode a symbol
//            std::uint32_t taken = hit.num_bits;
//            ++num_symbols_l;
//
//            if(write_output) {
//                if(out_pos < next_out_pos) {
//                    out_ptr[out_pos] = hit.symbol;
//                    ++out_pos;
//                }
//            }
//
//            UNIT_TYPE copy_next = next;
//            copy_next >>= bits_in_unit - taken;
//
//            next <<= taken;
//            window <<= taken;
//            last_word_bit = at;
//            at += taken;
//            window += copy_next;
//            last_word_unit = current_unit;
//        }
//
//        // refill decoder window if necessary
//        ++current_unit;
//        ++in_pos;
//
//        window = in_ptr[in_pos];
//        next = in_ptr[in_pos + 1];
//
//        if(at == bits_in_unit) {
//            at = 0;
//        }
//
//        else {
//            at -= bits_in_unit;
//            window <<= at;
//            next <<= at;
//
//            UNIT_TYPE copy_next = in_ptr[in_pos + 1];
//            copy_next >>= bits_in_unit - at;
//            window += copy_next;
//        }
//    }
//
//    num_symbols = num_symbols_l;
//    last_at = at;
//}