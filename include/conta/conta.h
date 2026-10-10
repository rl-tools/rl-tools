#ifndef CONTA_CONTA_H
#define CONTA_CONTA_H
// conta: client for the sha1 content-addressed asset store; index contract: https://github.com/rl-tools/conta-index (format 1)
// C++17, optional zlib for gzip decoding; downloads via the curl CLI.
// A sha1 resolves from <cache>/<sha1>, then the cached index, then the index entry points; downloads are verified against the sha1 and the sha256 the index lists.
// CONTA_ROOT: read-only store (<root>/data/<sha1>), disables the network. CONTA_URL: fetch <url>/<sha1> directly, without the index. CONTA_CACHE: writable cache directory.

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <string>
#include <system_error>
#include <vector>
#if defined(RL_TOOLS_ENABLE_ZLIB) || defined(CONTA_ENABLE_ZLIB)
#include <zlib.h>
#endif
#if defined(_WIN32)
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace conta{
    constexpr const char* CACHED_INDEX_NAME = "index-1.json";
    inline std::vector<std::string> default_entry_points(){
        return {
            "https://conta.rl.tools/version/1/meta.json",
            "https://raw.githubusercontent.com/rl-tools/conta-index/main/version/1/meta.json",
            "https://huggingface.co/datasets/rl-tools/conta-index/resolve/main/version/1/meta.json",
        };
    }
    struct Config{
        std::string root;
        std::string cache;
        std::string url_base;
        std::vector<std::string> entry_points = default_entry_points();
        bool compressed_cache = true;
    };
    namespace detail{
        template <std::size_t T_WORDS>
        std::string hex_words(const std::uint32_t (&words)[T_WORDS]){
            std::string hex;
            hex.reserve(T_WORDS * 8);
            for(std::size_t word_i = 0; word_i < T_WORDS; word_i++){
                for(int shift = 28; shift >= 0; shift -= 4){
                    hex += "0123456789abcdef"[(words[word_i] >> unsigned(shift)) & 0xFu];
                }
            }
            return hex;
        }
        inline std::uint32_t big_endian_word(const unsigned char* bytes){
            return (std::uint32_t(bytes[0]) << 24u) | (std::uint32_t(bytes[1]) << 16u) | (std::uint32_t(bytes[2]) << 8u) | std::uint32_t(bytes[3]);
        }
        struct Sha1Compression{
            std::uint32_t state[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
            static std::uint32_t rotate_left(std::uint32_t value, unsigned bits){
                return (value << bits) | (value >> (32u - bits));
            }
            void process_block(const unsigned char* block){
                std::uint32_t schedule[80];
                for(unsigned i = 0; i < 16; i++){
                    schedule[i] = big_endian_word(block + 4 * i);
                }
                for(unsigned i = 16; i < 80; i++){
                    schedule[i] = rotate_left(schedule[i - 3] ^ schedule[i - 8] ^ schedule[i - 14] ^ schedule[i - 16], 1);
                }
                std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4];
                for(unsigned i = 0; i < 80; i++){
                    std::uint32_t f, k;
                    if(i < 20){
                        f = (b & c) | (~b & d);
                        k = 0x5A827999u;
                    }
                    else if(i < 40){
                        f = b ^ c ^ d;
                        k = 0x6ED9EBA1u;
                    }
                    else if(i < 60){
                        f = (b & c) | (b & d) | (c & d);
                        k = 0x8F1BBCDCu;
                    }
                    else{
                        f = b ^ c ^ d;
                        k = 0xCA62C1D6u;
                    }
                    std::uint32_t temporary = rotate_left(a, 5) + f + e + k + schedule[i];
                    e = d;
                    d = c;
                    c = rotate_left(b, 30);
                    b = a;
                    a = temporary;
                }
                state[0] += a;
                state[1] += b;
                state[2] += c;
                state[3] += d;
                state[4] += e;
            }
        };
        struct Sha256Compression{
            std::uint32_t state[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
            static std::uint32_t rotate_right(std::uint32_t value, unsigned bits){
                return (value >> bits) | (value << (32u - bits));
            }
            void process_block(const unsigned char* block){
                static constexpr std::uint32_t ROUND_CONSTANTS[64] = {
                    0x428A2F98u, 0x71374491u, 0xB5C0FBCFu, 0xE9B5DBA5u, 0x3956C25Bu, 0x59F111F1u, 0x923F82A4u, 0xAB1C5ED5u,
                    0xD807AA98u, 0x12835B01u, 0x243185BEu, 0x550C7DC3u, 0x72BE5D74u, 0x80DEB1FEu, 0x9BDC06A7u, 0xC19BF174u,
                    0xE49B69C1u, 0xEFBE4786u, 0x0FC19DC6u, 0x240CA1CCu, 0x2DE92C6Fu, 0x4A7484AAu, 0x5CB0A9DCu, 0x76F988DAu,
                    0x983E5152u, 0xA831C66Du, 0xB00327C8u, 0xBF597FC7u, 0xC6E00BF3u, 0xD5A79147u, 0x06CA6351u, 0x14292967u,
                    0x27B70A85u, 0x2E1B2138u, 0x4D2C6DFCu, 0x53380D13u, 0x650A7354u, 0x766A0ABBu, 0x81C2C92Eu, 0x92722C85u,
                    0xA2BFE8A1u, 0xA81A664Bu, 0xC24B8B70u, 0xC76C51A3u, 0xD192E819u, 0xD6990624u, 0xF40E3585u, 0x106AA070u,
                    0x19A4C116u, 0x1E376C08u, 0x2748774Cu, 0x34B0BCB5u, 0x391C0CB3u, 0x4ED8AA4Au, 0x5B9CCA4Fu, 0x682E6FF3u,
                    0x748F82EEu, 0x78A5636Fu, 0x84C87814u, 0x8CC70208u, 0x90BEFFFAu, 0xA4506CEBu, 0xBEF9A3F7u, 0xC67178F2u};
                std::uint32_t schedule[64];
                for(unsigned i = 0; i < 16; i++){
                    schedule[i] = big_endian_word(block + 4 * i);
                }
                for(unsigned i = 16; i < 64; i++){
                    std::uint32_t sigma0 = rotate_right(schedule[i - 15], 7) ^ rotate_right(schedule[i - 15], 18) ^ (schedule[i - 15] >> 3u);
                    std::uint32_t sigma1 = rotate_right(schedule[i - 2], 17) ^ rotate_right(schedule[i - 2], 19) ^ (schedule[i - 2] >> 10u);
                    schedule[i] = schedule[i - 16] + sigma0 + schedule[i - 7] + sigma1;
                }
                std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4], f = state[5], g = state[6], h = state[7];
                for(unsigned i = 0; i < 64; i++){
                    std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
                    std::uint32_t choice = (e & f) ^ (~e & g);
                    std::uint32_t temporary1 = h + sum1 + choice + ROUND_CONSTANTS[i] + schedule[i];
                    std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
                    std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
                    std::uint32_t temporary2 = sum0 + majority;
                    h = g;
                    g = f;
                    f = e;
                    e = d + temporary1;
                    d = c;
                    c = b;
                    b = a;
                    a = temporary1 + temporary2;
                }
                state[0] += a;
                state[1] += b;
                state[2] += c;
                state[3] += d;
                state[4] += e;
                state[5] += f;
                state[6] += g;
                state[7] += h;
            }
        };
        template <typename T_COMPRESSION>
        struct Hash{
            T_COMPRESSION compression;
            std::uint64_t message_length_bits = 0;
            unsigned char buffer[64];
            std::size_t buffer_size = 0;
            void update(const unsigned char* data, std::size_t size){
                message_length_bits += std::uint64_t(size) * 8u;
                while(size > 0){
                    std::size_t take = std::min(size, std::size_t(64) - buffer_size);
                    std::memcpy(buffer + buffer_size, data, take);
                    buffer_size += take;
                    data += take;
                    size -= take;
                    if(buffer_size == 64){
                        compression.process_block(buffer);
                        buffer_size = 0;
                    }
                }
            }
            std::string finalize(){
                std::uint64_t final_length_bits = message_length_bits; // update() below keeps counting, so the length is captured first
                unsigned char padding = 0x80;
                update(&padding, 1);
                unsigned char zero = 0x00;
                while(buffer_size != 56){
                    update(&zero, 1);
                }
                unsigned char length_bytes[8];
                for(unsigned i = 0; i < 8; i++){
                    length_bytes[i] = static_cast<unsigned char>((final_length_bits >> (56u - 8u * i)) & 0xFFu);
                }
                update(length_bytes, 8);
                return hex_words(compression.state);
            }
        };
        using Sha1 = Hash<Sha1Compression>;
        using Sha256 = Hash<Sha256Compression>;
        template <typename T_CONSUMER>
        bool read_file_chunks(const std::filesystem::path& path, T_CONSUMER&& consumer){
            std::ifstream file(path, std::ios::binary);
            if(!file.is_open()){
                return false;
            }
            std::vector<char> chunk(64 * 1024);
            while(file){
                file.read(chunk.data(), std::streamsize(chunk.size()));
                std::streamsize count = file.gcount();
                if(count > 0){
                    consumer(reinterpret_cast<const unsigned char*>(chunk.data()), std::size_t(count));
                }
            }
            return !file.bad();
        }
        inline bool sha1_file(const std::filesystem::path& path, std::string& hex_output){
            Sha1 sha1;
            if(!read_file_chunks(path, [&](const unsigned char* data, std::size_t size){ sha1.update(data, size); })){
                return false;
            }
            hex_output = sha1.finalize();
            return true;
        }
        inline bool digests_file(const std::filesystem::path& path, std::string& sha1_output, std::string& sha256_output){
            Sha1 sha1;
            Sha256 sha256;
            bool read = read_file_chunks(path, [&](const unsigned char* data, std::size_t size){
                sha1.update(data, size);
                sha256.update(data, size);
            });
            if(!read){
                return false;
            }
            sha1_output = sha1.finalize();
            sha256_output = sha256.finalize();
            return true;
        }
        inline std::string sha256_text(const std::string& text){
            Sha256 sha256;
            sha256.update(reinterpret_cast<const unsigned char*>(text.data()), text.size());
            return sha256.finalize();
        }
        inline bool read_file(const std::filesystem::path& path, std::string& output){
            std::ifstream file(path, std::ios::binary);
            if(!file.is_open()){
                return false;
            }
            output.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
            return !file.bad();
        }
        inline bool normalize_hex(std::string& text, std::size_t length){
            if(text.size() != length){
                return false;
            }
            for(char& character: text){
                if(character >= 'A' && character <= 'F'){
                    character = char(character - 'A' + 'a');
                }
                bool valid = (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
                if(!valid){
                    return false;
                }
            }
            return true;
        }
        inline bool normalize_hash(std::string& hash){
            return normalize_hex(hash, 40);
        }
        inline bool is_lowercase_hex(const std::string& text, std::size_t length){
            std::string normalized = text;
            return normalize_hex(normalized, length) && normalized == text;
        }
        inline bool normalize_reference(const std::string& reference, std::string& sha1_output, std::string& error_output){
            std::string hash = reference.rfind("conta:", 0) == 0 ? reference.substr(6) : reference;
            std::string as_sha256 = hash;
            if(normalize_hash(hash)){
                sha1_output = hash;
                return true;
            }
            if(normalize_hex(as_sha256, 64)){
                error_output = "conta: \"" + reference + "\" is a sha256, but conta references are sha1 (40 hexadecimal characters); the sha256 in the index only verifies downloads";
                return false;
            }
            error_output = "conta: invalid hash \"" + reference + "\": expected a sha1 (40 hexadecimal characters)";
            return false;
        }
        inline bool is_lfs_pointer(const std::filesystem::path& path){
            std::error_code error_code;
            std::uintmax_t size = std::filesystem::file_size(path, error_code);
            if(error_code || size >= 1024){
                return false;
            }
            constexpr char PREFIX[] = "version https://git-lfs";
            constexpr std::size_t PREFIX_LENGTH = sizeof(PREFIX) - 1;
            char head[PREFIX_LENGTH];
            std::ifstream file(path, std::ios::binary);
            if(!file.read(head, PREFIX_LENGTH)){
                return false;
            }
            return std::memcmp(head, PREFIX, PREFIX_LENGTH) == 0;
        }
        struct JsonReader{
            static constexpr int MAXIMUM_DEPTH = 64;
            const std::string& text;
            std::size_t position = 0;
            explicit JsonReader(const std::string& text_input): text(text_input){}
            void skip_whitespace(){
                while(position < text.size() && (text[position] == ' ' || text[position] == '\t' || text[position] == '\n' || text[position] == '\r')){
                    position++;
                }
            }
            bool next_is(char character){
                skip_whitespace();
                return position < text.size() && text[position] == character;
            }
            bool consume(char character){
                if(!next_is(character)){
                    return false;
                }
                position++;
                return true;
            }
            bool at_end(){
                skip_whitespace();
                return position == text.size();
            }
            bool digit_at(std::size_t index) const{
                return index < text.size() && text[index] >= '0' && text[index] <= '9';
            }
            bool consume_literal(const char* literal){
                std::size_t length = std::strlen(literal);
                if(text.compare(position, length, literal) != 0){
                    return false;
                }
                position += length;
                return true;
            }
            bool read_code_unit(std::uint32_t& value){
                if(position + 4 > text.size()){
                    return false;
                }
                value = 0;
                for(int digit_i = 0; digit_i < 4; digit_i++){
                    char character = text[position++];
                    value <<= 4u;
                    if(character >= '0' && character <= '9'){
                        value |= std::uint32_t(character - '0');
                    }
                    else if(character >= 'a' && character <= 'f'){
                        value |= std::uint32_t(character - 'a' + 10);
                    }
                    else if(character >= 'A' && character <= 'F'){
                        value |= std::uint32_t(character - 'A' + 10);
                    }
                    else{
                        return false;
                    }
                }
                return true;
            }
            static void append_utf8(std::string& output, std::uint32_t code_point){
                if(code_point < 0x80u){
                    output += char(code_point);
                }
                else if(code_point < 0x800u){
                    output += char(0xC0u | (code_point >> 6u));
                    output += char(0x80u | (code_point & 0x3Fu));
                }
                else if(code_point < 0x10000u){
                    output += char(0xE0u | (code_point >> 12u));
                    output += char(0x80u | ((code_point >> 6u) & 0x3Fu));
                    output += char(0x80u | (code_point & 0x3Fu));
                }
                else{
                    output += char(0xF0u | (code_point >> 18u));
                    output += char(0x80u | ((code_point >> 12u) & 0x3Fu));
                    output += char(0x80u | ((code_point >> 6u) & 0x3Fu));
                    output += char(0x80u | (code_point & 0x3Fu));
                }
            }
            bool parse_string(std::string& output){
                if(!consume('"')){
                    return false;
                }
                output.clear();
                while(position < text.size()){
                    unsigned char character = static_cast<unsigned char>(text[position++]);
                    if(character == '"'){
                        return true;
                    }
                    if(character < 0x20u){
                        return false;
                    }
                    if(character != '\\'){
                        output += char(character);
                        continue;
                    }
                    if(position >= text.size()){
                        return false;
                    }
                    char escape = text[position++];
                    switch(escape){
                        case '"': output += '"'; break;
                        case '\\': output += '\\'; break;
                        case '/': output += '/'; break;
                        case 'b': output += '\b'; break;
                        case 'f': output += '\f'; break;
                        case 'n': output += '\n'; break;
                        case 'r': output += '\r'; break;
                        case 't': output += '\t'; break;
                        case 'u': {
                            std::uint32_t code_point;
                            if(!read_code_unit(code_point)){
                                return false;
                            }
                            if(code_point >= 0xD800u && code_point <= 0xDBFFu){
                                std::size_t high_end = position;
                                std::uint32_t low = 0;
                                if(text.compare(position, 2, "\\u") == 0){
                                    position += 2;
                                }
                                if(position != high_end && read_code_unit(low) && low >= 0xDC00u && low <= 0xDFFFu){
                                    code_point = 0x10000u + ((code_point - 0xD800u) << 10u) + (low - 0xDC00u);
                                }
                                else{
                                    position = high_end;
                                    code_point = 0xFFFDu;
                                }
                            }
                            else if(code_point >= 0xDC00u && code_point <= 0xDFFFu){
                                code_point = 0xFFFDu;
                            }
                            append_utf8(output, code_point);
                            break;
                        }
                        default: return false;
                    }
                }
                return false;
            }
            bool parse_number(std::string& token){
                skip_whitespace();
                std::size_t start = position;
                if(position < text.size() && text[position] == '-'){
                    position++;
                }
                if(position < text.size() && text[position] == '0'){
                    position++;
                }
                else if(digit_at(position)){
                    while(digit_at(position)){
                        position++;
                    }
                }
                else{
                    return false;
                }
                if(position < text.size() && text[position] == '.'){
                    position++;
                    if(!digit_at(position)){
                        return false;
                    }
                    while(digit_at(position)){
                        position++;
                    }
                }
                if(position < text.size() && (text[position] == 'e' || text[position] == 'E')){
                    position++;
                    if(position < text.size() && (text[position] == '+' || text[position] == '-')){
                        position++;
                    }
                    if(!digit_at(position)){
                        return false;
                    }
                    while(digit_at(position)){
                        position++;
                    }
                }
                token = text.substr(start, position - start);
                return true;
            }
            template <typename T_CALLBACK>
            bool parse_object(int depth, T_CALLBACK&& callback){
                if(depth > MAXIMUM_DEPTH || !consume('{')){
                    return false;
                }
                if(consume('}')){
                    return true;
                }
                do{
                    std::string key;
                    if(!parse_string(key) || !consume(':') || !callback(key)){
                        return false;
                    }
                } while(consume(','));
                return consume('}');
            }
            template <typename T_CALLBACK>
            bool parse_array(int depth, T_CALLBACK&& callback){
                if(depth > MAXIMUM_DEPTH || !consume('[')){
                    return false;
                }
                if(consume(']')){
                    return true;
                }
                do{
                    if(!callback()){
                        return false;
                    }
                } while(consume(','));
                return consume(']');
            }
            bool skip_value(int depth){
                if(depth > MAXIMUM_DEPTH){
                    return false;
                }
                skip_whitespace();
                if(position >= text.size()){
                    return false;
                }
                char character = text[position];
                if(character == '{'){
                    return parse_object(depth, [&](const std::string&){ return skip_value(depth + 1); });
                }
                if(character == '['){
                    return parse_array(depth, [&](){ return skip_value(depth + 1); });
                }
                if(character == '"'){
                    std::string ignored;
                    return parse_string(ignored);
                }
                if(character == 't'){
                    return consume_literal("true");
                }
                if(character == 'f'){
                    return consume_literal("false");
                }
                if(character == 'n'){
                    return consume_literal("null");
                }
                std::string ignored;
                return parse_number(ignored);
            }
            bool parse_string_or_skip(std::string& output, int depth){
                if(next_is('"')){
                    return parse_string(output);
                }
                output.clear();
                return skip_value(depth);
            }
            bool parse_number_or_skip(std::string& token, int depth){
                skip_whitespace();
                if(position < text.size() && (text[position] == '-' || digit_at(position))){
                    return parse_number(token);
                }
                token.clear();
                return skip_value(depth);
            }
        };
        struct Meta{
            bool format_supported = false;
            std::string location;
            std::string sha256;
        };
        struct ShardLocation{
            std::string id;
            std::vector<std::string> compressed;
        };
        struct Entry{
            std::string sha256;
            std::vector<ShardLocation> shards;
            std::string description;
            std::string notice;
        };
        struct Index{
            std::string source;
            std::string sha256;
            std::size_t entry_count = 0;
            std::map<std::string, std::string> shard_urls;
            std::map<std::string, Entry> entries;
        };
        inline bool parse_meta(const std::string& text, Meta& meta){
            JsonReader reader(text);
            bool parsed = reader.parse_object(0, [&](const std::string& key){
                if(key == "format"){
                    std::string token;
                    if(!reader.parse_number_or_skip(token, 1)){
                        return false;
                    }
                    meta.format_supported = token == "1";
                    return true;
                }
                if(key == "index" && reader.next_is('{')){
                    return reader.parse_object(1, [&](const std::string& index_key){
                        if(index_key == "location"){
                            return reader.parse_string_or_skip(meta.location, 2);
                        }
                        if(index_key == "sha256"){
                            return reader.parse_string_or_skip(meta.sha256, 2);
                        }
                        return reader.skip_value(2);
                    });
                }
                return reader.skip_value(1);
            });
            return parsed && reader.at_end();
        }
        inline bool parse_index(const std::string& text, const std::set<std::string>& wanted, Index& index){
            JsonReader reader(text);
            bool has_shards = false;
            bool has_entries = false;
            bool parsed = reader.parse_object(0, [&](const std::string& key){
                if(key == "shards" && reader.next_is('{')){
                    has_shards = true;
                    return reader.parse_object(1, [&](const std::string& shard){
                        if(!reader.next_is('{')){
                            return reader.skip_value(2);
                        }
                        std::string url;
                        bool shard_parsed = reader.parse_object(2, [&](const std::string& shard_key){
                            if(shard_key == "url"){
                                return reader.parse_string_or_skip(url, 3);
                            }
                            return reader.skip_value(3);
                        });
                        if(shard_parsed && !url.empty()){
                            index.shard_urls[shard] = url;
                        }
                        return shard_parsed;
                    });
                }
                if(key == "entries" && reader.next_is('[')){
                    has_entries = true;
                    return reader.parse_array(1, [&](){
                        if(!reader.next_is('{')){
                            return reader.skip_value(2);
                        }
                        std::string sha1;
                        Entry entry;
                        bool entry_parsed = reader.parse_object(2, [&](const std::string& entry_key){
                            if(entry_key == "sha1"){
                                return reader.parse_string_or_skip(sha1, 3);
                            }
                            if(entry_key == "sha256"){
                                return reader.parse_string_or_skip(entry.sha256, 3);
                            }
                            if(entry_key == "description"){
                                return reader.parse_string_or_skip(entry.description, 3);
                            }
                            if(entry_key == "notice"){
                                return reader.parse_string_or_skip(entry.notice, 3);
                            }
                            if(entry_key == "shards" && reader.next_is('[')){
                                entry.shards.clear();
                                return reader.parse_array(3, [&](){
                                    if(!reader.next_is('{')){
                                        return reader.skip_value(4);
                                    }
                                    ShardLocation shard;
                                    bool shard_parsed = reader.parse_object(4, [&](const std::string& shard_key){
                                        if(shard_key == "id"){
                                            return reader.parse_string_or_skip(shard.id, 5);
                                        }
                                        if(shard_key == "compressed" && reader.next_is('[')){
                                            return reader.parse_array(5, [&](){
                                                std::string compression;
                                                if(!reader.parse_string_or_skip(compression, 6)){
                                                    return false;
                                                }
                                                if(!compression.empty()){
                                                    shard.compressed.push_back(compression);
                                                }
                                                return true;
                                            });
                                        }
                                        return reader.skip_value(5);
                                    });
                                    if(!shard_parsed){
                                        return false;
                                    }
                                    if(!shard.id.empty()){
                                        entry.shards.push_back(std::move(shard));
                                    }
                                    return true;
                                });
                            }
                            return reader.skip_value(3);
                        });
                        if(!entry_parsed){
                            return false;
                        }
                        if(is_lowercase_hex(sha1, 40) && is_lowercase_hex(entry.sha256, 64)){
                            index.entry_count++;
                            if(wanted.count(sha1) > 0 && index.entries.count(sha1) == 0){
                                index.entries.emplace(sha1, std::move(entry));
                            }
                        }
                        return true;
                    });
                }
                return reader.skip_value(1);
            });
            return parsed && reader.at_end() && has_shards && has_entries;
        }
        inline bool is_alphanumeric(char character){
            return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9');
        }
        inline bool is_document_url(const std::string& url){
            std::size_t start;
            if(url.rfind("https://", 0) == 0){
                start = 8;
            }
            else if(url.rfind("http://", 0) == 0){
                start = 7;
            }
            else if(url.rfind("file:///", 0) == 0){
                start = 8;
            }
            else{
                return false;
            }
            if(start >= url.size() || !is_alphanumeric(url[start])){
                return false;
            }
            for(std::size_t character_i = start; character_i < url.size(); character_i++){
                char character = url[character_i];
                if(!is_alphanumeric(character) && (character == '\0' || std::strchr("._~:/?&#=+,;@-", character) == nullptr)){
                    return false;
                }
            }
            return true;
        }
        inline bool resolve_location(const std::string& entry_point, const std::string& location, std::string& url_output){
            if(location.find("://") != std::string::npos){
                if(!is_document_url(location)){
                    return false;
                }
                url_output = location;
                return true;
            }
            if(location.empty() || location[0] == '/' || location.find("..") != std::string::npos){
                return false;
            }
            for(char character: location){
                if(!is_alphanumeric(character) && character != '.' && character != '_' && character != '~' && character != '-' && character != '/'){
                    return false;
                }
            }
            std::size_t slash = entry_point.rfind('/');
            if(slash == std::string::npos){
                return false;
            }
            url_output = entry_point.substr(0, slash + 1) + location;
            return true;
        }
        inline void replace_all(std::string& text, const std::string& pattern, const std::string& replacement){
            for(std::size_t position = text.find(pattern); position != std::string::npos; position = text.find(pattern, position + replacement.size())){
                text.replace(position, pattern.size(), replacement);
            }
        }
        inline bool shard_location(const std::string& url_template, const std::string& sha1, const std::string& sha256, std::string& url_output){
            std::string url = url_template;
            replace_all(url, "{sha1}", sha1);
            replace_all(url, "{sha256}", sha256);
            if(url.find_first_of("{}") != std::string::npos || !is_document_url(url)){
                return false;
            }
            url_output = url;
            return true;
        }
        inline std::string printable(const std::string& text){
            std::string output;
            for(char character: text.substr(0, 4096)){
                unsigned char byte = static_cast<unsigned char>(character);
                output += byte >= 0x20u && byte <= 0x7Eu ? character : '?';
            }
            return output;
        }
        inline bool shell_quote(const std::string& input, std::string& output){
#if defined(_WIN32)
            if(input.find_first_of("\"%!") != std::string::npos){
                return false;
            }
            output = "\"" + input + "\"";
#else
            if(input.find('\'') != std::string::npos){
                return false;
            }
            output = "'" + input + "'";
#endif
            return true;
        }
        inline long process_id(){
#if defined(_WIN32)
            return _getpid();
#else
            return long(getpid());
#endif
        }
        inline std::string unique_suffix(){
            std::random_device device;
            char random_hex[9];
            std::snprintf(random_hex, sizeof(random_hex), "%08x", static_cast<unsigned int>(device()));
            return std::to_string(process_id()) + "-" + random_hex;
        }
        inline int command_exit_code(int status){
#if defined(_WIN32)
            return status;
#else
            if(status != -1 && WIFSIGNALED(status)){
                int signal_number = WTERMSIG(status);
                if(signal_number == SIGINT || signal_number == SIGQUIT){ // std::system ignores both in the caller, so a Ctrl-C that killed curl is re-raised here
                    std::raise(signal_number);
                }
                return 128 + signal_number;
            }
            if(status == -1 || !WIFEXITED(status)){
                return -1;
            }
            return WEXITSTATUS(status);
#endif
        }
        inline std::string curl_failure(int code){
            std::string described = "curl exit status " + std::to_string(code);
            if(code == 127 || code == 9009){
                return described + ": is curl installed and on the PATH?";
            }
            if(code == 22){
                return described + ": HTTP error, the host does not serve it";
            }
            if(code == 37){
                return described + ": file not found";
            }
            if(code == 6 || code == 7 || code == 28){
                return described + ": host unreachable or transfer stalled";
            }
            if(code == 35 || code == 51 || code == 53 || code == 58 || code == 59 || code == 60 || code == 77 || code == 83 || code == 90 || code == 91){
                return described + ": TLS failed; in an old environment point CURL_CA_BUNDLE or SSL_CERT_FILE at a current CA bundle";
            }
            return described;
        }
        inline bool run_curl(const std::string& url, const std::filesystem::path& output, bool quiet, std::string& failure_output){
            std::string quoted_url, quoted_output;
            if(!shell_quote(url, quoted_url)){
                failure_output = "unsupported character in the URL";
                return false;
            }
            if(!shell_quote(output.string(), quoted_output)){
                failure_output = "unsupported character in the path " + output.string();
                return false;
            }
            std::string command = std::string("curl -f -L -g ") + (quiet ? "-sS " : "") + "--retry 3 --connect-timeout 15 --speed-limit 1024 --speed-time 60 -o " + quoted_output + " " + quoted_url;
            int code = command_exit_code(std::system(command.c_str()));
            if(code != 0){
                failure_output = curl_failure(code);
                return false;
            }
            return true;
        }
        constexpr bool gzip_supported(){
#if defined(RL_TOOLS_ENABLE_ZLIB) || defined(CONTA_ENABLE_ZLIB)
            return true;
#else
            return false;
#endif
        }
        inline bool verify_digests(const std::string& actual_sha1, const std::string& actual_sha256, const std::string& sha1, const std::string& sha256, std::string& failure_output){
            if(actual_sha1 != sha1){
                failure_output = "hash mismatch: the content hashes to " + actual_sha1 + ", not caching";
                return false;
            }
            if(!sha256.empty() && actual_sha256 != sha256){
                failure_output = "sha256 mismatch: the index lists " + sha256 + " but the content hashes to " + actual_sha256 + ", not caching";
                return false;
            }
            return true;
        }
        inline bool same_file_contents(const std::filesystem::path& left, const std::filesystem::path& right){
            std::error_code error_code;
            if(!std::filesystem::is_regular_file(right, error_code)){
                return false;
            }
            std::uintmax_t left_size = std::filesystem::file_size(left, error_code);
            if(error_code){
                return false;
            }
            std::uintmax_t right_size = std::filesystem::file_size(right, error_code);
            if(error_code || left_size != right_size){
                return false;
            }
            std::ifstream left_file(left, std::ios::binary), right_file(right, std::ios::binary);
            if(!left_file.is_open() || !right_file.is_open()){
                return false;
            }
            std::vector<char> left_chunk(64 * 1024), right_chunk(64 * 1024);
            while(left_file && right_file){
                left_file.read(left_chunk.data(), std::streamsize(left_chunk.size()));
                right_file.read(right_chunk.data(), std::streamsize(right_chunk.size()));
                if(left_file.gcount() != right_file.gcount() || !std::equal(left_chunk.begin(), left_chunk.begin() + left_file.gcount(), right_chunk.begin())){
                    return false;
                }
            }
            return left_file.eof() && right_file.eof() && !left_file.bad() && !right_file.bad();
        }
        inline bool publish_file(const std::filesystem::path& temporary, const std::filesystem::path& target, std::string& failure_output){
            std::error_code error_code;
            std::filesystem::rename(temporary, target, error_code);
            if(error_code){
                if(same_file_contents(temporary, target)){
                    std::filesystem::remove(temporary, error_code);
                    return true;
                }
                failure_output = "failed to move " + temporary.string() + " to " + target.string() + ": " + error_code.message();
                std::filesystem::remove(temporary, error_code);
                return false;
            }
            return true;
        }
        template <typename T_CONSUMER>
        bool read_gzip_chunks(const std::filesystem::path& path, T_CONSUMER&& consumer, std::string& failure_output){
#if defined(RL_TOOLS_ENABLE_ZLIB) || defined(CONTA_ENABLE_ZLIB)
            std::ifstream file(path, std::ios::binary);
            if(!file.is_open()){
                failure_output = "the gzip file is unreadable";
                return false;
            }
            struct Inflater{
                z_stream stream{};
                bool initialized = false;
                ~Inflater(){
                    if(initialized){
                        inflateEnd(&stream);
                    }
                }
            } inflater;
            z_stream& stream = inflater.stream;
            if(inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK){
                failure_output = "failed to initialize gzip decoding";
                return false;
            }
            inflater.initialized = true;
            std::vector<unsigned char> input(64 * 1024), output(64 * 1024);
            bool member_end = false;
            while(true){
                if(stream.avail_in == 0 && !file.eof()){
                    file.read(reinterpret_cast<char*>(input.data()), std::streamsize(input.size()));
                    if(file.bad()){
                        failure_output = "failed to read the gzip file";
                        return false;
                    }
                    stream.next_in = input.data();
                    stream.avail_in = uInt(file.gcount());
                }
                if(member_end){
                    while(stream.avail_in > 0 && *stream.next_in == 0){
                        stream.next_in++;
                        stream.avail_in--;
                    }
                    if(stream.avail_in == 0){
                        if(file.eof()){
                            return true;
                        }
                        continue;
                    }
                    Bytef* next_in = stream.next_in;
                    uInt available = stream.avail_in;
                    if(inflateReset2(&stream, 16 + MAX_WBITS) != Z_OK){
                        failure_output = "failed to reset gzip decoding";
                        return false;
                    }
                    stream.next_in = next_in;
                    stream.avail_in = available;
                    member_end = false;
                }
                stream.next_out = output.data();
                stream.avail_out = uInt(output.size());
                int status = inflate(&stream, Z_NO_FLUSH);
                if(status != Z_OK && status != Z_STREAM_END){
                    failure_output = status == Z_BUF_ERROR ? "truncated gzip file" : "invalid gzip file";
                    if(stream.msg != nullptr){
                        failure_output += ": " + std::string(stream.msg);
                    }
                    return false;
                }
                std::size_t count = output.size() - stream.avail_out;
                if(count > 0 && !consumer(output.data(), count)){
                    failure_output = "failed to write the decompressed content";
                    return false;
                }
                member_end = status == Z_STREAM_END;
            }
#else
            (void)path;
            (void)consumer;
            failure_output = "gzip decoding is unavailable; build with CONTA_ENABLE_ZLIB and link zlib, or provide the plain blob";
            return false;
#endif
        }
        inline bool decode_gzip(const std::filesystem::path& source, const std::string& sha1, const std::string& sha256, const std::filesystem::path& target, std::vector<std::uint8_t>* bytes_output, std::string& failure_output){
            std::filesystem::path temporary;
            std::ofstream file;
            std::vector<std::uint8_t> bytes;
            if(bytes_output == nullptr){
                temporary = target;
                temporary += ".partial." + unique_suffix();
                file.open(temporary, std::ios::binary);
                if(!file.is_open()){
                    failure_output = "failed to create " + temporary.string();
                    return false;
                }
            }
            Sha1 actual_sha1;
            Sha256 actual_sha256;
            bool decoded = read_gzip_chunks(source, [&](const unsigned char* data, std::size_t size){
                actual_sha1.update(data, size);
                actual_sha256.update(data, size);
                if(bytes_output != nullptr){
                    bytes.insert(bytes.end(), data, data + size);
                    return true;
                }
                file.write(reinterpret_cast<const char*>(data), std::streamsize(size));
                return bool(file);
            }, failure_output);
            if(decoded){
                decoded = verify_digests(actual_sha1.finalize(), actual_sha256.finalize(), sha1, sha256, failure_output);
            }
            if(bytes_output == nullptr){
                file.close();
                if(decoded && !file){
                    failure_output = "failed to close " + temporary.string();
                    decoded = false;
                }
                if(decoded){
                    return publish_file(temporary, target, failure_output);
                }
                std::error_code error_code;
                std::filesystem::remove(temporary, error_code);
            }
            else if(decoded){
                bytes_output->swap(bytes);
            }
            return decoded;
        }
        inline std::string compressed_location(const std::string& url){
            std::size_t query = url.find_first_of("?#");
            return url.substr(0, query) + ".gz" + (query == std::string::npos ? std::string() : url.substr(query));
        }
        inline bool fetch_gzip(const std::string& url, const std::string& sha1, const std::string& sha256, const std::string& description, const std::filesystem::path& target, std::vector<std::uint8_t>* bytes_output, std::string& failure_output){
            std::filesystem::path gzip_target = target;
            gzip_target += ".gz";
            std::filesystem::path temporary = gzip_target;
            temporary += ".partial." + unique_suffix();
            std::cerr << "conta: downloading " << sha1 << (description.empty() ? std::string() : " (" + printable(description) + ")") << " from " << url << std::endl;
            bool fetched = run_curl(url, temporary, false, failure_output) && decode_gzip(temporary, sha1, sha256, target, bytes_output, failure_output);
            if(fetched && bytes_output != nullptr){
                fetched = publish_file(temporary, gzip_target, failure_output);
                if(!fetched){
                    bytes_output->clear();
                }
            }
            std::error_code error_code;
            std::filesystem::remove(temporary, error_code);
            return fetched;
        }
        inline bool fetch_blob(const std::string& url, const std::string& sha1, const std::string& sha256, const std::string& description, const std::filesystem::path& target, std::string& failure_output){
            std::filesystem::path temporary = target;
            temporary += ".partial." + unique_suffix();
            std::cerr << "conta: downloading " << sha1 << (description.empty() ? std::string() : " (" + printable(description) + ")") << " from " << url << std::endl;
            std::error_code error_code;
            if(!run_curl(url, temporary, false, failure_output)){
                std::filesystem::remove(temporary, error_code);
                return false;
            }
            std::string actual_sha1, actual_sha256;
            if(!digests_file(temporary, actual_sha1, actual_sha256)){
                std::filesystem::remove(temporary, error_code);
                failure_output = "the downloaded file is unreadable";
                return false;
            }
            if(!verify_digests(actual_sha1, actual_sha256, sha1, sha256, failure_output)){
                std::filesystem::remove(temporary, error_code);
                return false;
            }
            return publish_file(temporary, target, failure_output);
        }
        inline bool read_blob(const std::filesystem::path& path, std::vector<std::uint8_t>* bytes_output, std::string& failure_output){
            if(bytes_output == nullptr){
                return true;
            }
            std::vector<std::uint8_t> bytes;
            if(!read_file_chunks(path, [&](const unsigned char* data, std::size_t size){ bytes.insert(bytes.end(), data, data + size); })){
                failure_output = "conta: cannot read " + path.string();
                return false;
            }
            bytes_output->swap(bytes);
            return true;
        }
        struct Session{
            std::set<std::string> wanted;
            std::vector<Index> indexes;
            std::set<std::string> attempted_urls;
            std::size_t next_entry_point = 0;
            bool cached_index_loaded = false;
            std::string cached_index_sha256;
            std::size_t cached_index_entry_count = 0;
        };
        inline const Entry* find_entry(const Session& session, const std::string& sha1){
            for(const Index& index: session.indexes){
                auto entry = index.entries.find(sha1);
                if(entry != index.entries.end()){
                    return &entry->second;
                }
            }
            return nullptr;
        }
        inline bool try_index(Session& session, const Index& index, const std::string& sha1, const std::filesystem::path& target, std::vector<std::uint8_t>* bytes_output, std::vector<std::string>& report){
            auto found = index.entries.find(sha1);
            if(found == index.entries.end()){
                report.push_back(index.source + ": does not list it");
                return false;
            }
            const Entry& entry = found->second;
            if(entry.shards.empty()){
                report.push_back(index.source + ": lists no location");
            }
            for(const ShardLocation& shard: entry.shards){
                auto url_template = index.shard_urls.find(shard.id);
                if(url_template == index.shard_urls.end()){
                    report.push_back(index.source + ": shard " + shard.id + " is not in its shard table");
                    continue;
                }
                std::string url;
                if(!shard_location(url_template->second, sha1, entry.sha256, url)){
                    report.push_back(index.source + ": skipped shard " + shard.id + ", unsupported url " + url_template->second);
                    continue;
                }
                std::string failure;
                if(gzip_supported() && std::find(shard.compressed.begin(), shard.compressed.end(), "gz") != shard.compressed.end()){
                    std::string gzip_url = compressed_location(url);
                    if(session.attempted_urls.insert(gzip_url).second){
                        if(fetch_gzip(gzip_url, sha1, entry.sha256, entry.description, target, bytes_output, failure)){
                            return true;
                        }
                        report.push_back(gzip_url + ": " + failure);
                    }
                }
                if(!session.attempted_urls.insert(url).second){
                    continue;
                }
                if(fetch_blob(url, sha1, entry.sha256, entry.description, target, failure) && read_blob(target, bytes_output, failure)){
                    return true;
                }
                report.push_back(url + ": " + failure);
            }
            return false;
        }
        inline void load_cached_index(const Config& config, Session& session){
            session.cached_index_loaded = true;
            std::filesystem::path path = std::filesystem::path(config.cache) / CACHED_INDEX_NAME;
            std::string text;
            if(!read_file(path, text)){
                return;
            }
            Index index;
            index.source = path.string();
            index.sha256 = sha256_text(text);
            if(!parse_index(text, session.wanted, index)){
                return;
            }
            session.cached_index_sha256 = index.sha256;
            session.cached_index_entry_count = index.entry_count;
            session.indexes.push_back(std::move(index));
        }
        inline bool fetch_text(const Config& config, const std::string& url, const std::string& name, std::string& text_output, std::string& failure_output){
            std::filesystem::path temporary = std::filesystem::path(config.cache) / (name + ".partial." + unique_suffix());
            std::error_code error_code;
            bool fetched = run_curl(url, temporary, true, failure_output);
            if(fetched && !read_file(temporary, text_output)){
                failure_output = "the downloaded file is unreadable";
                fetched = false;
            }
            std::filesystem::remove(temporary, error_code);
            return fetched;
        }
        inline bool visit_entry_point(const Config& config, Session& session, const std::string& sha1, const std::filesystem::path& target, std::vector<std::uint8_t>* bytes_output, std::vector<std::string>& report){
            const std::string& entry_point = config.entry_points[session.next_entry_point++];
            std::cerr << "conta: refreshing the index from " << entry_point << std::endl;
            std::string meta_text, failure;
            if(!fetch_text(config, entry_point, "meta.json", meta_text, failure)){
                report.push_back(entry_point + ": " + failure);
                return false;
            }
            Meta meta;
            if(!parse_meta(meta_text, meta) || !meta.format_supported || !is_lowercase_hex(meta.sha256, 64)){
                report.push_back(entry_point + ": not a format 1 meta.json");
                return false;
            }
            for(const Index& index: session.indexes){
                if(index.sha256 == meta.sha256){
                    report.push_back(entry_point + ": serves the index already searched");
                    return false;
                }
            }
            std::string index_url;
            if(!resolve_location(entry_point, meta.location, index_url)){
                report.push_back(entry_point + ": unsupported index.location " + meta.location);
                return false;
            }
            std::string text;
            if(!fetch_text(config, index_url, CACHED_INDEX_NAME, text, failure)){
                report.push_back(index_url + ": " + failure);
                return false;
            }
            Index index;
            index.source = index_url;
            index.sha256 = sha256_text(text);
            if(!parse_index(text, session.wanted, index)){
                report.push_back(index_url + ": not a format 1 index");
                return false;
            }
            if(index.sha256 != meta.sha256){
                report.push_back(index_url + ": differs from the sha256 in meta.json (a publish may still be propagating), used for this call only");
            }
            else if(session.cached_index_sha256.empty() || index.entry_count >= session.cached_index_entry_count){ // entries are never removed, so a smaller index is older and must not replace the cached one
                std::filesystem::path temporary = std::filesystem::path(config.cache) / (std::string(CACHED_INDEX_NAME) + ".partial." + unique_suffix());
                std::error_code error_code;
                {
                    std::ofstream file(temporary, std::ios::binary);
                    file.write(text.data(), std::streamsize(text.size()));
                }
                std::filesystem::rename(temporary, std::filesystem::path(config.cache) / CACHED_INDEX_NAME, error_code);
                if(error_code){
                    std::filesystem::remove(temporary, error_code);
                }
                else{
                    session.cached_index_sha256 = index.sha256;
                    session.cached_index_entry_count = index.entry_count;
                }
            }
            session.indexes.push_back(std::move(index));
            return try_index(session, session.indexes.back(), sha1, target, bytes_output, report);
        }
        inline std::string failure_message(const Session& session, const std::string& sha1, const std::vector<std::string>& report){
            std::string message = "conta: cannot resolve " + sha1;
            const Entry* entry = find_entry(session, sha1);
            if(entry == nullptr){
                message += ", which no reachable index lists";
            }
            else if(!entry->description.empty()){
                message += " (" + printable(entry->description) + ")";
            }
            if(entry != nullptr && !entry->notice.empty()){
                message += "\nnotice: " + printable(entry->notice);
            }
            for(const std::string& line: report){
                message += "\n  " + printable(line);
            }
            message += "\nA copy you already have works with CONTA_URL=file:///<directory>/ holding it as " + sha1 + " (verified), or with CONTA_ROOT=<directory> holding it as data/" + sha1 + " (no network).";
            return message;
        }
#if defined(_WIN32)
        constexpr const char* CACHE_DIRECTORY_VARIABLES = "CONTA_CACHE or LOCALAPPDATA";
#else
        constexpr const char* CACHE_DIRECTORY_VARIABLES = "CONTA_CACHE, XDG_CACHE_HOME, or HOME";
#endif
        inline bool prepare_cache(const Config& config, std::string& error_output){
            if(config.cache.empty()){
                error_output = std::string("conta: cannot determine cache directory: set ") + CACHE_DIRECTORY_VARIABLES;
                return false;
            }
            std::error_code error_code;
            std::filesystem::create_directories(config.cache, error_code);
            if(error_code){
                error_output = "conta: failed to create cache directory " + config.cache + ": " + error_code.message();
                return false;
            }
            return true;
        }
        inline bool resolve_one(const Config& config, Session& session, const std::string& reference, std::string& path_output, std::string& error_output, std::vector<std::uint8_t>* bytes_output = nullptr){
            std::string sha1;
            if(!normalize_reference(reference, sha1, error_output)){
                return false;
            }
            if(!config.root.empty()){
                std::filesystem::path path = std::filesystem::path(config.root) / "data" / sha1;
                if(std::filesystem::exists(path)){
                    if(is_lfs_pointer(path)){
                        error_output = "conta: " + path.string() + " is a git-lfs pointer file, not the blob. Run \"git lfs pull\" in " + config.root + ".";
                        return false;
                    }
                    path_output = path.string();
                    return read_blob(path, bytes_output, error_output);
                }
                std::filesystem::path gzip_path = path;
                gzip_path += ".gz";
                if(!std::filesystem::exists(gzip_path)){
                    error_output = "conta: " + sha1 + " not found at " + path.string() + " or " + gzip_path.string() + ". CONTA_ROOT points to a read-only store, so downloading is disabled; populate the store or unset CONTA_ROOT to enable the download cache.";
                    return false;
                }
                if(is_lfs_pointer(gzip_path)){
                    error_output = "conta: " + gzip_path.string() + " is a git-lfs pointer file, not the blob. Run \"git lfs pull\" in " + config.root + ".";
                    return false;
                }
                std::filesystem::path target = std::filesystem::path(config.cache) / sha1;
                if(!config.cache.empty() && std::filesystem::exists(target)){
                    path_output = target.string();
                    return read_blob(target, bytes_output, error_output);
                }
                if(!gzip_supported()){
                    error_output = "conta: " + gzip_path.string() + " requires gzip decoding; build with CONTA_ENABLE_ZLIB and link zlib, or provide the plain blob in CONTA_ROOT";
                    return false;
                }
                if(bytes_output == nullptr && !prepare_cache(config, error_output)){
                    return false;
                }
                std::string failure;
                if(!decode_gzip(gzip_path, sha1, "", target, bytes_output, failure)){
                    error_output = "conta: cannot decode " + gzip_path.string() + ": " + failure;
                    return false;
                }
                path_output = target.string();
                return true;
            }
            if(config.cache.empty()){
                error_output = std::string("conta: cannot determine cache directory: set ") + CACHE_DIRECTORY_VARIABLES;
                return false;
            }
            std::filesystem::path target = std::filesystem::path(config.cache) / sha1;
            if(std::filesystem::exists(target)){
                path_output = target.string();
                return read_blob(target, bytes_output, error_output);
            }
            if(!prepare_cache(config, error_output)){
                return false;
            }
            std::vector<std::string> report;
            std::filesystem::path gzip_path = target;
            gzip_path += ".gz";
            if(std::filesystem::exists(gzip_path)){
                if(config.url_base.empty() && !session.cached_index_loaded){
                    load_cached_index(config, session);
                }
                const Entry* entry = find_entry(session, sha1);
                std::string failure;
                if(decode_gzip(gzip_path, sha1, entry == nullptr ? "" : entry->sha256, target, bytes_output, failure)){
                    path_output = target.string();
                    return true;
                }
                report.push_back(gzip_path.string() + ": " + failure);
            }
            if(!config.url_base.empty()){
                std::string url = config.url_base + sha1;
                std::string failure;
                if(!fetch_blob(url, sha1, "", "", target, failure) || !read_blob(target, bytes_output, failure)){
                    error_output = "conta: download of " + sha1 + " from " + url + " failed: " + failure;
                    for(const std::string& line: report){
                        error_output += "\n  " + line;
                    }
                    return false;
                }
                path_output = target.string();
                return true;
            }
            if(!session.cached_index_loaded){
                load_cached_index(config, session);
            }
            for(const Index& index: session.indexes){
                if(try_index(session, index, sha1, target, bytes_output, report)){
                    path_output = target.string();
                    return true;
                }
            }
            while(session.next_entry_point < config.entry_points.size()){
                if(visit_entry_point(config, session, sha1, target, bytes_output, report)){
                    path_output = target.string();
                    return true;
                }
            }
            error_output = failure_message(session, sha1, report);
            return false;
        }
    }
    inline bool resolve(const Config& config, const std::string& hash, std::string& path_output, std::string& error_output){
        detail::Session session;
        std::string sha1, ignored;
        if(detail::normalize_reference(hash, sha1, ignored)){
            session.wanted.insert(sha1);
        }
        return detail::resolve_one(config, session, hash, path_output, error_output);
    }
    inline bool load(const Config& config, const std::string& hash, std::vector<std::uint8_t>& bytes_output, std::string& error_output){
        bytes_output.clear();
        std::string path;
        if(!config.compressed_cache){
            return resolve(config, hash, path, error_output) && detail::read_blob(path, &bytes_output, error_output);
        }
        detail::Session session;
        std::string sha1, ignored;
        if(detail::normalize_reference(hash, sha1, ignored)){
            session.wanted.insert(sha1);
        }
        return detail::resolve_one(config, session, hash, path, error_output, &bytes_output);
    }
    inline bool resolve(const Config& config, const std::vector<std::string>& hashes, std::vector<std::string>& paths_output, std::string& error_output){
        paths_output.clear();
        detail::Session session;
        for(const std::string& hash: hashes){
            std::string sha1, ignored;
            if(detail::normalize_reference(hash, sha1, ignored)){
                session.wanted.insert(sha1);
            }
        }
        for(const std::string& hash: hashes){
            std::string path;
            if(!detail::resolve_one(config, session, hash, path, error_output)){
                paths_output.clear();
                return false;
            }
            paths_output.push_back(path);
        }
        return true;
    }
    inline Config config_from_environment(){
        Config config;
        const char* compressed_cache = std::getenv("CONTA_COMPRESSED_CACHE");
        config.compressed_cache = compressed_cache == nullptr || std::string(compressed_cache) != "0";
        const char* root = std::getenv("CONTA_ROOT");
        if(root != nullptr && root[0] != '\0'){
            config.root = root;
        }
        const char* cache = std::getenv("CONTA_CACHE");
        if(cache != nullptr && cache[0] != '\0'){
            config.cache = cache;
        }
        else{
#if defined(_WIN32)
            const char* local_app_data = std::getenv("LOCALAPPDATA");
            if(local_app_data != nullptr && local_app_data[0] != '\0'){
                config.cache = (std::filesystem::path(local_app_data) / "conta").string();
            }
#else
            const char* xdg_cache_home = std::getenv("XDG_CACHE_HOME");
            const char* home = std::getenv("HOME");
            if(xdg_cache_home != nullptr && xdg_cache_home[0] != '\0'){
                config.cache = (std::filesystem::path(xdg_cache_home) / "conta").string();
            }
            else if(home != nullptr && home[0] != '\0'){
                config.cache = (std::filesystem::path(home) / ".cache" / "conta").string();
            }
#endif
        }
        const char* url_base = std::getenv("CONTA_URL");
        if(url_base != nullptr && url_base[0] != '\0'){
            config.url_base = url_base;
            if(config.url_base.back() != '/'){
                config.url_base += '/';
            }
        }
        return config;
    }
    inline bool resolve(const std::string& hash, std::string& path_output, std::string& error_output){
        return resolve(config_from_environment(), hash, path_output, error_output);
    }
    inline bool load(const std::string& hash, std::vector<std::uint8_t>& bytes_output, std::string& error_output){
        return load(config_from_environment(), hash, bytes_output, error_output);
    }
    inline bool resolve(const std::vector<std::string>& hashes, std::vector<std::string>& paths_output, std::string& error_output){
        return resolve(config_from_environment(), hashes, paths_output, error_output);
    }
}

#endif
