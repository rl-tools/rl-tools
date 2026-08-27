#ifndef CONTA_CONTA_H
#define CONTA_CONTA_H
// conta: client for the sha1 content-addressed asset store (https://huggingface.co/datasets/rl-tools/conta)
// Freestanding, C++17, STL-only; downloads via the curl CLI.
// CONTA_ROOT: read-only store (<root>/data/<sha1>), disables downloads. CONTA_CACHE: writable cache directory (flat <sha1>). CONTA_URL: download base URL.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <system_error>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace conta{
    constexpr const char* DEFAULT_URL_BASE = "https://huggingface.co/datasets/rl-tools/conta/resolve/main/data/";
    struct Config{
        std::string root;
        std::string cache;
        std::string url_base;
    };
    namespace detail{
        struct Sha1{
            std::uint32_t state[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
            std::uint64_t message_length_bits = 0;
            unsigned char buffer[64];
            std::size_t buffer_size = 0;
            static std::uint32_t rotate_left(std::uint32_t value, unsigned bits){
                return (value << bits) | (value >> (32u - bits));
            }
            void process_block(const unsigned char* block){
                std::uint32_t schedule[80];
                for(unsigned i = 0; i < 16; i++){
                    schedule[i] = (std::uint32_t(block[4 * i]) << 24u) | (std::uint32_t(block[4 * i + 1]) << 16u) | (std::uint32_t(block[4 * i + 2]) << 8u) | std::uint32_t(block[4 * i + 3]);
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
                    std::uint32_t temp = rotate_left(a, 5) + f + e + k + schedule[i];
                    e = d;
                    d = c;
                    c = rotate_left(b, 30);
                    b = a;
                    a = temp;
                }
                state[0] += a;
                state[1] += b;
                state[2] += c;
                state[3] += d;
                state[4] += e;
            }
            void update(const unsigned char* data, std::size_t size){
                message_length_bits += std::uint64_t(size) * 8u;
                while(size > 0){
                    std::size_t take = std::min(size, std::size_t(64) - buffer_size);
                    std::memcpy(buffer + buffer_size, data, take);
                    buffer_size += take;
                    data += take;
                    size -= take;
                    if(buffer_size == 64){
                        process_block(buffer);
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
                std::string hex;
                hex.reserve(40);
                for(unsigned i = 0; i < 5; i++){
                    for(int shift = 28; shift >= 0; shift -= 4){
                        hex += "0123456789abcdef"[(state[i] >> unsigned(shift)) & 0xFu];
                    }
                }
                return hex;
            }
        };
        inline bool sha1_file(const std::filesystem::path& path, std::string& hex_output){
            std::ifstream file(path, std::ios::binary);
            if(!file.is_open()){
                return false;
            }
            Sha1 sha1;
            std::vector<char> chunk(64 * 1024);
            while(file){
                file.read(chunk.data(), std::streamsize(chunk.size()));
                std::streamsize count = file.gcount();
                if(count > 0){
                    sha1.update(reinterpret_cast<const unsigned char*>(chunk.data()), std::size_t(count));
                }
            }
            if(file.bad()){
                return false;
            }
            hex_output = sha1.finalize();
            return true;
        }
        inline bool normalize_hash(std::string& hash){
            if(hash.size() != 40){
                return false;
            }
            for(char& character: hash){
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
        inline bool shell_quote(const std::string& input, std::string& output){
#if defined(_WIN32)
            if(input.find('"') != std::string::npos || input.find('%') != std::string::npos){
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
        inline bool download(const Config& config, const std::string& hash, const std::filesystem::path& target, std::string& error_output){
            std::filesystem::path temporary = target;
            temporary += ".partial." + std::to_string(process_id());
            std::string url = config.url_base + hash;
            std::string quoted_temporary, quoted_url;
            if(!shell_quote(temporary.string(), quoted_temporary)){
                error_output = "conta: unsupported character in path: " + temporary.string();
                return false;
            }
            if(!shell_quote(url, quoted_url)){
                error_output = "conta: unsupported character in URL: " + url;
                return false;
            }
            std::cerr << "conta: downloading " << hash << " from " << url << std::endl;
            std::string command = "curl -f -L --retry 3 --connect-timeout 15 -o " + quoted_temporary + " " + quoted_url;
            int status = std::system(command.c_str());
            std::error_code error_code;
            if(status != 0){
                std::filesystem::remove(temporary, error_code);
                error_output = "conta: download of " + hash + " failed (curl exit status " + std::to_string(status) + "): " + url;
                if(status == 127 || status == 32512){
                    error_output += ". Is curl installed and on the PATH?";
                }
                else{
                    error_output += ". Check network connectivity.";
                }
                return false;
            }
            std::string actual_hash;
            if(!sha1_file(temporary, actual_hash) || actual_hash != hash){
                std::filesystem::remove(temporary, error_code);
                error_output = "conta: hash mismatch for " + hash + ": downloaded content hashes to " + (actual_hash.empty() ? "<unreadable>" : actual_hash) + "; not caching: " + url;
                return false;
            }
            std::filesystem::rename(temporary, target, error_code);
            if(error_code){
                if(std::filesystem::exists(target)){ // a concurrent download of the same content won the race
                    std::filesystem::remove(temporary, error_code);
                    return true;
                }
                error_output = "conta: failed to move " + temporary.string() + " to " + target.string() + ": " + error_code.message();
                return false;
            }
            return true;
        }
    }
    inline bool resolve(const Config& config, const std::string& hash, std::string& path_output, std::string& error_output){
        std::string normalized_hash = hash;
        if(!detail::normalize_hash(normalized_hash)){
            error_output = "conta: invalid hash \"" + hash + "\": expected 40 hexadecimal characters";
            return false;
        }
        if(!config.root.empty()){
            std::filesystem::path path = std::filesystem::path(config.root) / "data" / normalized_hash;
            if(!std::filesystem::exists(path)){
                error_output = "conta: " + normalized_hash + " not found at " + path.string() + ". CONTA_ROOT points to a read-only store, so downloading is disabled; populate the store or unset CONTA_ROOT to enable the download cache.";
                return false;
            }
            if(detail::is_lfs_pointer(path)){
                error_output = "conta: " + path.string() + " is a git-lfs pointer file, not the blob. Run \"git lfs pull\" in " + config.root + ".";
                return false;
            }
            path_output = path.string();
            return true;
        }
        if(config.cache.empty()){
            error_output = "conta: cannot determine cache directory: set CONTA_CACHE, XDG_CACHE_HOME, or HOME";
            return false;
        }
        std::filesystem::path target = std::filesystem::path(config.cache) / normalized_hash;
        if(std::filesystem::exists(target)){
            path_output = target.string();
            return true;
        }
        std::error_code error_code;
        std::filesystem::create_directories(config.cache, error_code);
        if(error_code){
            error_output = "conta: failed to create cache directory " + config.cache + ": " + error_code.message();
            return false;
        }
        if(!detail::download(config, normalized_hash, target, error_output)){
            return false;
        }
        path_output = target.string();
        return true;
    }
    inline bool resolve(const Config& config, const std::vector<std::string>& hashes, std::vector<std::string>& paths_output, std::string& error_output){
        paths_output.clear();
        for(const std::string& hash: hashes){
            std::string path;
            if(!resolve(config, hash, path, error_output)){
                paths_output.clear();
                return false;
            }
            paths_output.push_back(path);
        }
        return true;
    }
    inline Config config_from_environment(){
        Config config;
        const char* root = std::getenv("CONTA_ROOT");
        if(root != nullptr && root[0] != '\0'){
            config.root = root;
        }
        const char* cache = std::getenv("CONTA_CACHE");
        if(cache != nullptr && cache[0] != '\0'){
            config.cache = cache;
        }
        else{
            const char* xdg_cache_home = std::getenv("XDG_CACHE_HOME");
            const char* home = std::getenv("HOME");
            if(xdg_cache_home != nullptr && xdg_cache_home[0] != '\0'){
                config.cache = (std::filesystem::path(xdg_cache_home) / "rl_tools" / "conta").string();
            }
            else if(home != nullptr && home[0] != '\0'){
                config.cache = (std::filesystem::path(home) / ".cache" / "rl_tools" / "conta").string();
            }
#if defined(_WIN32)
            else{
                const char* local_app_data = std::getenv("LOCALAPPDATA");
                if(local_app_data != nullptr && local_app_data[0] != '\0'){
                    config.cache = (std::filesystem::path(local_app_data) / "rl_tools" / "conta").string();
                }
            }
#endif
        }
        const char* url_base = std::getenv("CONTA_URL");
        config.url_base = (url_base != nullptr && url_base[0] != '\0') ? url_base : DEFAULT_URL_BASE;
        if(config.url_base.back() != '/'){
            config.url_base += '/';
        }
        return config;
    }
    inline bool resolve(const std::string& hash, std::string& path_output, std::string& error_output){
        return resolve(config_from_environment(), hash, path_output, error_output);
    }
    inline bool resolve(const std::vector<std::string>& hashes, std::vector<std::string>& paths_output, std::string& error_output){
        return resolve(config_from_environment(), hashes, paths_output, error_output);
    }
}

#endif
