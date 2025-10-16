#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_GENERIC)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_GENERIC

#include "tar.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace persist::backends::tar{
        template <typename TI>
        TI calculate_checksum(const header& header) {
            const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&header);
            // Sum all bytes, treating the checksum field as spaces
            TI sum = 0;
            // Sum bytes before checksum field (0 to 147)
            for (TI i = 0; i < 148; i++) {
                sum += bytes[i];
            }
            // Add 8 spaces for the checksum field (148 to 155)
            sum += (' ' * 8);
            // Sum bytes after checksum field (156 to BLOCK_SIZE)
            for (TI i = 156; i < BLOCK_SIZE<TI>; i++) {
                sum += bytes[i];
            }
            return sum;
        }
        template <typename TI, TI MAX_LEN = 200>
        TI strlen(const char* str) {
            TI len = 0;
            while (str[len] != '\0' && len < MAX_LEN) {
                len++;
            }
            return len;
        }
        template <typename TI>
        void strncpy(char* dest, const char* src, TI n) {
            for (TI i = 0; i < n; i++){
                dest[i] = src[i];
                if (src[i] == '\0') return;
            }
            dest[n - 1] = '\0';
        }
        template <typename TI>
        bool strcmp(const char* a, const char* b, TI n) {
            TI i = 0;
            while (a[i] != '\0' && b[i] != '\0' && i < n) {
                if (a[i] != b[i]) return false;
                i++;
            }
            if (i < n && (a[i] == '\0' || b[i] == '\0')){
                return a[i] == b[i];
            }
            return true;
        }
        template <typename T, typename TI>
        void format_octal(char* dest, TI dest_size, T value) {
            // Convert value to octal and write to dest with zero-padding
            // dest_size includes the null terminator
            TI pos = dest_size - 2; // Start from the last position before null terminator
            dest[dest_size - 1] = '\0';
            
            if (value == 0) {
                for (TI i = 0; i < dest_size - 1; i++) {
                    dest[i] = '0';
                }
                return;
            }
            
            // Fill with zeros first
            for (TI i = 0; i < dest_size - 1; i++) {
                dest[i] = '0';
            }
            
            // Convert to octal from right to left
            while (value > 0 && pos >= 0) {
                dest[pos] = '0' + (value & 7); // value % 8
                value >>= 3; // value / 8
                pos--;
            }
        }
        template <typename TI>
        void memcpy(char* dest, const char* src, TI n) {
            for (TI i = 0; i < n; i++) {
                dest[i] = src[i];
            }
        }
        template <typename TI>
        TI parse_octal(const char* str, TI max_len) {
            // Parse octal string to integer
            // Skips leading spaces, stops at first non-octal digit or null terminator
            TI result = 0;
            TI i = 0;
            
            // Skip leading spaces
            while (i < max_len && str[i] == ' ') {
                i++;
            }
            
            // Parse octal digits (0-7)
            while (i < max_len && str[i] != '\0') {
                char c = str[i];
                if (c >= '0' && c <= '7') {
                    result = (result << 3) | (c - '0'); // result * 8 + digit
                } else {
                    // Stop at first non-octal character
                    break;
                }
                i++;
            }
            
            return result;
        }
        template <typename DEVICE, typename WRITER, typename TI>
        void write_entry(DEVICE& device, WRITER& writer, const char* entry_name, const char* data, TI data_size) {
            constexpr TI MAX_LEN = 100;
            utils::assert_exit(device, strlen<TI, MAX_LEN+1>(entry_name) < MAX_LEN, "persist::backends::tar: Entry name is too long");

            header header{};

            strncpy(header.name, entry_name, 99);
            format_octal<unsigned int, TI>(header.mode, sizeof(header.mode), 0644); // Octal permissions
            format_octal<unsigned int, TI>(header.uid, sizeof(header.uid), 1000);
            format_octal<unsigned int, TI>(header.gid, sizeof(header.gid), 1000);
            format_octal<unsigned long long, TI>(header.size, sizeof(header.size), (unsigned long long)data_size);
            format_octal<unsigned long, TI>(header.mtime, sizeof(header.mtime), 0); // Using 0 for timestamp (epoch)
            header.typeflag = '0'; // Regular file
            memcpy<TI>(header.magic, "ustar", 5);
            memcpy<TI>(header.version, "00", 2);
            strncpy(header.uname, "user", 31);
            strncpy(header.gname, "group", 31);

            unsigned int chksum = calculate_checksum<TI>(header);
            format_octal<unsigned int, TI>(header.chksum, sizeof(header.chksum), chksum);

            write(device, writer, reinterpret_cast<const char*>(&header), BLOCK_SIZE<TI>);
            write(device, writer, data, data_size);

            size_t padding_size = (BLOCK_SIZE<TI> - (data_size % BLOCK_SIZE<TI>)) % BLOCK_SIZE<TI>;
            if (padding_size > 0) {
                const char padding[1] = {0};
                for (TI i = 0; i < padding_size; i++){
                    write(device, writer, padding, 1);
                }
            }
        }

        template <typename DEVICE, typename WRITER>
        void finalize(DEVICE& device, WRITER& writer) {
            // Write two empty blocks to signify the end of the archive
            using TI = typename DEVICE::index_t;
            const char padding[1] = {0};
            for (TI i = 0; i < BLOCK_SIZE<TI>*2; i++){
                write(device, writer, padding, 1);
            }
        }

        template <typename DEVICE>
        bool get(DEVICE& device, const char* tar_data, typename DEVICE::index_t length, const char* entry_name, char* output_data, typename DEVICE::index_t output_size) {
            using TI = typename DEVICE::index_t;
            char* ptr = const_cast<char*>(tar_data);
            while (ptr <= tar_data + length - BLOCK_SIZE<TI>) {
                header* h = reinterpret_cast<header*>(ptr);
                ptr += BLOCK_SIZE<TI>;

                // An all-zero block marks the end of the archive
                if (h->name[0] == '\0') {
                    break;
                }

                utils::assert_exit(device, strcmp(h->magic, "ustar", 5), "Warning: Not a UStar format archive or header is corrupted.");

                TI size = parse_octal<TI>(h->size, 12);
                if (strcmp(h->name, entry_name, 100)){
                    utils::assert_exit(device, size <= output_size, "persist::backends::tar: Output buffer is too small for the requested entry");
                    memcpy<TI>(output_data, ptr, size);
                    return true;
                }
                ptr += size;

                size_t padding_size = (BLOCK_SIZE<TI> - (size % BLOCK_SIZE<TI>)) % BLOCK_SIZE<TI>;
                if (padding_size > 0) {
                    ptr += padding_size;
                }
            }
            return false;
        }
    }


}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

