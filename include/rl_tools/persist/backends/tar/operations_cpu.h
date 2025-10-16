#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_GENERIC)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_GENERIC

#include "tar.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace persist::backends::tar{
        template <typename DEVICE>
        void write(DEVICE& device, Writer& writer, const char* data, typename DEVICE::index_t size) {
            using TI = typename DEVICE::index_t;
            for (TI i = 0; i < size; i++) {
                writer.buffer.push_back(data[i]);
            }
        }
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

        // std::map<std::string, std::vector<char>> read_archive(const std::filesystem::path& archive_path) {
        //     std::map<std::string, std::vector<char>> entries;
        //     std::ifstream archive(archive_path, std::ios::binary);
        //     if (!archive) {
        //         std::cerr << "Failed to open archive: " << archive_path << std::endl;
        //         return entries;
        //     }
        //
        //     std::array<char, TAR_BLOCK_SIZE> buffer;
        //     while (archive.read(buffer.data(), TAR_BLOCK_SIZE)) {
        //         detail::tar_header* header = reinterpret_cast<detail::tar_header*>(buffer.data());
        //
        //         // An all-zero block marks the end of the archive
        //         if (header->name[0] == '\0') {
        //             break;
        //         }
        //
        //         if (std::string(header->magic, 5) != "ustar") {
        //              std::cerr << "Warning: Not a UStar format archive or header is corrupted." << std::endl;
        //              break;
        //         }
        //
        //         std::string name(header->name);
        //         long size = 0;
        //         try {
        //             size = std::stol(std::string(header->size, 12), nullptr, 8);
        //         } catch (const std::invalid_argument& e) {
        //             std::cerr << "Error parsing size for entry: " << name << std::endl;
        //             break;
        //         }
        //
        //         std::vector<char> data(size);
        //         archive.read(data.data(), size);
        //
        //         entries[name] = std::move(data);
        //
        //         // Seek past padding to the next header
        //         size_t padding_size = (TAR_BLOCK_SIZE - (size % TAR_BLOCK_SIZE)) % TAR_BLOCK_SIZE;
        //         if (padding_size > 0) {
        //             archive.seekg(padding_size, std::ios_base::cur);
        //         }
        //
        //         if (!archive) {
        //             std::cerr << "Error reading archive mid-stream." << std::endl;
        //             break;
        //         }
        //     }
        //     return entries;
        // }
    }


}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

