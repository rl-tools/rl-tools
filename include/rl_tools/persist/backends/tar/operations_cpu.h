#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_GENERIC)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_GENERIC

#include "hdf5.h"
#include <highfive/H5File.hpp>
#include "../../../containers/matrix/persist_hdf5.h"
#include "../../../containers/tensor/persist_hdf5.h"

#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <map>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <ctime>
#include <array>
#include <cstring> // For memcpy and memset

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace persist::backends::tar{
        unsigned int calculate_checksum(const tar_header& header) {
            const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&header);
            // Sum all bytes, treating the checksum field as spaces
            return std::accumulate(bytes, bytes + 148, 0u) + (' ' * 8) + std::accumulate(bytes + 156, bytes + TAR_BLOCK_SIZE, 0u);
        }
        bool write_entry(std::ostream& archive, std::string_view entry_name, const std::vector<char>& data) {
            if (entry_name.length() >= 100) {
                std::cerr << "Error: Entry name is too long." << std::endl;
                return false;
            }

            detail::tar_header header{}; // Zero-initialize

            // Populate header fields
            strncpy(header.name, entry_name.data(), 99);
            snprintf(header.mode, sizeof(header.mode), "%07o", 0644); // Octal permissions
            snprintf(header.uid, sizeof(header.uid), "%07o", 1000);
            snprintf(header.gid, sizeof(header.gid), "%07o", 1000);
            snprintf(header.size, sizeof(header.size), "%011llo", (long long)data.size());
            snprintf(header.mtime, sizeof(header.mtime), "%011lo", time(nullptr));
            header.typeflag = '0'; // Regular file
            memcpy(header.magic, "ustar", 5);
            memcpy(header.version, "00", 2);
            strncpy(header.uname, "user", 31);
            strncpy(header.gname, "group", 31);

            // Calculate checksum and write it back
            unsigned int chksum = detail::calculate_checksum(header);
            snprintf(header.chksum, sizeof(header.chksum), "%07o", chksum);

            // Write header to archive
            archive.write(reinterpret_cast<const char*>(&header), TAR_BLOCK_SIZE);
            if (!archive) return false;

            // Write file content
            archive.write(data.data(), data.size());
            if (!archive) return false;

            // Pad to the next block boundary
            size_t padding_size = (TAR_BLOCK_SIZE - (data.size() % TAR_BLOCK_SIZE)) % TAR_BLOCK_SIZE;
            if (padding_size > 0) {
                std::array<char, TAR_BLOCK_SIZE> padding_buffer{};
                archive.write(padding_buffer.data(), padding_size);
            }

            return static_cast<bool>(archive);
        }

        void finalize(std::ostream& archive) {
            // Write two empty blocks to signify the end of the archive
            const std::array<char, TAR_BLOCK_SIZE> zero_block{};
            archive.write(zero_block.data(), TAR_BLOCK_SIZE);
            archive.write(zero_block.data(), TAR_BLOCK_SIZE);
        }

        std::map<std::string, std::vector<char>> read_archive(const std::filesystem::path& archive_path) {
            std::map<std::string, std::vector<char>> entries;
            std::ifstream archive(archive_path, std::ios::binary);
            if (!archive) {
                std::cerr << "Failed to open archive: " << archive_path << std::endl;
                return entries;
            }

            std::array<char, TAR_BLOCK_SIZE> buffer;
            while (archive.read(buffer.data(), TAR_BLOCK_SIZE)) {
                detail::tar_header* header = reinterpret_cast<detail::tar_header*>(buffer.data());

                // An all-zero block marks the end of the archive
                if (header->name[0] == '\0') {
                    break;
                }

                if (std::string(header->magic, 5) != "ustar") {
                     std::cerr << "Warning: Not a UStar format archive or header is corrupted." << std::endl;
                     break;
                }

                std::string name(header->name);
                long size = 0;
                try {
                    size = std::stol(std::string(header->size, 12), nullptr, 8);
                } catch (const std::invalid_argument& e) {
                    std::cerr << "Error parsing size for entry: " << name << std::endl;
                    break;
                }

                std::vector<char> data(size);
                archive.read(data.data(), size);

                entries[name] = std::move(data);

                // Seek past padding to the next header
                size_t padding_size = (TAR_BLOCK_SIZE - (size % TAR_BLOCK_SIZE)) % TAR_BLOCK_SIZE;
                if (padding_size > 0) {
                    archive.seekg(padding_size, std::ios_base::cur);
                }

                if (!archive) {
                    std::cerr << "Error reading archive mid-stream." << std::endl;
                    break;
                }
            }
            return entries;
        }
    }


}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

