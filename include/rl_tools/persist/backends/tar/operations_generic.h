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
        TI strncpy(char* dest, const char* src, TI n) {
            for (TI i = 0; i < n; i++){
                dest[i] = src[i];
                if (src[i] == '\0'){
                    return i;
                }
            }
            dest[n - 1] = '\0';
            return n-1;
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
        template <typename TI>
        bool seek_in_metadata(const char* metadata, TI metadata_size, const char* key, TI& position, TI& value_len) {
            TI key_len = strlen<TI, 100>(key);
            bool previous_was_newline = true;
            for (position = 0; position < metadata_size; position++){
                if (previous_was_newline && strcmp<TI>(metadata + position, key, key_len) && metadata[position + key_len] == ':'){
                    position += key_len + 2;
                    value_len = 0;
                    while (position + value_len < metadata_size && metadata[position+value_len] != '\n'){
                        value_len++;
                    }
                    return true;
                }
                previous_was_newline = (metadata[position] == '\n');
            }
            return -1; // Key not found
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
        bool get(DEVICE& device, const char* tar_data, typename DEVICE::index_t length, const char* entry_name, char* output_data, typename DEVICE::index_t output_size, typename DEVICE::index_t& read_size) {
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

                read_size = parse_octal<TI>(h->size, 12);
                if (strcmp(h->name, entry_name, 100)){
                    utils::assert_exit(device, read_size <= output_size, "persist::backends::tar: Output buffer is too small for the requested entry");
                    memcpy<TI>(output_data, ptr, read_size);
                    return true;
                }
                ptr += read_size;

                size_t padding_size = (BLOCK_SIZE<TI> - (read_size % BLOCK_SIZE<TI>)) % BLOCK_SIZE<TI>;
                if (padding_size > 0) {
                    ptr += padding_size;
                }
            }
            return false;
        }
    }

    namespace containers::tensor{
        template <typename SPEC, typename TI = typename SPEC::TI, TI METADATA_SIZE, TI DIM = 0>
        void dim_helper(char* metadata, TI& metadata_position){
            if constexpr(DIM < SPEC::SHAPE::LENGTH){
                std::string dim_key = "dim_" + std::to_string(DIM) + ": ";
                metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, dim_key.c_str(), METADATA_SIZE - metadata_position);
                metadata_position += persist::backends::tar::strncpy<TI>(metadata + metadata_position, std::to_string(SPEC::SHAPE::template GET<DIM>).c_str(), METADATA_SIZE - metadata_position);
                metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, "\n", METADATA_SIZE - metadata_position);
                dim_helper<SPEC, TI, METADATA_SIZE, DIM + 1>(metadata, metadata_position);
            }
        }

    }

    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void load(DEVICE& device, Tensor<SPEC>& tensor, persist::backends::tar::ReaderGroup<GROUP_SPEC>& group, const char* name) {
        using TI = typename DEVICE::index_t;
        char group_path[GROUP_SPEC::MAX_PATH_LENGTH];
        persist::backends::tar::strncpy<TI>(group_path, group.path, GROUP_SPEC::MAX_PATH_LENGTH);
        TI group_path_length = persist::backends::tar::strlen<TI, GROUP_SPEC::MAX_PATH_LENGTH+1>(group_path);
        TI name_length = persist::backends::tar::strlen<TI, GROUP_SPEC::MAX_PATH_LENGTH+1>(name);
        utils::assert_exit(device, group_path_length + 1 + name_length < GROUP_SPEC::MAX_PATH_LENGTH, "persist::backends::tar: Group path and name exceed maximum length");
        TI current_position = group_path_length;
        if (group_path_length > 0){
            group_path[group_path_length] = '/';
            current_position += 1;
        }
        persist::backends::tar::strncpy<TI>(group_path + current_position, name, GROUP_SPEC::MAX_PATH_LENGTH - group_path_length - 1);
        std::cout << "Reading tensor from tar entry: " << group_path << std::endl;
        constexpr TI METADATA_SIZE = 100;
        char metadata[METADATA_SIZE];
        TI read_size = 0;
        utils::assert_exit(device, persist::backends::tar::get(device, group.data, group.size, "meta", metadata, METADATA_SIZE, read_size), "persist::backends::tar: Failed to read metadata entry from tar archive");
        metadata[read_size] = '\0';
        TI type_position = 0;
        TI type_value_length = 0;
        utils::assert_exit(device, rl_tools::persist::backends::tar::seek_in_metadata(metadata, METADATA_SIZE, "type", type_position, type_value_length), "persist::backends::tar: 'type' not found in metadata");
        char value[20];
        persist::backends::tar::strncpy<TI>(value, metadata + type_position, std::min<TI>(type_value_length + 1, 20));
        std::cout << "type: " << value << std::endl;
    }

    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void save(DEVICE& device, Tensor<SPEC>& tensor, persist::backends::tar::WriterGroup<GROUP_SPEC>& group, const char* name) {
        using TI = typename DEVICE::index_t;
        char group_path[GROUP_SPEC::MAX_PATH_LENGTH];
        persist::backends::tar::strncpy<TI>(group_path, group.path, GROUP_SPEC::MAX_PATH_LENGTH);
        TI group_path_length = persist::backends::tar::strlen<TI, GROUP_SPEC::MAX_PATH_LENGTH+1>(group_path);
        TI name_length = persist::backends::tar::strlen<TI, GROUP_SPEC::MAX_PATH_LENGTH+1>(name);
        utils::assert_exit(device, group_path_length + 1 + name_length < GROUP_SPEC::MAX_PATH_LENGTH, "persist::backends::tar: Group path and name exceed maximum length");
        TI current_position = group_path_length;
        if (group_path_length > 0){
            group_path[group_path_length] = '/';
            current_position += 1;
        }
        persist::backends::tar::strncpy<TI>(group_path + current_position, name, GROUP_SPEC::MAX_PATH_LENGTH - group_path_length - 1);
        std::cout << "Saving tensor to tar entry: " << group_path << std::endl;
        constexpr TI METADATA_SIZE = 100;
        char metadata[METADATA_SIZE];
        TI metadata_position = 0;
        metadata_position += persist::backends::tar::strncpy<TI>(metadata, "type: tensor\n", METADATA_SIZE - metadata_position);
        metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, "num_dims: ", METADATA_SIZE - metadata_position);
        metadata_position += persist::backends::tar::strncpy<TI>(metadata + metadata_position, std::to_string(SPEC::SHAPE::LENGTH).c_str(), METADATA_SIZE - metadata_position);
        metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, "\n", METADATA_SIZE - metadata_position);
        containers::tensor::dim_helper<SPEC, TI, METADATA_SIZE>(metadata, metadata_position);
        std::cout << "metadata: \n" << metadata << std::endl;
        write_entry(device, group.writer, "meta", metadata, metadata_position);
        Tensor<tensor::Specification<typename SPEC::T, typename SPEC::TI, typename SPEC::SHAPE>> tensor_dense;
        malloc(device, tensor_dense);
        copy(device, device, tensor, tensor_dense);
        write_entry(device, group.writer, "data", reinterpret_cast<const char*>(data(tensor_dense)), SPEC::SIZE_BYTES);
        free(device, tensor_dense);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

