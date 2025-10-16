#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_GENERIC)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_GENERIC

#include "tar.h"


/*
import tarfile;
import numpy as np;
f = tarfile.open("test_persist_backends_tar_dense_layer.tar", "r");
meta = dict([l.split(": ") for l in f.extractfile("weights/parameters/meta").read().decode("utf-8").split("\n")][:-1]);
data = np.frombuffer(f.extractfile("weights/parameters/data").read(), dtype=np.float32 if meta["dtype"] == "float32" else np.float64);
data = data.reshape([int(meta[f"dim_{i}"]) for i in range(int(meta["num_dims"]))]) if meta["type"] == "tensor" else data.reshape((int(meta["rows"]), int(meta["cols"])));
print(data)
 */

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
        template <typename T, typename TI>
        TI int_to_string(char* dest, TI dest_size, T value) {
            // Convert integer to decimal string
            // Returns the number of characters written (excluding null terminator)
            if (dest_size == 0) {
                return 0;
            }
            
            TI pos = 0;
            bool is_negative = false;
            
            // Handle negative numbers
            if (value < 0) {
                is_negative = true;
                value = -value;
            }
            
            // Handle zero specially
            if (value == 0) {
                if (dest_size > 1) {
                    dest[0] = '0';
                    dest[1] = '\0';
                    return 1;
                } else {
                    dest[0] = '\0';
                    return 0;
                }
            }
            
            // Convert digits in reverse order
            char temp[32]; // Enough for any 64-bit integer
            TI temp_pos = 0;
            
            while (value > 0 && temp_pos < 32) {
                temp[temp_pos++] = '0' + (value % 10);
                value /= 10;
            }
            
            // Add negative sign if needed
            if (is_negative && pos < dest_size - 1) {
                dest[pos++] = '-';
            }
            
            // Copy digits in correct order
            for (TI i = temp_pos; i > 0 && pos < dest_size - 1; i--) {
                dest[pos++] = temp[i - 1];
            }
            
            dest[pos] = '\0';
            return pos;
        }
        template <typename TI>
        TI string_to_int(const char* str, TI max_len) {
            // Parse decimal string to integer
            // Skips leading spaces, handles negative numbers
            TI result = 0;
            TI i = 0;
            bool is_negative = false;
            
            // Skip leading spaces
            while (i < max_len && str[i] == ' ') {
                i++;
            }
            
            // Check for negative sign
            if (i < max_len && str[i] == '-') {
                is_negative = true;
                i++;
            } else if (i < max_len && str[i] == '+') {
                i++;
            }
            
            // Parse decimal digits (0-9)
            while (i < max_len && str[i] != '\0') {
                char c = str[i];
                if (c >= '0' && c <= '9') {
                    result = result * 10 + (c - '0');
                } else {
                    // Stop at first non-decimal character
                    break;
                }
                i++;
            }
            
            return is_negative ? -result : result;
        }
        template <typename T>
        T min(T a, T b) {
            return a < b ? a : b;
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
        bool get(DEVICE& device, const char* tar_data, typename DEVICE::index_t length, const char* entry_name, char*& output_data, typename DEVICE::index_t& read_size) {
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
                    output_data = ptr;
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
        template <typename DEVICE>
        bool get(DEVICE& device, const char* tar_data, typename DEVICE::index_t length, const char* entry_name, char* output_data, typename DEVICE::index_t output_size, typename DEVICE::index_t& read_size){
            using TI = typename DEVICE::index_t;
            char* ptr;
            if(!get(device, tar_data, length, entry_name, ptr, read_size)){
                return false;
            }
            utils::assert_exit(device, read_size <= output_size, "persist::backends::tar: Output buffer is too small for the requested entry");
            memcpy<TI>(output_data, ptr, read_size);
            return true;
        }
        namespace containers::tensor{
            template <typename SPEC, typename TI = typename SPEC::TI, TI METADATA_SIZE, TI DIM = 0>
            void dim_helper(char* metadata, TI& metadata_position){
                if constexpr(DIM < SPEC::SHAPE::LENGTH){
                    char dim_key[64];
                    char dim_num[16];
                    char dim_value[16];
                    TI pos = 0;
                    pos += persist::backends::tar::strncpy(dim_key, "dim_", 64);
                    pos += persist::backends::tar::int_to_string<TI, TI>(dim_key + pos, 64 - pos, DIM);
                    pos += persist::backends::tar::strncpy(dim_key + pos, ": ", 64 - pos);
                    metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, dim_key, METADATA_SIZE - metadata_position);
                    persist::backends::tar::int_to_string<TI, TI>(dim_value, 16, SPEC::SHAPE::template GET<DIM>);
                    metadata_position += persist::backends::tar::strncpy<TI>(metadata + metadata_position, dim_value, METADATA_SIZE - metadata_position);
                    metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, "\n", METADATA_SIZE - metadata_position);
                    dim_helper<SPEC, TI, METADATA_SIZE, DIM + 1>(metadata, metadata_position);
                }
            }
            template <typename DEVICE, typename SPEC, typename TI = typename SPEC::TI, TI METADATA_SIZE, TI DIM = 0>
            void dim_helper_read(DEVICE& device, char* metadata){
                static_assert(SPEC::SHAPE::LENGTH <= 9, "Only tensors with up to 9 dimensions are supported for now");
                char key[] = "dim_0";
                key[4] = '0' + DIM;
                if constexpr(DIM < SPEC::SHAPE::LENGTH){
                    TI type_position;
                    TI type_value_length;
                    utils::assert_exit(device, persist::backends::tar::seek_in_metadata(metadata, METADATA_SIZE, key, type_position, type_value_length), "persist::backends::tar: 'type' not found in metadata");
                    TI value = persist::backends::tar::string_to_int<TI>(metadata + type_position, type_value_length);
                    utils::assert_exit(device, value == SPEC::SHAPE::template GET<DIM>, "persist::backends::tar: Dimension mismatch in metadata");
                    dim_helper_read<DEVICE, SPEC, TI, METADATA_SIZE, DIM + 1>(device, metadata);
                }
            }

        }
        namespace containers::matrix{
            template <typename SPEC, typename TI = typename SPEC::TI, TI METADATA_SIZE>
            void write_metadata(char* metadata, TI& metadata_position){
                metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, "rows: ", METADATA_SIZE - metadata_position);
                char rows_str[16];
                persist::backends::tar::int_to_string<TI, TI>(rows_str, 16, SPEC::ROWS);
                metadata_position += persist::backends::tar::strncpy<TI>(metadata + metadata_position, rows_str, METADATA_SIZE - metadata_position);
                metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, "\n", METADATA_SIZE - metadata_position);

                metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, "cols: ", METADATA_SIZE - metadata_position);
                char cols_str[16];
                persist::backends::tar::int_to_string<TI, TI>(cols_str, 16, SPEC::COLS);
                metadata_position += persist::backends::tar::strncpy<TI>(metadata + metadata_position, cols_str, METADATA_SIZE - metadata_position);
                metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, "\n", METADATA_SIZE - metadata_position);
            }
            template <typename DEVICE, typename SPEC, typename TI = typename SPEC::TI, TI METADATA_SIZE>
            void read_metadata(DEVICE& device, char* metadata){
                TI rows_position;
                TI rows_value_length;
                utils::assert_exit(device, persist::backends::tar::seek_in_metadata(metadata, METADATA_SIZE, "rows", rows_position, rows_value_length), "persist::backends::tar: 'rows' not found in metadata");
                TI rows_value = persist::backends::tar::string_to_int<TI>(metadata + rows_position, rows_value_length);
                utils::assert_exit(device, rows_value == SPEC::ROWS, "persist::backends::tar: Rows mismatch in metadata");

                TI cols_position;
                TI cols_value_length;
                utils::assert_exit(device, persist::backends::tar::seek_in_metadata(metadata, METADATA_SIZE, "cols", cols_position, cols_value_length), "persist::backends::tar: 'cols' not found in metadata");
                TI cols_value = persist::backends::tar::string_to_int<TI>(metadata + cols_position, cols_value_length);
                utils::assert_exit(device, cols_value == SPEC::COLS, "persist::backends::tar: Cols mismatch in metadata");
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
        char current_path[GROUP_SPEC::MAX_PATH_LENGTH];
        persist::backends::tar::strncpy<TI>(current_path, group_path, GROUP_SPEC::MAX_PATH_LENGTH);
        TI current_path_length = persist::backends::tar::strlen<TI, GROUP_SPEC::MAX_PATH_LENGTH+1>(current_path);
        TI meta_current_position = current_path_length;
        if (current_path_length > 0){
            current_path[current_path_length] = '/';
            meta_current_position += 1;
        }
        utils::assert_exit(device, current_path_length + 1 + sizeof("meta") - 1 < GROUP_SPEC::MAX_PATH_LENGTH, "persist::backends::tar: Meta path and name exceed maximum length");
        persist::backends::tar::strncpy<TI>(current_path + meta_current_position, "meta", GROUP_SPEC::MAX_PATH_LENGTH - current_path_length - 1);
        constexpr TI METADATA_SIZE = 100;
        char metadata[METADATA_SIZE];
        TI read_size = 0;
        utils::assert_exit(device, persist::backends::tar::get(device, group.data, group.size, current_path, metadata, METADATA_SIZE, read_size), "persist::backends::tar: Failed to read metadata entry from tar archive");
        metadata[read_size] = '\0';
        TI type_position = 0;
        TI type_value_length = 0;
        utils::assert_exit(device, persist::backends::tar::seek_in_metadata(metadata, METADATA_SIZE, "type", type_position, type_value_length), "persist::backends::tar: 'type' not found in metadata");
        utils::assert_exit(device, persist::backends::tar::strcmp<TI>(metadata + type_position, "tensor", sizeof("tensor")-1), "persist::backends::tar: 'type' is not 'tensor' in metadata");

        // constexpr TI MAX_VALUE_LENGTH = 20;
        // char value[MAX_VALUE_LENGTH];
        // persist::backends::tar::strncpy<TI>(value, metadata + type_position, type_value_length + 1 < MAX_VALUE_LENGTH ? type_value_length + 1 : MAX_VALUE_LENGTH);
        // std::cout << "type: " << value << std::endl;
        persist::backends::tar::containers::tensor::dim_helper_read<DEVICE, SPEC, TI, METADATA_SIZE>(device, metadata);

        utils::assert_exit(device, current_path_length + 1 + sizeof("data") - 1 < GROUP_SPEC::MAX_PATH_LENGTH, "persist::backends::tar: Meta path and name exceed maximum length");
        persist::backends::tar::strncpy<TI>(current_path + meta_current_position, "data", GROUP_SPEC::MAX_PATH_LENGTH - current_path_length - 1);
        using DENSE_SPEC = tensor::Specification<typename SPEC::T, typename SPEC::TI, typename SPEC::SHAPE>;
        Tensor<DENSE_SPEC> tensor_dense;
        TI data_size;
        utils::assert_exit(device, persist::backends::tar::get(device, group.data, group.size, current_path, (char*&)tensor_dense._data, data_size), "persist::backends::tar: 'data' not found in metadata");
        utils::assert_exit(device, data_size == DENSE_SPEC::SIZE_BYTES, "persist::backends::tar: Data size mismatch");
        copy(device, device, tensor_dense, tensor);
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
        char current_path[GROUP_SPEC::MAX_PATH_LENGTH];
        persist::backends::tar::strncpy<TI>(current_path, group_path, GROUP_SPEC::MAX_PATH_LENGTH);
        TI current_path_length = persist::backends::tar::strlen<TI, GROUP_SPEC::MAX_PATH_LENGTH+1>(current_path);
        TI meta_current_position = current_path_length;
        if (current_path_length > 0){
            current_path[current_path_length] = '/';
            meta_current_position += 1;
        }
        constexpr TI METADATA_SIZE = 100;
        char metadata[METADATA_SIZE];
        TI metadata_position = 0;
        metadata_position += persist::backends::tar::strncpy<TI>(metadata, "type: tensor\n", METADATA_SIZE - metadata_position);
        static_assert(utils::typing::is_same_v<typename SPEC::T, float> || utils::typing::is_same_v<typename SPEC::T, double>, "Only float32 and float64 are supported for now");
        if constexpr(utils::typing::is_same_v<typename SPEC::T, float>){
            metadata_position += persist::backends::tar::strncpy<TI>(metadata+metadata_position, "dtype: float32\n", METADATA_SIZE - metadata_position);
        }
        else if constexpr(utils::typing::is_same_v<typename SPEC::T, double>){
            metadata_position += persist::backends::tar::strncpy<TI>(metadata+metadata_position, "dtype: float64\n", METADATA_SIZE - metadata_position);
        }
        metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, "num_dims: ", METADATA_SIZE - metadata_position);
        char num_dims_str[16];
        persist::backends::tar::int_to_string<TI, TI>(num_dims_str, 16, SPEC::SHAPE::LENGTH);
        metadata_position += persist::backends::tar::strncpy<TI>(metadata + metadata_position, num_dims_str, METADATA_SIZE - metadata_position);
        metadata_position += persist::backends::tar::strncpy(metadata + metadata_position, "\n", METADATA_SIZE - metadata_position);
        persist::backends::tar::containers::tensor::dim_helper<SPEC, TI, METADATA_SIZE>(metadata, metadata_position);
        std::cout << "metadata: \n" << metadata << std::endl;
        utils::assert_exit(device, current_path_length + 1 + sizeof("meta") - 1 < GROUP_SPEC::MAX_PATH_LENGTH, "persist::backends::tar: Meta path and name exceed maximum length");
        persist::backends::tar::strncpy<TI>(current_path + meta_current_position, "meta", GROUP_SPEC::MAX_PATH_LENGTH - current_path_length - 1);
        write_entry(device, group.writer, current_path, metadata, metadata_position);
        Tensor<tensor::Specification<typename SPEC::T, typename SPEC::TI, typename SPEC::SHAPE>> tensor_dense;
        malloc(device, tensor_dense);
        copy(device, device, tensor, tensor_dense);
        utils::assert_exit(device, current_path_length + 1 + sizeof("data") - 1 < GROUP_SPEC::MAX_PATH_LENGTH, "persist::backends::tar: Meta path and name exceed maximum length");
        persist::backends::tar::strncpy<TI>(current_path + meta_current_position, "data", GROUP_SPEC::MAX_PATH_LENGTH - current_path_length - 1);
        write_entry(device, group.writer, current_path, reinterpret_cast<const char*>(data(tensor_dense)), SPEC::SIZE_BYTES);
        free(device, tensor_dense);
    }

    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void load(DEVICE& device, Matrix<SPEC>& matrix, persist::backends::tar::ReaderGroup<GROUP_SPEC>& group, const char* name) {
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
        std::cout << "Reading matrix from tar entry: " << group_path << std::endl;
        char current_path[GROUP_SPEC::MAX_PATH_LENGTH];
        persist::backends::tar::strncpy<TI>(current_path, group_path, GROUP_SPEC::MAX_PATH_LENGTH);
        TI current_path_length = persist::backends::tar::strlen<TI, GROUP_SPEC::MAX_PATH_LENGTH+1>(current_path);
        TI meta_current_position = current_path_length;
        if (current_path_length > 0){
            current_path[current_path_length] = '/';
            meta_current_position += 1;
        }
        utils::assert_exit(device, current_path_length + 1 + sizeof("meta") - 1 < GROUP_SPEC::MAX_PATH_LENGTH, "persist::backends::tar: Meta path and name exceed maximum length");
        persist::backends::tar::strncpy<TI>(current_path + meta_current_position, "meta", GROUP_SPEC::MAX_PATH_LENGTH - current_path_length - 1);
        std::cout << "meta path: " << current_path << std::endl;
        constexpr TI METADATA_SIZE = 100;
        char metadata[METADATA_SIZE];
        TI read_size = 0;
        utils::assert_exit(device, persist::backends::tar::get(device, group.data, group.size, current_path, metadata, METADATA_SIZE, read_size), "persist::backends::tar: Failed to read metadata entry from tar archive");
        metadata[read_size] = '\0';
        TI type_position = 0;
        TI type_value_length = 0;
        utils::assert_exit(device, persist::backends::tar::seek_in_metadata(metadata, METADATA_SIZE, "type", type_position, type_value_length), "persist::backends::tar: 'type' not found in metadata");
        utils::assert_exit(device, persist::backends::tar::strcmp<TI>(metadata + type_position, "matrix", sizeof("matrix")-1), "persist::backends::tar: 'type' is not 'matrix' in metadata");

        persist::backends::tar::containers::matrix::read_metadata<DEVICE, SPEC, TI, METADATA_SIZE>(device, metadata);

        utils::assert_exit(device, current_path_length + 1 + sizeof("data") - 1 < GROUP_SPEC::MAX_PATH_LENGTH, "persist::backends::tar: Meta path and name exceed maximum length");
        persist::backends::tar::strncpy<TI>(current_path + meta_current_position, "data", GROUP_SPEC::MAX_PATH_LENGTH - current_path_length - 1);
        using DENSE_SPEC = matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::ROWS, SPEC::COLS>;
        Matrix<DENSE_SPEC> matrix_dense;
        TI data_size;
        utils::assert_exit(device, persist::backends::tar::get(device, group.data, group.size, current_path, (char*&)matrix_dense._data, data_size), "persist::backends::tar: 'data' not found in metadata");
        utils::assert_exit(device, data_size == DENSE_SPEC::SIZE_BYTES, "persist::backends::tar: Data size mismatch");
        copy(device, device, matrix_dense, matrix);
    }

    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void save(DEVICE& device, Matrix<SPEC>& matrix, persist::backends::tar::WriterGroup<GROUP_SPEC>& group, const char* name) {
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
        std::cout << "Saving matrix to tar entry: " << group_path << std::endl;
        constexpr TI METADATA_SIZE = 100;
        char metadata[METADATA_SIZE];
        TI metadata_position = 0;
        metadata_position += persist::backends::tar::strncpy<TI>(metadata, "type: matrix\n", METADATA_SIZE - metadata_position);
        static_assert(utils::typing::is_same_v<typename SPEC::T, float> || utils::typing::is_same_v<typename SPEC::T, double>, "Only float32 and float64 are supported for now");
        if constexpr(utils::typing::is_same_v<typename SPEC::T, float>){
            metadata_position += persist::backends::tar::strncpy<TI>(metadata+metadata_position, "dtype: float32\n", METADATA_SIZE - metadata_position);
        }
        else if constexpr(utils::typing::is_same_v<typename SPEC::T, double>){
            metadata_position += persist::backends::tar::strncpy<TI>(metadata+metadata_position, "dtype: float64\n", METADATA_SIZE - metadata_position);
        }
        persist::backends::tar::containers::matrix::write_metadata<SPEC, TI, METADATA_SIZE>(metadata, metadata_position);
        std::cout << "metadata: \n" << metadata << std::endl;

        char current_path[GROUP_SPEC::MAX_PATH_LENGTH];
        persist::backends::tar::strncpy<TI>(current_path, group_path, GROUP_SPEC::MAX_PATH_LENGTH);
        TI current_path_length = persist::backends::tar::strlen<TI, GROUP_SPEC::MAX_PATH_LENGTH+1>(current_path);
        TI meta_current_position = current_path_length;
        if (current_path_length > 0){
            current_path[current_path_length] = '/';
            meta_current_position += 1;
        }
        utils::assert_exit(device, current_path_length + 1 + sizeof("meta") - 1 < GROUP_SPEC::MAX_PATH_LENGTH, "persist::backends::tar: Meta path and name exceed maximum length");
        persist::backends::tar::strncpy<TI>(current_path + meta_current_position, "meta", GROUP_SPEC::MAX_PATH_LENGTH - current_path_length - 1);
        std::cout << "meta path: " << current_path << std::endl;
        write_entry(device, group.writer, current_path, metadata, metadata_position);
        Matrix<matrix::Specification<typename SPEC::T, typename SPEC::TI, SPEC::ROWS, SPEC::COLS>> matrix_dense;
        malloc(device, matrix_dense);
        copy(device, device, matrix, matrix_dense);
        utils::assert_exit(device, current_path_length + 1 + sizeof("data") - 1 < GROUP_SPEC::MAX_PATH_LENGTH, "persist::backends::tar: Data path and name exceed maximum length");
        persist::backends::tar::strncpy<TI>(current_path + meta_current_position, "data", GROUP_SPEC::MAX_PATH_LENGTH - current_path_length - 1);
        std::cout << "data path: " << current_path << std::endl;
        write_entry(device, group.writer, current_path, reinterpret_cast<const char*>(matrix_dense._data), SPEC::SIZE_BYTES);
        free(device, matrix_dense);
    }
    namespace persist::backends::tar{
        template<typename DEVICE, typename GROUP>
        GROUP create_group(DEVICE& device, GROUP& group, const char* name) {
            using TI = typename DEVICE::index_t;
            using GROUP_SPEC = typename GROUP::SPEC;
            GROUP new_group = group;
            persist::backends::tar::strncpy<TI>(new_group.path, group.path, GROUP_SPEC::MAX_PATH_LENGTH);
            TI group_path_length = persist::backends::tar::strlen<TI, GROUP_SPEC::MAX_PATH_LENGTH+1>(new_group.path);
            TI name_length = persist::backends::tar::strlen<TI, GROUP_SPEC::MAX_PATH_LENGTH+1>(name);
            utils::assert_exit(device, group_path_length + 1 + name_length < GROUP_SPEC::MAX_PATH_LENGTH, "persist::backends::tar: Group path and name exceed maximum length");
            TI current_position = group_path_length;
            if (group_path_length > 0){
                new_group.path[group_path_length] = '/';
                current_position += 1;
            }
            persist::backends::tar::strncpy<TI>(new_group.path + current_position, name, GROUP_SPEC::MAX_PATH_LENGTH - group_path_length - 1);
            return new_group;
        }
    }
    template<typename DEVICE, typename GROUP_SPEC>
    persist::backends::tar::WriterGroup<GROUP_SPEC> create_group(DEVICE& device, persist::backends::tar::WriterGroup<GROUP_SPEC>& group, const char* name){
        return persist::backends::tar::create_group(device, group, name);
    }
    template<typename DEVICE, typename GROUP_SPEC>
    persist::backends::tar::WriterGroup<GROUP_SPEC> get_group(DEVICE& device, persist::backends::tar::WriterGroup<GROUP_SPEC>& group, const char* name){
        return persist::backends::tar::create_group(device, group, name);
    }
    template<typename DEVICE, typename GROUP_SPEC>
    persist::backends::tar::ReaderGroup<GROUP_SPEC> create_group(DEVICE& device, persist::backends::tar::ReaderGroup<GROUP_SPEC>& group, const char* name){
        return persist::backends::tar::create_group(device, group, name);
    }
    template<typename DEVICE, typename GROUP_SPEC>
    persist::backends::tar::ReaderGroup<GROUP_SPEC> get_group(DEVICE& device, persist::backends::tar::ReaderGroup<GROUP_SPEC>& group, const char* name){
        return persist::backends::tar::create_group(device, group, name);
    }
    template<typename TYPE, typename DEVICE, typename SPEC>
    void set_attribute(DEVICE& device, persist::backends::tar::WriterGroup<SPEC>& group, std::string name, TYPE value) {
    }
    // template<typename DEVICE, typename SPEC>
    // persist::backends::tar::WriterGroup<SPEC> get_group(DEVICE& device, persist::backends::tar::WriterGroup<SPEC>& group, std::string name) {
    //
    // }
    // template<typename TYPE, typename DEVICE, typename SPEC>
    // TYPE get_attribute(DEVICE& device, persist::backends::tar::WriterGroup<SPEC>& group, std::string name) {
    //     return group.group.getAttribute(name).template read<TYPE>();
    // }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

