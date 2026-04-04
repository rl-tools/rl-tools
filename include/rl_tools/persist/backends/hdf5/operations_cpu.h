#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_HDF5_OPERATIONS_CPU)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_HDF5_OPERATIONS_CPU

#include "hdf5.h"
#include <vector>
#include <string>
#include <cstring>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{

    namespace persist::backends::hdf5::detail{
        inline bool read_string_attribute(hid_t loc_id, const char* attr_name, char* output, size_t output_size){
            hid_t attr = H5Aopen(loc_id, attr_name, H5P_DEFAULT);
            if(attr < 0) return false;
            hid_t atype = H5Aget_type(attr);
            hid_t memtype = H5Tcopy(atype);
            if(H5Tis_variable_str(atype) > 0){
                char* str = nullptr;
                H5Aread(attr, memtype, &str);
                if(str){
                    size_t len = 0;
                    while(str[len] != '\0') len++;
                    size_t copy_len = len < output_size - 1 ? len : output_size - 1;
                    for(size_t i = 0; i < copy_len; i++) output[i] = str[i];
                    output[copy_len] = '\0';
                    hid_t space = H5Aget_space(attr);
                    H5Dvlen_reclaim(memtype, space, H5P_DEFAULT, &str);
                    H5Sclose(space);
                }
                else{
                    output[0] = '\0';
                }
            }
            else{
                size_t size = H5Tget_size(atype);
                if(size >= output_size) size = output_size - 1;
                H5Aread(attr, memtype, output);
                output[size] = '\0';
                while(size > 0 && output[size-1] == ' '){ output[--size] = '\0'; }
            }
            H5Tclose(memtype);
            H5Tclose(atype);
            H5Aclose(attr);
            return true;
        }
        inline void write_string_attribute(hid_t loc_id, const char* name, const char* value){
            hid_t space = H5Screate(H5S_SCALAR);
            hid_t atype = H5Tcopy(H5T_C_S1);
            H5Tset_size(atype, H5T_VARIABLE);
            hid_t attr = H5Acreate2(loc_id, name, atype, space, H5P_DEFAULT, H5P_DEFAULT);
            const char* val_ptr = value;
            H5Awrite(attr, atype, &val_ptr);
            H5Aclose(attr);
            H5Tclose(atype);
            H5Sclose(space);
        }

        template<typename T> inline hid_t native_type();
        template<> inline hid_t native_type<float>(){ return H5T_NATIVE_FLOAT; }
        template<> inline hid_t native_type<double>(){ return H5T_NATIVE_DOUBLE; }
        template<> inline hid_t native_type<uint8_t>(){ return H5T_NATIVE_UINT8; }
        template<> inline hid_t native_type<int8_t>(){ return H5T_NATIVE_INT8; }
        template<> inline hid_t native_type<int32_t>(){ return H5T_NATIVE_INT32; }
        template<> inline hid_t native_type<int64_t>(){ return H5T_NATIVE_INT64; }
        template<> inline hid_t native_type<unsigned long>(){ return H5T_NATIVE_ULONG; }
        template<> inline hid_t native_type<unsigned long long>(){ return H5T_NATIVE_ULLONG; }
        template<> inline hid_t native_type<int16_t>(){ return H5T_NATIVE_INT16; }
        template<> inline hid_t native_type<uint16_t>(){ return H5T_NATIVE_UINT16; }
        template<> inline hid_t native_type<uint32_t>(){ return H5T_NATIVE_UINT32; }
        template<> inline hid_t native_type<bool>(){ return H5T_NATIVE_HBOOL; }


        template<typename SHAPE, int DIM = 0>
        inline void fill_dims(hsize_t* dims){
            if constexpr(DIM < SHAPE::LENGTH){
                dims[DIM] = SHAPE::template GET<DIM>;
                fill_dims<SHAPE, DIM + 1>(dims);
            }
        }

        template<typename SHAPE, int DIM = 0>
        inline bool check_dims(const hsize_t* dims){
            if constexpr(DIM < SHAPE::LENGTH){
                return dims[DIM] == (hsize_t)SHAPE::template GET<DIM> && check_dims<SHAPE, DIM + 1>(dims);
            }
            else{ return true; }
        }

        template<typename SHAPE, int DIM = 0>
        inline void write_dim_attrs(hid_t ds_id){
            if constexpr(DIM < SHAPE::LENGTH){
                char key[] = "dim_0";
                key[4] = '0' + DIM;
                std::string val = std::to_string(SHAPE::template GET<DIM>);
                write_string_attribute(ds_id, key, val.c_str());
                write_dim_attrs<SHAPE, DIM + 1>(ds_id);
            }
        }
    }

    template<typename DEVICE, typename SPEC>
    persist::backends::hdf5::Group<SPEC> get_group(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, const char* name){
        return {H5Gopen2(group.id, name, H5P_DEFAULT)};
    }
    template<typename DEVICE, typename SPEC>
    bool group_exists(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, const char* name){
        return H5Lexists(group.id, name, H5P_DEFAULT) > 0;
    }
    template<typename DEVICE, typename SPEC>
    persist::backends::hdf5::Group<SPEC> create_group(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, const char* name){
        return {H5Gcreate2(group.id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
    }

    template<typename DEVICE>
    persist::backends::hdf5::Group<> get_group(DEVICE& device, persist::backends::hdf5::File& file, const char* name){
        return {H5Gopen2(file.id, name, H5P_DEFAULT)};
    }
    template<typename DEVICE>
    persist::backends::hdf5::Group<> get_group(DEVICE& device, persist::backends::hdf5::File& file, std::string name){
        return get_group(device, file, name.c_str());
    }
    template<typename DEVICE>
    persist::backends::hdf5::Group<> create_group(DEVICE& device, persist::backends::hdf5::File& file, const char* name){
        return {H5Gcreate2(file.id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
    }
    template<typename DEVICE>
    persist::backends::hdf5::Group<> create_group(DEVICE& device, persist::backends::hdf5::File& file, std::string name){
        return create_group(device, file, name.c_str());
    }


    template<typename DEVICE, typename SPEC>
    persist::backends::hdf5::Group<SPEC> get_group(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name){
        return get_group(device, group, name.c_str());
    }
    template<typename DEVICE, typename SPEC>
    bool group_exists(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name){
        return group_exists(device, group, name.c_str());
    }
    template<typename DEVICE, typename SPEC>
    persist::backends::hdf5::Group<SPEC> create_group(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name){
        return create_group(device, group, name.c_str());
    }

    template<typename TYPE, typename DEVICE, typename SPEC>
    void get_attribute(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, const char* name, char* output, typename DEVICE::index_t output_size){
        persist::backends::hdf5::detail::read_string_attribute(group.id, name, output, output_size);
    }
    template<typename DEVICE, typename SPEC>
    std::string get_attribute(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name){
        char buf[256];
        persist::backends::hdf5::detail::read_string_attribute(group.id, name.c_str(), buf, sizeof(buf));
        return std::string(buf);
    }

    template<typename TYPE, typename DEVICE, typename SPEC>
    TYPE get_attribute_int(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, const char* name){
        char buf[64];
        persist::backends::hdf5::detail::read_string_attribute(group.id, name, buf, sizeof(buf));
        TYPE result = 0;
        bool neg = false;
        int i = 0;
        if(buf[0] == '-'){ neg = true; i = 1; }
        for(; buf[i] >= '0' && buf[i] <= '9'; i++) result = result * 10 + (buf[i] - '0');
        return neg ? -result : result;
    }
    template<typename TYPE, typename DEVICE, typename SPEC>
    TYPE get_attribute_int(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, std::string name){
        return get_attribute_int<TYPE>(device, group, name.c_str());
    }

    template<typename DEVICE, typename SPEC>
    void set_attribute(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group, const char* name, const char* value){
        persist::backends::hdf5::detail::write_string_attribute(group.id, name, value);
    }
    template<typename DEVICE, typename SPEC>
    void write_attributes(DEVICE& device, persist::backends::hdf5::Group<SPEC>& group){}

    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    bool load(DEVICE& device, Tensor<SPEC>& tensor, persist::backends::hdf5::Group<GROUP_SPEC>& group, std::string dataset_name, bool fallback_to_zero = false){
        return load(device, tensor, group, dataset_name.c_str(), fallback_to_zero);
    }
    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    bool load(DEVICE& device, Tensor<SPEC>& tensor, persist::backends::hdf5::Group<GROUP_SPEC>& group, const char* dataset_name, bool fallback_to_zero = false){
        using T = typename SPEC::T;
        if(fallback_to_zero && H5Lexists(group.id, dataset_name, H5P_DEFAULT) <= 0){
            std::memset(data(tensor), 0, SPEC::SIZE_BYTES);
            return true;
        }
        hid_t ds = H5Dopen2(group.id, dataset_name, H5P_DEFAULT);
        if(ds < 0) return false;
        hid_t space = H5Dget_space(ds);
        int rank = H5Sget_simple_extent_ndims(space);
        if((int)SPEC::SHAPE::LENGTH != rank){
            H5Sclose(space); H5Dclose(ds); return false;
        }
        hsize_t dims[SPEC::SHAPE::LENGTH];
        H5Sget_simple_extent_dims(space, dims, nullptr);
        if(!persist::backends::hdf5::detail::check_dims<typename SPEC::SHAPE>(dims)){
            H5Sclose(space); H5Dclose(ds); return false;
        }
        hid_t memtype = persist::backends::hdf5::detail::native_type<T>();
        herr_t err = H5Dread(ds, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data(tensor));
        H5Sclose(space);
        H5Dclose(ds);
        return err >= 0;
    }

    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void save(DEVICE& device, Tensor<SPEC>& tensor, persist::backends::hdf5::Group<GROUP_SPEC>& group, std::string dataset_name){
        save(device, tensor, group, dataset_name.c_str());
    }
    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void save(DEVICE& device, Tensor<SPEC>& tensor, persist::backends::hdf5::Group<GROUP_SPEC>& group, const char* dataset_name){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr TI RANK = SPEC::SHAPE::LENGTH;
        hsize_t dims[RANK];
        persist::backends::hdf5::detail::fill_dims<typename SPEC::SHAPE>(dims);
        hid_t space = H5Screate_simple(RANK, dims, nullptr);
        hid_t memtype = persist::backends::hdf5::detail::native_type<T>();
        hid_t ds = H5Dcreate2(group.id, dataset_name, memtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if constexpr(!tensor::dense_row_major_layout<SPEC>()){
            Tensor<tensor::Specification<T, TI, typename SPEC::SHAPE>> tensor_dense;
            malloc(device, tensor_dense);
            copy(device, device, tensor, tensor_dense);
            H5Dwrite(ds, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data(tensor_dense));
            free(device, tensor_dense);
        }
        else{
            H5Dwrite(ds, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data(tensor));
        }

        persist::backends::hdf5::detail::write_string_attribute(ds, "type", "tensor");
        std::string num_dims = std::to_string(RANK);
        persist::backends::hdf5::detail::write_string_attribute(ds, "num_dims", num_dims.c_str());
        persist::backends::hdf5::detail::write_dim_attrs<typename SPEC::SHAPE>(ds);
        H5Dclose(ds);
        H5Sclose(space);
    }

    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    bool load(DEVICE& device, Matrix<SPEC>& matrix, persist::backends::hdf5::Group<GROUP_SPEC>& group, std::string dataset_name, bool fallback_to_zero = false){
        auto tensor = to_tensor(device, matrix);
        return load(device, tensor, group, dataset_name, fallback_to_zero);
    }
    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    bool load(DEVICE& device, Matrix<SPEC>& matrix, persist::backends::hdf5::Group<GROUP_SPEC>& group, const char* dataset_name, bool fallback_to_zero = false){
        auto tensor = to_tensor(device, matrix);
        return load(device, tensor, group, dataset_name, fallback_to_zero);
    }
    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void save(DEVICE& device, Matrix<SPEC>& matrix, persist::backends::hdf5::Group<GROUP_SPEC>& group, std::string dataset_name){
        auto tensor = to_tensor(device, matrix);
        save(device, tensor, group, dataset_name);
    }
    template<typename DEVICE, typename SPEC, typename GROUP_SPEC>
    void save(DEVICE& device, Matrix<SPEC>& matrix, persist::backends::hdf5::Group<GROUP_SPEC>& group, const char* dataset_name){
        auto tensor = to_tensor(device, matrix);
        save(device, tensor, group, dataset_name);
    }

    template <typename DEVICE, typename STRUCT, typename GROUP_SPEC>
    void save_binary(DEVICE& device, const STRUCT* structs, typename DEVICE::index_t count, persist::backends::hdf5::Group<GROUP_SPEC>& group, std::string name){
        save_binary(device, structs, count, group, name.c_str());
    }
    template <typename DEVICE, typename STRUCT, typename GROUP_SPEC>
    void save_binary(DEVICE& device, const STRUCT* structs, typename DEVICE::index_t count, persist::backends::hdf5::Group<GROUP_SPEC>& group, const char* name){
        using TI = typename DEVICE::index_t;
        constexpr TI STRUCT_SIZE = sizeof(STRUCT);
        hsize_t dims[] = {(hsize_t)(STRUCT_SIZE * count)};
        hid_t space = H5Screate_simple(1, dims, nullptr);
        hid_t ds = H5Dcreate2(group.id, name, H5T_NATIVE_UINT8, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, reinterpret_cast<const uint8_t*>(structs));
        persist::backends::hdf5::detail::write_string_attribute(ds, "type", "binary");
        std::string size_str = std::to_string(STRUCT_SIZE * count);
        persist::backends::hdf5::detail::write_string_attribute(ds, "size", size_str.c_str());
        H5Dclose(ds);
        H5Sclose(space);
    }

    template <typename DEVICE, typename STRUCT, typename GROUP_SPEC>
    bool load_binary(DEVICE& device, STRUCT* structs, typename DEVICE::index_t count, persist::backends::hdf5::Group<GROUP_SPEC>& group, std::string name){
        return load_binary(device, structs, count, group, name.c_str());
    }
    template <typename DEVICE, typename STRUCT, typename GROUP_SPEC>
    bool load_binary(DEVICE& device, STRUCT* structs, typename DEVICE::index_t count, persist::backends::hdf5::Group<GROUP_SPEC>& group, const char* name){
        using TI = typename DEVICE::index_t;
        constexpr TI STRUCT_SIZE = sizeof(STRUCT);
        hid_t ds = H5Dopen2(group.id, name, H5P_DEFAULT);
        if(ds < 0) return false;
        hid_t space = H5Dget_space(ds);
        hsize_t n;
        H5Sget_simple_extent_dims(space, &n, nullptr);
        if(n != (hsize_t)(STRUCT_SIZE * count)){
            H5Sclose(space); H5Dclose(ds); return false;
        }
        H5Dread(ds, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, reinterpret_cast<uint8_t*>(structs));
        H5Sclose(space);
        H5Dclose(ds);
        return true;
    }

    namespace persist::backends::hdf5{
        template<typename T, typename GROUP_SPEC>
        void read_dataset(persist::backends::hdf5::Group<GROUP_SPEC>& group, const char* name, std::vector<T>& output){
            hid_t ds = H5Dopen2(group.id, name, H5P_DEFAULT);
            hid_t space = H5Dget_space(ds);
            hsize_t n;
            H5Sget_simple_extent_dims(space, &n, nullptr);
            output.resize(n);
            H5Dread(ds, detail::native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, output.data());
            H5Sclose(space);
            H5Dclose(ds);
        }
        template<typename T, typename GROUP_SPEC>
        void read_dataset(persist::backends::hdf5::Group<GROUP_SPEC>& group, const char* name, std::vector<std::vector<T>>& output){
            hid_t ds = H5Dopen2(group.id, name, H5P_DEFAULT);
            hid_t space = H5Dget_space(ds);
            hsize_t dims[2];
            H5Sget_simple_extent_dims(space, dims, nullptr);
            std::vector<T> flat(dims[0] * dims[1]);
            H5Dread(ds, detail::native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, flat.data());
            output.resize(dims[0]);
            for(hsize_t i = 0; i < dims[0]; i++){
                output[i].resize(dims[1]);
                std::memcpy(output[i].data(), flat.data() + i * dims[1], dims[1] * sizeof(T));
            }
            H5Sclose(space);
            H5Dclose(ds);
        }
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
