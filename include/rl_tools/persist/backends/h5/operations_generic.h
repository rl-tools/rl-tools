#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_H5_OPERATIONS_GENERIC)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_H5_OPERATIONS_GENERIC

#include "h5.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{

    // --- Group operations ---
    template<typename DEVICE, typename SPEC>
    persist::backends::h5::Group<SPEC> get_group(DEVICE& device, persist::backends::h5::Group<SPEC>& group, const char* name){
        return {H5Gopen2(group.id, name, H5P_DEFAULT)};
    }

    template<typename DEVICE>
    persist::backends::h5::Group<> get_group(DEVICE& device, hid_t file_id, const char* name){
        return {H5Gopen2(file_id, name, H5P_DEFAULT)};
    }

    template<typename DEVICE, typename SPEC>
    bool group_exists(DEVICE& device, persist::backends::h5::Group<SPEC>& group, const char* name){
        htri_t exists = H5Lexists(group.id, name, H5P_DEFAULT);
        return exists > 0;
    }

    // --- Attribute operations ---
    namespace persist::backends::h5::detail{
        inline bool read_string_attribute(hid_t loc_id, const char* attr_name, char* output, size_t output_size){
            hid_t attr = H5Aopen(loc_id, attr_name, H5P_DEFAULT);
            if(attr < 0) return false;
            hid_t atype = H5Aget_type(attr);
            // Build memory type matching the file type's encoding (ASCII or UTF-8)
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
                // Trim trailing spaces (H5T_STR_SPACEPAD)
                while(size > 0 && output[size-1] == ' '){ output[--size] = '\0'; }
            }
            H5Tclose(memtype);
            H5Tclose(atype);
            H5Aclose(attr);
            return true;
        }
    }

    template<typename TYPE, typename DEVICE, typename SPEC>
    void get_attribute(DEVICE& device, persist::backends::h5::Group<SPEC>& group, const char* name, char* output, typename DEVICE::index_t output_size){
        persist::backends::h5::detail::read_string_attribute(group.id, name, output, output_size);
    }

    template<typename TYPE, typename DEVICE, typename SPEC>
    TYPE get_attribute_int(DEVICE& device, persist::backends::h5::Group<SPEC>& group, const char* name){
        char buf[64];
        persist::backends::h5::detail::read_string_attribute(group.id, name, buf, sizeof(buf));
        TYPE result = 0;
        bool neg = false;
        int i = 0;
        if(buf[0] == '-'){ neg = true; i = 1; }
        for(; buf[i] >= '0' && buf[i] <= '9'; i++) result = result * 10 + (buf[i] - '0');
        return neg ? -result : result;
    }

    // --- Write operations (for completeness, not needed for load) ---
    template<typename DEVICE, typename SPEC>
    persist::backends::h5::Group<SPEC> create_group(DEVICE& device, persist::backends::h5::Group<SPEC>& group, const char* name){
        return {H5Gcreate2(group.id, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
    }

    template<typename DEVICE, typename SPEC>
    void set_attribute(DEVICE& device, persist::backends::h5::Group<SPEC>& group, const char* name, const char* value){
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t atype = H5Tcopy(H5T_C_S1);
        H5Tset_size(atype, H5T_VARIABLE);
        hid_t attr = H5Acreate2(group.id, name, atype, space, H5P_DEFAULT, H5P_DEFAULT);
        const char* val_ptr = value;
        H5Awrite(attr, atype, &val_ptr);
        H5Aclose(attr);
        H5Tclose(atype);
        H5Sclose(space);
    }

    template<typename DEVICE, typename SPEC>
    void write_attributes(DEVICE& device, persist::backends::h5::Group<SPEC>& group){}

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
