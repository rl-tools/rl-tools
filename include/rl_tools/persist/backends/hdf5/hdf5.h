#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_HDF5_HDF5)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_HDF5_HDF5

#include "../../../rl_tools.h"
#include <hdf5.h>
#include <mutex>
#include <string>
#include <stdexcept>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::persist::backends::hdf5{
    inline std::mutex& global_mutex(){
        static std::mutex mutex;
        return mutex;
    }
    template <typename T=void>
    struct GroupSpecification{};
    template <typename SPEC = GroupSpecification<>>
    struct Group{
        hid_t id;
        Group(hid_t id) : id(id){}
        ~Group(){ if(id >= 0) H5Gclose(id); }
        Group(Group&& o) noexcept : id(o.id){ o.id = -1; }
        Group& operator=(Group&& o) noexcept { if(this != &o){ if(id >= 0) H5Gclose(id); id = o.id; o.id = -1; } return *this; }
        Group(const Group&) = delete;
        Group& operator=(const Group&) = delete;
    };
    enum class Mode { READ, WRITE };
    struct File{
        hid_t id;
        File(const char* path, Mode mode){
            if(mode == Mode::READ){
                id = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
            }
            else{
                id = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            }
            if(id < 0){
                throw std::runtime_error(std::string("Failed to open HDF5 file: ") + path);
            }
        }
        File(const std::string& path, Mode mode): File(path.c_str(), mode){}
        ~File(){ if(id >= 0) H5Fclose(id); }
        File(File&& o) noexcept : id(o.id){ o.id = -1; }
        File& operator=(File&& o) noexcept { if(this != &o){ if(id >= 0) H5Fclose(id); id = o.id; o.id = -1; } return *this; }
        File(const File&) = delete;
        File& operator=(const File&) = delete;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
