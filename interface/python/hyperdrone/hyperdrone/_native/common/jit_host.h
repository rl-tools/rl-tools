#pragma once
// Host side of the JIT ABI seam: dlopen a per-config artifact, resolve its
// hyperdrone_<component>_* symbols, validate iface version and canonical config string,
// and manage instance lifetime. Libraries stay loaded for the process lifetime so repeated
// construction is cheap and CUDA/driver teardown ordering stays trivial.
#include <dlfcn.h>
#include <map>
#include <stdexcept>
#include <string>

namespace hyperdrone {
    template <typename IFACE>
    struct JitLibrary {
        IFACE* (*create)();
        void (*destroy)(IFACE*);
        const char* (*config_string)();
        int (*iface_version)();
    };

    template <typename IFACE>
    JitLibrary<IFACE>& get_jit_library(const std::string& path, const std::string& symbol_prefix, int expected_iface_version){
        static std::map<std::string, JitLibrary<IFACE>> libraries;
        auto existing = libraries.find(path);
        if(existing != libraries.end()){
            return existing->second;
        }
        void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if(handle == nullptr){
            throw std::runtime_error("hyperdrone: failed to load JIT library: " + std::string(dlerror()));
        }
        JitLibrary<IFACE> library;
        library.create = (IFACE* (*)())dlsym(handle, (symbol_prefix + "_create").c_str());
        library.destroy = (void (*)(IFACE*))dlsym(handle, (symbol_prefix + "_destroy").c_str());
        library.config_string = (const char* (*)())dlsym(handle, (symbol_prefix + "_config_string").c_str());
        library.iface_version = (int (*)())dlsym(handle, (symbol_prefix + "_iface_version").c_str());
        if(library.create == nullptr || library.destroy == nullptr || library.config_string == nullptr || library.iface_version == nullptr){
            throw std::runtime_error("hyperdrone: JIT library is missing the " + symbol_prefix + "_{create,destroy,config_string,iface_version} symbols: " + path);
        }
        if(library.iface_version() != expected_iface_version){
            throw std::runtime_error("hyperdrone: JIT library was built against iface version "
                + std::to_string(library.iface_version()) + " but the core module expects "
                + std::to_string(expected_iface_version) + " (stale cache): " + path);
        }
        return libraries.emplace(path, library).first->second;
    }

    template <typename IFACE>
    struct JitInstance {
        IFACE* instance = nullptr;
        void (*destroy)(IFACE*) = nullptr;

        JitInstance(const std::string& path, const std::string& symbol_prefix, int expected_iface_version, const std::string& expected_config){
            JitLibrary<IFACE>& library = get_jit_library<IFACE>(path, symbol_prefix, expected_iface_version);
            const std::string actual_config = library.config_string();
            if(actual_config != expected_config){
                throw std::runtime_error("hyperdrone: JIT library config mismatch (expected \"" + expected_config + "\", library reports \"" + actual_config + "\"): " + path);
            }
            destroy = library.destroy;
            instance = library.create();
        }
        ~JitInstance(){
            if(instance != nullptr){
                destroy(instance);
            }
        }
        JitInstance(const JitInstance&) = delete;
        JitInstance& operator=(const JitInstance&) = delete;
    };
}
