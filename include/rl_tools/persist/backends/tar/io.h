#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_TAR_IO_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_TAR_IO_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::persist::backends::tar {
    struct Writer{
        std::vector<char> buffer;
    };
    enum class Mode { READ, WRITE };
    template <typename T_TI>
    struct File{
        using TI = T_TI;
        Mode mode;
        std::string path;
        Writer writer;                // used in WRITE mode
        std::vector<char> read_buffer; // used in READ mode
        File(const char* p, Mode m): mode(m), path(p){
            if(mode == Mode::READ){
                std::ifstream ifs(path, std::ios::binary | std::ios::ate);
                if(!ifs.is_open()) throw std::runtime_error(std::string("Failed to open TAR file: ") + path);
                auto size = ifs.tellg();
                ifs.seekg(0, std::ios::beg);
                read_buffer.resize(size);
                ifs.read(read_buffer.data(), size);
            }
        }
        File(const std::string& p, Mode m): File(p.c_str(), m){}
        ~File(){
            if(mode == Mode::WRITE && !writer.buffer.empty()){
                // Note: finalize() must be called before destruction.
                // The destructor flushes the buffer to disk.
                std::ofstream ofs(path, std::ios::binary);
                if(ofs.is_open()){
                    ofs.write(writer.buffer.data(), writer.buffer.size());
                }
            }
        }
        File(File&&) = default;
        File(const File&) = delete;
        File& operator=(const File&) = delete;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif




