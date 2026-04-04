#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_CPU)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_CPU

#include "../../../rl_tools.h"
#include "io.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE>
    void write(DEVICE& device, persist::backends::tar::Writer& writer, const char* data, typename DEVICE::index_t size) {
        using TI = typename DEVICE::index_t;
        for (TI i = 0; i < size; i++) {
            writer.buffer.push_back(data[i]);
        }
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
#ifndef RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_CPU_NOT_INCLUDE_GENERIC
#include "operations_generic.h"
#endif

#ifdef RL_TOOLS_PERSIST_BACKENDS_TAR_OPERATIONS_GENERIC
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename TI>
    persist::backends::tar::WriterGroup<persist::backends::tar::WriterGroupSpecification<TI, persist::backends::tar::Writer>> create_group(DEVICE& device, persist::backends::tar::File<TI>& file, const char* name){
        using WGS = persist::backends::tar::WriterGroupSpecification<TI, persist::backends::tar::Writer>;
        persist::backends::tar::WriterGroup<WGS> root;
        root.path[0] = '\0';
        root.writer = &file.writer;
        root.meta[0] = '\0';
        root.meta_position = 0;
        return create_group(device, root, name);
    }
    template<typename DEVICE, typename TI>
    persist::backends::tar::WriterGroup<persist::backends::tar::WriterGroupSpecification<TI, persist::backends::tar::Writer>> create_group(DEVICE& device, persist::backends::tar::File<TI>& file, std::string name){
        return create_group(device, file, name.c_str());
    }
    template<typename DEVICE, typename TI>
    persist::backends::tar::ReaderGroup<persist::backends::tar::ReaderGroupSpecification<TI>> get_group(DEVICE& device, persist::backends::tar::File<TI>& file, const char* name){
        using RGS = persist::backends::tar::ReaderGroupSpecification<TI>;
        persist::backends::tar::ReaderGroup<RGS> root;
        root.path[0] = '\0';
        root.data = {file.read_buffer.data(), (TI)file.read_buffer.size()};
        return get_group(device, root, name);
    }
    template<typename DEVICE, typename TI>
    persist::backends::tar::ReaderGroup<persist::backends::tar::ReaderGroupSpecification<TI>> get_group(DEVICE& device, persist::backends::tar::File<TI>& file, std::string name){
        return get_group(device, file, name.c_str());
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif

#endif
