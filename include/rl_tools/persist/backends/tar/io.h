#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_TAR_IO_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_TAR_IO_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::persist::backends::tar {
    struct Writer{
        std::vector<char> buffer;
    };
    template <typename T_TI, typename T_WRITER>
    struct WriterGroupSpecification{
        using TI = T_TI;
        using WRITER = T_WRITER;
        static constexpr TI MAX_PATH_LENGTH = 100;
    };
    template <typename SPEC>
    struct WriterGroup{
        using TI = typename SPEC::TI;
        using WRITER = typename SPEC::WRITER;
        char path[SPEC::MAX_PATH_LENGTH] = "";
        WRITER writer;
    };
    template <typename T_TI>
    struct ReaderGroupSpecification{
        using TI = T_TI;
        static constexpr TI MAX_PATH_LENGTH = 100;
    };
    template <typename SPEC>
    struct ReaderGroup{
        using TI = typename SPEC::TI;
        char path[SPEC::MAX_PATH_LENGTH] = "";
        const char* data = nullptr;
        TI size = 0;
    };
    struct Reader{

    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif




