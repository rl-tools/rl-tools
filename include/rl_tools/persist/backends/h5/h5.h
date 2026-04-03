#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_PERSIST_BACKENDS_H5_H5)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_PERSIST_BACKENDS_H5_H5

#include <hdf5.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::persist::backends::h5{
    template <typename T=void>
    struct GroupSpecification{};
    template <typename SPEC = GroupSpecification<>>
    struct Group{
        hid_t id;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
