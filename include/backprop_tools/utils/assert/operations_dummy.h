#include "../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDEGUARDS) || !defined(BACKPROP_TOOLS_UTILS_ASSERT_OPERATIONS_DUMMY_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_UTILS_ASSERT_OPERATIONS_DUMMY_H

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::utils{
    template <typename DEV_SPEC, typename T>
    void assert_exit(const devices::Dummy<DEV_SPEC>& dev, bool condition, T message){
        if(!condition){
            logging::text(dev, dev.logger, message);
        }
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif
