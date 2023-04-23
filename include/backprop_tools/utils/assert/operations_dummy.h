#ifndef BACKPROP_TOOLS_UTILS_ASSERT_OPERATIONS_DUMMY_H
#define BACKPROP_TOOLS_UTILS_ASSERT_OPERATIONS_DUMMY_H

namespace backprop_tools::utils{
    template <typename DEV_SPEC, typename T>
    void assert_exit(const devices::Dummy<DEV_SPEC>& dev, bool condition, T message){
        if(!condition){
            logging::text(dev, dev.logger, message);
        }
    }
}

#endif
