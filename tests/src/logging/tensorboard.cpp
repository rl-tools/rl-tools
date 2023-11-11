#include <rl_tools/operations/cpu_tensorboard.h>
namespace bpt = rl_tools;


#include <gtest/gtest.h>


TEST(BACKPROP_TOOLS_LOGGING_TENSORBOARD, INIT){

    using LOGGER = bpt::devices::logging::CPU_TENSORBOARD<bpt::devices::logging::CPU_TENSORBOARD_FREQUENCY_EXTENSION>;
    using DEV_SPEC = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, LOGGER>;
    using DEVICE = bpt::devices::CPU<DEV_SPEC>;
    using TI = typename DEVICE::index_t;
    using T = float;

    DEVICE device;
    bpt::construct(device, device.logger);

    for(TI i = 0; i < 100; i++){
        bpt::set_step(device, device.logger, i);
        bpt::add_scalar(device, device.logger, "test", (T)(i * i));
    }

    bpt::destruct(device, device.logger);

}
