#ifndef LAYER_IN_C_DEVICES_CPU_TENSORBOARD_H
#define LAYER_IN_C_DEVICES_CPU_TENSORBOARD_H

#include "devices.h"
#include "cpu.h"

#include <tensorboard_logger.h>
#include <mutex>

namespace layer_in_c::devices{
    namespace logging{
        struct CPU_TENSORBOARD: logging::CPU{
            static constexpr DeviceId DEVICE_ID = DeviceId::CPU_TENSORBOARD;
            static constexpr Type TYPE = Type::logging;
            index_t step = 0;
            TensorBoardLogger* tb = nullptr;
            std::mutex mutex;
        };
    }
    using DefaultCPU_TENSORBOARDSpecification = cpu::Specification<math::CPU, random::CPU, logging::CPU_TENSORBOARD>;
    using DefaultCPU_TENSORBOARD = CPU<DefaultCPU_TENSORBOARDSpecification>;
}

#endif