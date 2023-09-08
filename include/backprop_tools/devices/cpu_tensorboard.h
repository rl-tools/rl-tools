#include "../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDEGUARDS) || !defined(BACKPROP_TOOLS_DEVICES_CPU_TENSORBOARD_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_DEVICES_CPU_TENSORBOARD_H

#include "devices.h"
#include "cpu.h"

#include <tensorboard_logger.h>
#include <mutex>

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::devices{
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
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif