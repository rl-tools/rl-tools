#ifndef LAYER_IN_C_DEVICES_H
#define LAYER_IN_C_DEVICES_H

#include <layer_in_c/utils/generic/typing.h>

namespace layer_in_c {
    namespace devices {
        enum class Device {
            Generic,
            Dummy,
            CPU
        };
        namespace math{
            struct Generic{
                static constexpr Device DEVICE = Device::Generic;
            };
            struct Dummy{
                static constexpr Device DEVICE = Device::Dummy;
            };
            struct CPU{
                static constexpr Device DEVICE = Device::CPU;
            };
            template <typename DEVICE>
            constexpr bool check = utils::typing::is_same_v<DEVICE, Generic> || utils::typing::is_same_v<DEVICE, Dummy> || utils::typing::is_same_v<DEVICE, CPU>;
        }

        namespace random{
            struct Generic{
                static constexpr Device DEVICE = Device::Generic;
            };
            struct Dummy{
                static constexpr Device DEVICE = Device::Dummy;
            };
            struct CPU{
                static constexpr Device DEVICE = Device::CPU;
            };
            template <typename DEVICE>
            constexpr bool check = utils::typing::is_same_v<DEVICE, Generic> || utils::typing::is_same_v<DEVICE, Dummy> || utils::typing::is_same_v<DEVICE, CPU>;
        }
        namespace logging{
            struct Generic{
                static constexpr Device DEVICE = Device::Generic;
            };
            struct Dummy{
                static constexpr Device DEVICE = Device::Dummy;
            };
            struct CPU{
                static constexpr Device DEVICE = Device::CPU;
            };
            struct CPU_WANDB{
                static constexpr Device DEVICE = Device::CPU;
                int counter = 0;
            };
            template <typename DEVICE>
            constexpr bool check =
                utils::typing::is_same_v<DEVICE, Generic> ||
                utils::typing::is_same_v<DEVICE,   Dummy> ||
                utils::typing::is_same_v<DEVICE,     CPU> ||
                utils::typing::is_same_v<DEVICE, CPU_WANDB>;
        }


        template <typename T_SPEC>
        struct Generic{
            using SPEC = T_SPEC;
            typename SPEC::LOGGING& logger;
            explicit Generic(typename SPEC::LOGGING& logger) : logger(logger) {}
        };
        template <typename T_SPEC>
        struct Dummy{
            using SPEC = T_SPEC;
            typename SPEC::LOGGING& logger;
            explicit Dummy(typename SPEC::LOGGING& logger) : logger(logger) {}
        };
        template <typename T_SPEC>
        struct CPU{
            using SPEC = T_SPEC;
            typename SPEC::LOGGING& logger;
            explicit CPU(typename SPEC::LOGGING& logger) : logger(logger) {}
        };


        struct DefaultGenericSpecification{
            using MATH = math::Generic;
            using RANDOM = random::Generic;
            using LOGGING = logging::Generic;
        };
        struct DefaultDummySpecification{
            using MATH = math::Dummy;
            using RANDOM = random::Dummy;
            using LOGGING = logging::Dummy;
        };
        struct DefaultCPUSpecification{
            using MATH = math::CPU;
            using RANDOM = random::CPU;
            using LOGGING = logging::CPU;
        };

        using DefaultGeneric = Generic<DefaultGenericSpecification>;
        using DefaultDummy = Dummy<DefaultDummySpecification>;
        using DefaultCPU = CPU<DefaultCPUSpecification>;

    }
}

#endif