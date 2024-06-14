#include <rl_tools/rl/environments/acrobot/operations_generic.h>

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>

namespace rlt = rl_tools;

namespace rl_tools::rl::zoo::sac{
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct AcrobotSwingupV0{
        using ENVIRONMENT_SPEC = rlt::rl::environments::acrobot::Specification<T, TI, rlt::rl::environments::acrobot::DefaultParameters<T>>;
        using ENVIRONMENT = rlt::rl::environments::AcrobotSwingup<ENVIRONMENT_SPEC>;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            struct SAC_PARAMETERS: rl::algorithms::sac::DefaultParameters<T, TI, ENVIRONMENT::ACTION_DIM>{
                static constexpr TI ACTOR_BATCH_SIZE = 128;
                static constexpr TI CRITIC_BATCH_SIZE = 128;
            };
            static constexpr TI STEP_LIMIT = 20000;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr TI ACTOR_HIDDEN_DIM = 256;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr TI CRITIC_HIDDEN_DIM = 256;
            static constexpr T ALPHA = 1.0;
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsSequential>;
    };
}
