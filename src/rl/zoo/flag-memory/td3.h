#include "environment.h"

#include <rl_tools/rl/algorithms/td3/loop/core/config.h>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::flag_memory::td3{
    namespace rlt = rl_tools;
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct FACTORY{
        using ENVIRONMENT = typename ENVIRONMENT_FACTORY<DEVICE, T, TI>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::td3::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            struct TD3_PARAMETERS: rl::algorithms::td3::DefaultParameters<T, TI>{};
            static constexpr TI STEP_LIMIT = 2000000;
            static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr TI ACTOR_HIDDEN_DIM = 128;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr TI CRITIC_HIDDEN_DIM = 128;
            static constexpr T EXPLORATION_NOISE = 0.1;
            static constexpr TI N_WARMUP_STEPS = 20000;

        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::td3::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
