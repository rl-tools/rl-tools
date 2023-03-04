#include "ppo.h"
#include <layer_in_c/rl/components/on_policy_runner/on_policy_runner.h>

namespace layer_in_c{
    template <typename DEVICE, typename SPEC, auto STEPS_PER_ENV>
    void estimate_generalized_advantages(DEVICE& device, rl::algorithms::PPO<SPEC>, rl::components::on_policy_runner::Buffer<rl::components::on_policy_runner::BufferSpecification<SPEC, STEPS_PER_ENV>& buffer){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            T previous_advantage = 0;
            bool previous_advantage_set = false;
            for(TI step_i = STEPS_PER_ENV-1; step_i >= 0; step_i--){
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;

            }
        }
    }
}