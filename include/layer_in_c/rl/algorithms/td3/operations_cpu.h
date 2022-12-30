
#include <layer_in_c/devices.h>
#include <layer_in_c/utils/generic/math.h>
#include "td3.h"

#include <random>

namespace layer_in_c{
    template <typename SPEC, typename CRITIC_TYPE, int CAPACITY, typename RNG>
    typename SPEC::T train_critic(rl::algorithms::td3::ActorCritic<devices::Generic, SPEC>& actor_critic, CRITIC_TYPE& critic, rl::algorithms::td3::ReplayBuffer<typename SPEC::T, SPEC::ENVIRONMENT::OBSERVATION_DIM, SPEC::ENVIRONMENT::ACTION_DIM, CAPACITY>& replay_buffer, RNG& rng) {
        typedef typename SPEC::T T;
        std::normal_distribution<T> target_next_action_noise_distribution(0, SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_STD);
        T action_noise[SPEC::PARAMETERS::CRITIC_BATCH_SIZE][SPEC::ENVIRONMENT::ACTION_DIM];
        for(int batch_sample_i=0; batch_sample_i < SPEC::PARAMETERS::CRITIC_BATCH_SIZE; batch_sample_i++){
            for(int action_i=0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++){
                action_noise[batch_sample_i][action_i] = lic::utils::math::clamp(
                        target_next_action_noise_distribution(rng),
                        -SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_CLIP,
                        SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_CLIP
                );
            }
        }
        return train_critic(actor_critic, critic, replay_buffer, action_noise, rng);
    }
}
