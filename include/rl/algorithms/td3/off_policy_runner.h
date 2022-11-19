#include "replay_buffer.h"

template <typename T, typename ENVIRONMENT, typename POLICY, int CAPACITY, int STEP_LIMIT>
struct OffPolicyRunner {
    ReplayBuffer<T, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, CAPACITY> replay_buffer;
    T state[ENVIRONMENT::STATE_DIM];
    uint32_t episode_step = 0;
    T episode_return = 0;
};


template <typename T, typename ENVIRONMENT, typename POLICY, int CAPACITY, int STEP_LIMIT, typename RNG>
void step(OffPolicyRunner<T, ENVIRONMENT, POLICY, CAPACITY, STEP_LIMIT>& runner, POLICY& policy, RNG& rng){
    if(runner.episode_step == STEP_LIMIT || (runner.replay_buffer.position == 0 && !runner.replay_buffer.full)){
        // first step
        ENVIRONMENT::sample_initial_state(runner.state, rng);
        runner.episode_step = 0;
        runner.episode_return = 0;
    }
    T observation[ENVIRONMENT::OBSERVATION_DIM];
    ENVIRONMENT::observe(runner.state, observation);
    T next_state[ENVIRONMENT::STATE_DIM];
    T action[ENVIRONMENT::ACTION_DIM];
    evaluate(policy, runner.state, action);
    T reward = ENVIRONMENT::step(runner.state, action, next_state);
    memcpy(runner.state, next_state, sizeof(T) * ENVIRONMENT::STATE_DIM);
    T next_observation[ENVIRONMENT::OBSERVATION_DIM];
    ENVIRONMENT::observe(next_state, next_observation);
    bool terminated = false;
    runner.episode_step += 1;
    runner.episode_return += reward;
    bool truncated = runner.episode_step == STEP_LIMIT;
    if(truncated || terminated){
        std::cout << "Episode return: " << runner.episode_return << std::endl;
    }
    add(runner.replay_buffer, observation, action, reward, next_observation, terminated, truncated);
}
