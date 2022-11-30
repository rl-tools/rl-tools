#include "replay_buffer.h"


template <typename T>
struct DefaultOffPolicyRunnerParameters{
    static constexpr uint32_t CAPACITY = 1000000;
    static constexpr uint32_t STEP_LIMIT = 100;
    static constexpr T EXPLORATION_NOISE = 0.1;
};

template <typename T, typename ENVIRONMENT, typename PARAMETERS>
struct OffPolicyRunner {
    ReplayBuffer<T, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, PARAMETERS::CAPACITY> replay_buffer;
    T state[ENVIRONMENT::STATE_DIM];
    uint32_t episode_step = 0;
    T episode_return = 0;
};


template <typename T, typename ENVIRONMENT, typename POLICY, typename PARAMETERS, typename RNG>
void step(OffPolicyRunner<T, ENVIRONMENT, PARAMETERS>& runner, POLICY& policy, RNG& rng){
    // if the episode is done (step limit activated for STEP_LIMIT > 0) or if the step is the first step for this runner, reset the environment
    if((PARAMETERS::STEP_LIMIT > 0 && runner.episode_step == PARAMETERS::STEP_LIMIT) || (runner.replay_buffer.position == 0 && !runner.replay_buffer.full)){
        // first step
        ENVIRONMENT::sample_initial_state(runner.state, rng);
        runner.episode_step = 0;
        runner.episode_return = 0;
    }
    // todo: increase efficiency by removing the double observation of each state
    T observation[ENVIRONMENT::OBSERVATION_DIM];
    ENVIRONMENT::observe(runner.state, observation);
    T next_state[ENVIRONMENT::STATE_DIM];
    T action[ENVIRONMENT::ACTION_DIM];
    evaluate(policy, observation, action);
    std::normal_distribution<T> exploration_noise_distribution(0, PARAMETERS::EXPLORATION_NOISE);
    for(int i = 0; i < ENVIRONMENT::ACTION_DIM; i++){
        action[i] += exploration_noise_distribution(rng);
        action[i] = std::clamp<T>(action[i], -1, 1);
    }
    T reward = ENVIRONMENT::step(runner.state, action, next_state);
    memcpy(runner.state, next_state, sizeof(T) * ENVIRONMENT::STATE_DIM);
    T next_observation[ENVIRONMENT::OBSERVATION_DIM];
    ENVIRONMENT::observe(next_state, next_observation);
    bool terminated = false;
    runner.episode_step += 1;
    runner.episode_return += reward;
    bool truncated = runner.episode_step == PARAMETERS::STEP_LIMIT;
    if(truncated || terminated){
        std::cout << "Episode return: " << runner.episode_return << std::endl;
    }
    // todo: add truncation / termination handling (stemming from the environment)
    add(runner.replay_buffer, observation, action, reward, next_observation, terminated, truncated);
}
