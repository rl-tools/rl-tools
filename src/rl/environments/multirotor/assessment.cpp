#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>
#include <backprop_tools/rl/environments/multirotor/ui.h>
#include <backprop_tools/nn_models/operations_cpu.h>
#include <backprop_tools/nn_models/persist.h>

namespace bpt = backprop_tools;

#ifdef BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
#include "ppo/parameters.h"
#else
#include "td3/parameters.h"
#endif

#include <chrono>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <highfive/H5File.hpp>
#include <CLI/CLI.hpp>
#include <tuple>


namespace variations {
    template <typename TI, typename T_ENVIRONMENT, typename T_RNG>
    struct Specification{
        using ENVIRONMENT = T_ENVIRONMENT;
        using T = typename ENVIRONMENT::T;
        using RNG = T_RNG;
        static constexpr TI MAX_EPISODE_LENGTH = 1000;
    };
    namespace init{
        template <typename DEVICE, typename SPEC>
        void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::simple<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
        template <typename DEVICE, typename SPEC>
        void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::orientation_small_angle<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
        template <typename DEVICE, typename SPEC>
        void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::orientation_big_angle<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
        template <typename DEVICE, typename SPEC>
        void variation_3(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::orientation_bigger_angle<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
        template <typename DEVICE, typename SPEC>
        void variation_4(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::orientation_biggest_angle<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
        template <typename DEVICE, typename SPEC>
        void variation_5(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::orientation_all_around<typename SPEC::ENVIRONMENT::T, typename DEVICE::index_t, 4, typename SPEC::ENVIRONMENT::REWARD_FUNCTION>;
        }
    }
    namespace observation_noise{
        namespace position{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.position = 0.05;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.position = 0.1;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.position = 0.2;
            }
        }
        namespace orientation{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.orientation = 0.05;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.orientation = 0.1;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.orientation = 0.2;
            }
        }
        namespace linear_velocity{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.linear_velocity = 0.05;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.linear_velocity = 0.1;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.linear_velocity = 0.2;
            }
        }
        namespace angular_velocity{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.angular_velocity = 0.1;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.angular_velocity = 0.5;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                env.parameters.mdp.observation_noise.angular_velocity = 1;
            }
        }
    }
    namespace action_noise{
        template <typename DEVICE, typename SPEC>
        void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.action_noise.normalized_rpm = 0.1;
        }
        template <typename DEVICE, typename SPEC>
        void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.action_noise.normalized_rpm = 0.2;
        }
        template <typename DEVICE, typename SPEC>
        void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
            env.parameters.mdp.action_noise.normalized_rpm = 0.5;
        }
    }
    namespace dynamics{
        namespace inertia{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T J_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.8, (T)1.2, rng);
                env.parameters.dynamics.J[0][0] *= J_factor;
                env.parameters.dynamics.J[1][1] *= J_factor;
                env.parameters.dynamics.J[2][2] *= J_factor;
                env.parameters.dynamics.J_inv[0][0] /= J_factor;
                env.parameters.dynamics.J_inv[1][1] /= J_factor;
                env.parameters.dynamics.J_inv[2][2] /= J_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T J_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.5, (T)2, rng);
                env.parameters.dynamics.J[0][0] *= J_factor;
                env.parameters.dynamics.J[1][1] *= J_factor;
                env.parameters.dynamics.J[2][2] *= J_factor;
                env.parameters.dynamics.J_inv[0][0] /= J_factor;
                env.parameters.dynamics.J_inv[1][1] /= J_factor;
                env.parameters.dynamics.J_inv[2][2] /= J_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T J_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.2, (T)5, rng);
                env.parameters.dynamics.J[0][0] *= J_factor;
                env.parameters.dynamics.J[1][1] *= J_factor;
                env.parameters.dynamics.J[2][2] *= J_factor;
                env.parameters.dynamics.J_inv[0][0] /= J_factor;
                env.parameters.dynamics.J_inv[1][1] /= J_factor;
                env.parameters.dynamics.J_inv[2][2] /= J_factor;
            }
        }
        namespace mass{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T mass_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.8, (T)1.2, rng);
                env.parameters.dynamics.mass *= mass_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T mass_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.7, (T)1.5, rng);
                env.parameters.dynamics.mass *= mass_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T mass_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.5, (T)2, rng);
                env.parameters.dynamics.mass *= mass_factor;
            }
        }
        namespace max_rpm{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T max_rpm_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.8, (T)1.2, rng);
                env.parameters.dynamics.action_limit.max *= max_rpm_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T max_rpm_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.7, (T)1.5, rng);
                env.parameters.dynamics.action_limit.max *= max_rpm_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T max_rpm_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.5, (T)2, rng);
                env.parameters.dynamics.action_limit.max *= max_rpm_factor;
            }
        }
        namespace rpm_time_constant{
            template <typename DEVICE, typename SPEC>
            void variation_0(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T rpm_time_constant_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.8, (T)1.2, rng);
                env.parameters.dynamics.rpm_time_constant *= rpm_time_constant_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_1(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T rpm_time_constant_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.7, (T)1.5, rng);
                env.parameters.dynamics.rpm_time_constant *= rpm_time_constant_factor;
            }
            template <typename DEVICE, typename SPEC>
            void variation_2(typename SPEC::ENVIRONMENT& env, typename SPEC::RNG& rng){
                using T = typename SPEC::ENVIRONMENT::T;
                T rpm_time_constant_factor = bpt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0.5, (T)2, rng);
                env.parameters.dynamics.rpm_time_constant *= rpm_time_constant_factor;
            }
        }


    }
}



template <typename DEVICE, typename SPEC, typename ACTOR_TYPE, auto VARIATION, typename RNG>
std::tuple<typename SPEC::T, typename SPEC::T> assess(DEVICE& device, ACTOR_TYPE& actor, typename SPEC::ENVIRONMENT::PARAMETERS nominal_parameters, RNG& rng){
    constexpr bool REALTIME = false;
    using T = typename SPEC::T;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT = typename SPEC::ENVIRONMENT;

    ENVIRONMENT env;
    bpt::rl::environments::multirotor::UI<ENVIRONMENT> ui;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    typename ENVIRONMENT::State state, next_state;

    bpt::malloc(device, env);
    bpt::malloc(device, action);
    bpt::malloc(device, observation);

    ui.host = "localhost";
    ui.port = "8080";
    if(REALTIME){
        bpt::init(device, env, ui);
    }
    T total_rewards = 0;
    T total_steps = 0;
    constexpr TI NUM_EPISODES = 100;
    for(TI episode_i = 0; episode_i < NUM_EPISODES; episode_i++){
        env.parameters = nominal_parameters;
        VARIATION(env, rng);
        T reward_acc = 0;
        bpt::sample_initial_state(device, env, state, rng);
        for(int step_i = 0; step_i < SPEC::MAX_EPISODE_LENGTH; step_i++){
            auto start = std::chrono::high_resolution_clock::now();
            bpt::observe(device, env, state, observation, rng);
            bpt::evaluate(device, actor, observation, action);
//            for(TI action_i = 0; action_i < penv::ENVIRONMENT::ACTION_DIM; action_i++){
//                increment(action, 0, action_i, bpt::random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, (T)(T)prl::OFF_POLICY_RUNNER_PARAMETERS::EXPLORATION_NOISE, rng));
//            }
            bpt::clamp(device, action, (T)-1, (T)1);
            T dt = bpt::step(device, env, state, action, next_state, rng);
            bool terminated_flag = bpt::terminated(device, env, next_state, rng);
            T reward = bpt::reward(device, env, state, action, next_state, rng);
            if(std::isnan(reward)){
                std::cout << "NAN reward" << std::endl;
            }
            reward_acc += reward;
            state = next_state;
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end-start;
            if(REALTIME){
                bpt::set_state(device, ui, state, action);
                std::this_thread::sleep_for(std::chrono::milliseconds((int)((dt - diff.count())*1000)));
            }
            if(terminated_flag || step_i == (SPEC::MAX_EPISODE_LENGTH - 1)){
//                std::cout << "Episode terminated after " << step_i << " steps with reward " << reward_acc << std::endl;
                total_rewards += reward_acc;
                total_steps += step_i + 1;
                break;
            }
        }
    }
    bpt::free(device, action);
    bpt::free(device, observation);
    return {total_rewards / NUM_EPISODES, total_steps / NUM_EPISODES / SPEC::MAX_EPISODE_LENGTH};
}

template <typename DEVICE, typename ENVIRONMENT, typename ACTOR>
void full_assessment(DEVICE& device, ACTOR& actor, typename ENVIRONMENT::PARAMETERS nominal_parameters){
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM(), 10);

    using VARIATION_SPEC = variations::Specification<typename DEVICE::index_t, ENVIRONMENT, decltype(rng)>;
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "init.variation_0 (simple): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "init.variation_1 (orientation small): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "init.variation_2 (orientation big): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_3<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "init.variation_3 (orientation bigger): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_4<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "init.variation_3 (orientation biggest): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::init::variation_5<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "init.variation_4 (orientation all around): mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::position::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.position.variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::position::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.position.variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::position::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.position.variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::orientation::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.orientation.variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::orientation::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.orientation.variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::orientation::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.orientation.variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::linear_velocity::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.linear_velocity.variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::linear_velocity::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.linear_velocity.variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::linear_velocity::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.linear_velocity.variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::angular_velocity::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.angular_velocity.variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::angular_velocity::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.angular_velocity.variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::observation_noise::angular_velocity::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "observation_noise.angular_velocity.variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::action_noise::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "action_noise.variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::action_noise::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "action_noise.variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::action_noise::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "action_noise.variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::inertia::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::inertia::variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::inertia::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::inertia::variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::inertia::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::inertia::variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::mass::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::mass::variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::mass::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::mass::variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::mass::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::mass::variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::max_rpm::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::max_rpm::variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::max_rpm::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::max_rpm::variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::max_rpm::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::max_rpm::variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::rpm_time_constant::variation_0<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::rpm_time_constant::variation_0: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::rpm_time_constant::variation_1<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::rpm_time_constant::variation_1: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
    {
        auto stats = assess<DEVICE, VARIATION_SPEC, ACTOR, variations::dynamics::rpm_time_constant::variation_2<DEVICE, VARIATION_SPEC>, decltype(rng)>(device, actor, nominal_parameters, rng);
        std::cout << "variations::dynamics::rpm_time_constant::variation_2: mean return: " << std::get<0>(stats) << " mean steps: " << std::get<1>(stats) <<  std::endl;
    }
}

namespace TEST_DEFINITIONS{
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;
    namespace parameter_set = parameters_0;

    using penv = parameter_set::environment<T, TI>;
    using ENVIRONMENT = penv::ENVIRONMENT;
    using UI = bpt::rl::environments::multirotor::UI<ENVIRONMENT>;

    using prl = parameter_set::rl<T, TI, penv::ENVIRONMENT>;
    constexpr TI MAX_EPISODE_LENGTH = 1000;
    constexpr bool RANDOMIZE_DOMAIN_PARAMETERS = true;
    constexpr bool INIT_SIMPLE = true;
    constexpr bool DEACTIVATE_OBSERVATION_NOISE = true;
}
template <typename DEVICE, typename ACTOR_TYPE>
void load_actor(DEVICE& device, std::string arg_run, std::string arg_checkpoint, ACTOR_TYPE& actor){

    std::string run = arg_run;
    std::string checkpoint = arg_checkpoint;

    std::filesystem::path actor_run;
    if(run == "" && checkpoint == ""){
#ifdef BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
        std::filesystem::path actor_checkpoints_dir = std::filesystem::path("checkpoints") / "multirotor_ppo";
#else
        std::filesystem::path actor_checkpoints_dir = std::filesystem::path("checkpoints") / "multirotor_td3";
#endif
        std::vector<std::filesystem::path> actor_runs;

        for (const auto& run : std::filesystem::directory_iterator(actor_checkpoints_dir)) {
            if (run.is_directory()) {
                actor_runs.push_back(run.path());
            }
        }
        std::sort(actor_runs.begin(), actor_runs.end());
        actor_run = actor_runs.back();
    }
    else{
        actor_run = run;
    }
    if(checkpoint == ""){
        std::vector<std::filesystem::path> actor_checkpoints;
        for (const auto& checkpoint : std::filesystem::directory_iterator(actor_run)) {
            if (checkpoint.is_regular_file()) {
                if(checkpoint.path().extension() == ".h5" || checkpoint.path().extension() == ".hdf5"){
                    actor_checkpoints.push_back(checkpoint.path());
                }
            }
        }
        std::sort(actor_checkpoints.begin(), actor_checkpoints.end());
        checkpoint = actor_checkpoints.back().string();
    }

    std::cout << "Loading actor from " << checkpoint << std::endl;
    {
        auto data_file = HighFive::File(checkpoint, HighFive::File::ReadOnly);
        bpt::load(device, actor, data_file.getGroup("actor"));
#ifdef BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MUJOCO_ANT_EVALUATE_ACTOR_PPO
        bpt::load(device, observation_normalizer.mean, data_file.getGroup("observation_normalizer"), "mean");
        bpt::load(device, observation_normalizer.std, data_file.getGroup("observation_normalizer"), "std");
#endif
    }
}
int main(int argc, char** argv) {
    using DEVICE = TEST_DEFINITIONS::DEVICE;


    CLI::App app;
    std::string arg_run = "", arg_checkpoint = "";
    app.add_option("--run", arg_run, "path to the run's directory");
    app.add_option("--checkpoint", arg_checkpoint, "path to the checkpoint");

    CLI11_PARSE(app, argc, argv);

    DEVICE device;
    typename TEST_DEFINITIONS::prl::ACTOR_TYPE actor;
    bpt::malloc(device, actor);
    load_actor<DEVICE, typename TEST_DEFINITIONS::prl::ACTOR_TYPE>(device, arg_run, arg_checkpoint, actor);

    full_assessment<DEVICE, TEST_DEFINITIONS::ENVIRONMENT, typename TEST_DEFINITIONS::prl::ACTOR_TYPE>(device, actor, TEST_DEFINITIONS::penv::parameters);

    bpt::free(device, actor);
}

