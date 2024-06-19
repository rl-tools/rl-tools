#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_MULTI_AGENT_BOTTLENECK_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_MULTI_AGENT_BOTTLENECK_OPERATIONS_GENERIC_H
#include "bottleneck.h"
#include "../../operations_generic.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::environments::multi_agent::bottleneck {
        template <typename DEVICE, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T f_mod_python(const DEVICE& dev, T a, T b){
            return a - b * math::floor(dev, a / b);
        }

        template <typename DEVICE, typename T>
        RL_TOOLS_FUNCTION_PLACEMENT T angle_normalize(const DEVICE& dev, T x){
            return f_mod_python(dev, (x + math::PI<T>), (2 * math::PI<T>)) - math::PI<T>;
        }
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::multi_agent::Bottleneck<SPEC>& env, typename rl::environments::multi_agent::Bottleneck<SPEC>::State& state, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(SPEC::PARAMETERS::ARENA_WIDTH/2 - SPEC::PARAMETERS::AGENT_DIAMETER > SPEC::PARAMETERS::AGENT_DIAMETER, "Arena not wide enough for a single agent");
        static_assert((SPEC::PARAMETERS::ARENA_WIDTH/2 - 2*SPEC::PARAMETERS::AGENT_DIAMETER) * (SPEC::PARAMETERS::ARENA_HEIGHT - 2*SPEC::PARAMETERS::AGENT_DIAMETER) > SPEC::PARAMETERS::N_AGENTS * SPEC::PARAMETERS::AGENT_DIAMETER * SPEC::PARAMETERS::AGENT_DIAMETER/4 * 9, "Arena area not large enough for the number of agents");
        for(TI agent_i = 0; agent_i < SPEC::PARAMETERS::N_AGENTS; agent_i++){
            auto& agent_state = state.agent_states[agent_i];
            bool illegal = false;
            do{
                agent_state.position[0] = random::uniform_real_distribution(device.random, SPEC::PARAMETERS::AGENT_DIAMETER, SPEC::PARAMETERS::ARENA_WIDTH / 2 - SPEC::PARAMETERS::AGENT_DIAMETER, rng);
                agent_state.position[1] = random::uniform_real_distribution(device.random, SPEC::PARAMETERS::AGENT_DIAMETER, SPEC::PARAMETERS::ARENA_HEIGHT - SPEC::PARAMETERS::AGENT_DIAMETER, rng);
                agent_state.orientation = random::uniform_real_distribution(device.random, -math::PI<T>, math::PI<T>, rng);
                agent_state.velocity[0] = 0;
                agent_state.velocity[1] = 0;
                agent_state.angular_velocity = 0;
                illegal = false;
                for(TI agent_j = 0; agent_j < agent_i; agent_j++){
                    auto& agent_state_j = state.agent_states[agent_j];
                    T dx = agent_state.position[0] - agent_state_j.position[0];
                    T dy = agent_state.position[1] - agent_state_j.position[1];
                    T d = math::sqrt(device.math, dx * dx + dy * dy);
                    if(d < SPEC::PARAMETERS::AGENT_DIAMETER){
                        illegal = true;
                        break;
                    }
                }
            }while(illegal);
        }
    }
    template<typename DEVICE, typename SPEC>
    static void initial_state(DEVICE& device, const rl::environments::multi_agent::Bottleneck<SPEC>& env, typename rl::environments::multi_agent::Bottleneck<SPEC>::State& state){
        static_assert(SPEC::PARAMETERS::ARENA_WIDTH/2 - SPEC::PARAMETERS::AGENT_DIAMETER > SPEC::PARAMETERS::AGENT_DIAMETER, "Arena not wide enough for a single agent");
        static_assert((SPEC::PARAMETERS::ARENA_HEIGHT - SPEC::PARAMETERS::AGENT_DIAMETER) > SPEC::PARAMETERS::N_AGENTS * SPEC::PARAMETERS::AGENT_DIAMETER, "Arena not tall enough for initializing the agents on a line");
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        for(TI agent_i = 0; agent_i < SPEC::PARAMETERS::N_AGENTS; agent_i++){
            auto& agent_state = state.agent_states[agent_i];
            agent_state.position[0] = SPEC::PARAMETERS::ARENA_HEIGHT / 2;
            agent_state.position[1] = SPEC::PARAMETERS::AGENT_DIAMETER/2 + (agent_i) * SPEC::PARAMETERS::AGENT_DIAMETER * 1.5;
            agent_state.orientation = 0;
            agent_state.velocity[0] = 0;
            agent_state.velocity[1] = 0;
            agent_state.angular_velocity = 0;
        }
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::multi_agent::Bottleneck<SPEC>& env, const typename rl::environments::multi_agent::Bottleneck<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::multi_agent::Bottleneck<SPEC>::State& next_state, RNG& rng) {
        using ENV = rl::environments::multi_agent::Bottleneck<SPEC>;
        static_assert(ACTION_SPEC::ROWS == ENV::PARAMETERS::N_AGENTS);
        static_assert(ACTION_SPEC::COLS == ENV::ACTION_DIM);
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        using PARAMS = typename SPEC::PARAMETERS;
        return SPEC::PARAMETERS::DT;

        for(TI agent_i=0; agent_i < ENV::PARAMETERS::N_AGENTS; agent_i++){
            auto& agent_state = state.agent_states[agent_i];
            auto& agent_next_state = next_state.agent_states[agent_i];
            T force = math::clamp(device.math, get(action, agent_i, 0), (T)-1, (T)1);
            T u = PARAMS::AGENT_MAX_ACCELERATION * force;
            T dt = PARAMS::DT;
            T dx = agent_state.velocity[0] * dt;
            T dy = agent_state.velocity[1] * dt;
            T dtheta = agent_state.angular_velocity * dt;
            T new_x = agent_state.position[0] + dx;
            T new_y = agent_state.position[1] + dy;
            T new_theta = rl::environments::multi_agent::bottleneck::angle_normalize(device, agent_state.orientation + dtheta);
            T new_vx = agent_state.velocity[0] + u * math::cos(device.math, new_theta) * dt;
            T new_vy = agent_state.velocity[1] + u * math::sin(device.math, new_theta) * dt;
            T new_omega = agent_state.angular_velocity + u * dt;
            agent_next_state.position[0] = new_x;
            agent_next_state.position[1] = new_y;
            agent_next_state.orientation = new_theta;
            agent_next_state.velocity[0] = new_vx;
            agent_next_state.velocity[1] = new_vy;
            agent_next_state.angular_velocity = new_omega;
        }
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename REWARD_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reward(DEVICE& device, const rl::environments::multi_agent::Bottleneck<SPEC>& env, const typename rl::environments::multi_agent::Bottleneck<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::multi_agent::Bottleneck<SPEC>::State& next_state, Matrix<REWARD_SPEC>& reward, RNG& rng){
        using ENV = rl::environments::multi_agent::Bottleneck<SPEC>;
        static_assert(ACTION_SPEC::ROWS == ENV::PARAMETERS::N_AGENTS);
        static_assert(ACTION_SPEC::COLS == ENV::ACTION_DIM);
        static_assert(REWARD_SPEC::ROWS == 1);
        static_assert(REWARD_SPEC::COLS == ENV::PARAMETERS::N_AGENTS);
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        for(TI agent_i = 0; agent_i < ENV::PARAMETERS::N_AGENTS; agent_i++){
            set(reward, 0, agent_i, 1);
        }
    }

    template<typename DEVICE, typename SPEC, typename OBS_SPEC, typename OBS_PARAMETERS, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::multi_agent::Bottleneck<SPEC>& env, const typename rl::environments::multi_agent::Bottleneck<SPEC>::State& state, rl::environments::multi_agent::bottleneck::Observation<OBS_PARAMETERS>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        using OBS = rl::environments::multi_agent::bottleneck::Observation<OBS_PARAMETERS>;
        static_assert(OBS_SPEC::ROWS == SPEC::PARAMETERS::N_AGENTS);
        static_assert(OBS_SPEC::COLS == OBS::DIM);
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        for(TI agent_i = 0; agent_i < SPEC::PARAMETERS::N_AGENTS; agent_i++){
            auto& agent_state = state.agent_states[agent_i];
            set(observation, agent_i, 0, agent_state.position[0]);
            set(observation, agent_i, 1, agent_state.position[1]);
            set(observation, agent_i, 2, agent_state.orientation);
            set(observation, agent_i, 3, agent_state.velocity[0]);
            set(observation, agent_i, 4, agent_state.velocity[1]);
            set(observation, agent_i, 5, agent_state.angular_velocity);
            for(TI lidar_i = 0; lidar_i < SPEC::PARAMETERS::LIDAR_RESOLUTION; lidar_i++){
                set(observation, agent_i, 6 + lidar_i, 0); //tbi
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename OBS_SPEC, typename OBS_PARAMETERS, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::multi_agent::Bottleneck<SPEC>& env, const typename rl::environments::multi_agent::Bottleneck<SPEC>::State& state, rl::environments::multi_agent::bottleneck::ObservationPrivileged<OBS_PARAMETERS>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        using OBS = rl::environments::multi_agent::bottleneck::Observation<OBS_PARAMETERS>;
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == OBS::DIM);
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        for(TI agent_i = 0; agent_i < SPEC::PARAMETERS::N_AGENTS; agent_i++){
            TI agent_offset = agent_i * 6;
            auto& agent_state = state.agent_states[agent_i];
            set(observation, 0, agent_offset + 0, agent_state.position[0]);
            set(observation, 0, agent_offset + 1, agent_state.position[1]);
            set(observation, 0, agent_offset + 2, agent_state.orientation);
            set(observation, 0, agent_offset + 3, agent_state.velocity[0]);
            set(observation, 0, agent_offset + 4, agent_state.velocity[1]);
            set(observation, 0, agent_offset + 5, agent_state.angular_velocity);
        }
    }
    template<typename DEVICE, typename SPEC, typename TERMINATED_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void terminated(DEVICE& device, const rl::environments::multi_agent::Bottleneck<SPEC>& env, const typename rl::environments::multi_agent::Bottleneck<SPEC>::State state, Matrix<TERMINATED_SPEC>& terminated, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(TERMINATED_SPEC::ROWS == 1);
        static_assert(TERMINATED_SPEC::COLS == SPEC::PARAMETERS::N_AGENTS);
        static_assert(utils::typing::is_same_v<typename TERMINATED_SPEC::T, bool>);
        for(TI agent_i = 0; agent_i < SPEC::PARAMETERS::N_AGENTS; agent_i++){
            set(terminated, 0, agent_i, false);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
