#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_REACHER_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_REACHER_OPERATIONS_GENERIC_H
#include "reacher.h"
#include "../operations_generic.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::reacher{
    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T clip(T x, T min, T max){
        return x < min ? min : (x > max ? max : x);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void malloc(DEVICE& device, const rl::environments::Reacher<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void free(DEVICE& device, const rl::environments::Reacher<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void init(DEVICE& device, const rl::environments::Reacher<SPEC>& env){}
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_parameters(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, RNG& rng){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void initial_parameters(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters){}
    template<typename DEVICE, typename SPEC, typename STATE_SPEC>
    static void initial_state(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, typename rl::environments::reacher::State<STATE_SPEC>& state){
        state.x = 0;
        state.y = 0;
        state.target_x = 0.5;
        state.target_y = 0.5;
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, typename rl::environments::reacher::State<STATE_SPEC>& state, RNG& rng){
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        state.x = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.y = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.target_x = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.target_y = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::reacher::State<STATE_SPEC>& next_state, RNG& rng){
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 2);
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        T vx = rl::environments::reacher::clip(get(action, 0, 0), (T)-PARAMS::ACTION_LIMIT, (T)PARAMS::ACTION_LIMIT) * PARAMS::MAX_VELOCITY;
        T vy = rl::environments::reacher::clip(get(action, 0, 1), (T)-PARAMS::ACTION_LIMIT, (T)PARAMS::ACTION_LIMIT) * PARAMS::MAX_VELOCITY;
        next_state.x = rl::environments::reacher::clip(state.x + vx * PARAMS::DT, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE);
        next_state.y = rl::environments::reacher::clip(state.y + vy * PARAMS::DT, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE);
        next_state.target_x = state.target_x;
        next_state.target_y = state.target_y;
        return PARAMS::DT;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::reacher::State<STATE_SPEC>& next_state, RNG& rng){
        using T = typename SPEC::T;
        T dx = next_state.x - next_state.target_x;
        T dy = next_state.y - next_state.target_y;
        T distance = math::sqrt(device.math, dx * dx + dy * dy);
        return -distance;
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Reacher<SPEC>& env, const typename rl::environments::Reacher<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const typename rl::environments::reacher::ObservationDense<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 4);
        set(observation, 0, 0, state.x);
        set(observation, 0, 1, state.y);
        set(observation, 0, 2, state.target_x);
        set(observation, 0, 3, state.target_y);
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename OBS_TYPE_TI, OBS_TYPE_TI OBS_HEIGHT, OBS_TYPE_TI OBS_WIDTH, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Reacher<SPEC>& env, const typename rl::environments::Reacher<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const typename rl::environments::reacher::ObservationImage<OBS_TYPE_TI, OBS_HEIGHT, OBS_WIDTH>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using PARAMS = typename SPEC::PARAMETERS;
        static_assert(OBS_SPEC::ROWS == 1);
        static constexpr TI H = OBS_HEIGHT;
        static constexpr TI W = OBS_WIDTH;
        static constexpr TI C = 3;
        static_assert(OBS_SPEC::COLS == H * W * C);
        T sigma_sq = (T)(H * H) / (T)64;
        T agent_px = (state.x + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * H;
        T agent_py = (state.y + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * W;
        T target_px = (state.target_x + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * H;
        T target_py = (state.target_y + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * W;
        for(TI h = 0; h < H; h++){
            for(TI w = 0; w < W; w++){
                T dh_agent = (T)h + (T)0.5 - agent_px;
                T dw_agent = (T)w + (T)0.5 - agent_py;
                T agent_intensity = math::exp(device.math, -(dh_agent * dh_agent + dw_agent * dw_agent) / sigma_sq);
                T dh_target = (T)h + (T)0.5 - target_px;
                T dw_target = (T)w + (T)0.5 - target_py;
                T target_intensity = math::exp(device.math, -(dh_target * dh_target + dw_target * dw_target) / sigma_sq);
                TI base = h * W * C + w * C;
                set(observation, 0, base + 0, target_intensity);
                set(observation, 0, base + 1, (T)0);
                set(observation, 0, base + 2, agent_intensity);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC> state, RNG& rng){
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        T dx = state.x - state.target_x;
        T dy = state.y - state.target_y;
        T distance = math::sqrt(device.math, dx * dx + dy * dy);
        return distance < PARAMS::TARGET_RADIUS;
    }
    template<typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT typename STATE_SPEC::T abs_diff(DEVICE& device, const rl::environments::reacher::State<STATE_SPEC>& s1, const rl::environments::reacher::State<STATE_SPEC>& s2){
        using T = typename STATE_SPEC::T;
        T acc = 0;
        acc += math::abs(device.math, s1.x - s2.x);
        acc += math::abs(device.math, s1.y - s2.y);
        acc += math::abs(device.math, s1.target_x - s2.target_x);
        acc += math::abs(device.math, s1.target_y - s2.target_y);
        return acc;
    }
    template<typename DEVICE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T abs_diff(DEVICE& device, const rl::environments::reacher::DefaultParameters<T>& p1, const rl::environments::reacher::DefaultParameters<T>& p2){
        return 0;
    }
    // ======================== ReacherVisual operations (delegate to Reacher) ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void malloc(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void free(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void init(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env){}
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_parameters(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env, typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters, RNG& rng){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void initial_parameters(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env, typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters){}
    template<typename DEVICE, typename SPEC, typename STATE_SPEC>
    static void initial_state(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env, typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters, typename rl::environments::reacher::State<STATE_SPEC>& state){
        state.x = 0;
        state.y = 0;
        state.target_x = 0.5;
        state.target_y = 0.5;
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env, typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters, typename rl::environments::reacher::State<STATE_SPEC>& state, RNG& rng){
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        state.x = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.y = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.target_x = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.target_y = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env, typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::reacher::State<STATE_SPEC>& next_state, RNG& rng){
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 2);
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        T vx = rl::environments::reacher::clip(get(action, 0, 0), (T)-PARAMS::ACTION_LIMIT, (T)PARAMS::ACTION_LIMIT) * PARAMS::MAX_VELOCITY;
        T vy = rl::environments::reacher::clip(get(action, 0, 1), (T)-PARAMS::ACTION_LIMIT, (T)PARAMS::ACTION_LIMIT) * PARAMS::MAX_VELOCITY;
        next_state.x = rl::environments::reacher::clip(state.x + vx * PARAMS::DT, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE);
        next_state.y = rl::environments::reacher::clip(state.y + vy * PARAMS::DT, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE);
        next_state.target_x = state.target_x;
        next_state.target_y = state.target_y;
        return PARAMS::DT;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env, typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::reacher::State<STATE_SPEC>& next_state, RNG& rng){
        using T = typename SPEC::T;
        T dx = next_state.x - next_state.target_x;
        T dy = next_state.y - next_state.target_y;
        T distance = math::sqrt(device.math, dx * dx + dy * dy);
        return -distance;
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env, const typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const typename rl::environments::reacher::ObservationDense<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 4);
        set(observation, 0, 0, state.x);
        set(observation, 0, 1, state.y);
        set(observation, 0, 2, state.target_x);
        set(observation, 0, 3, state.target_y);
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename OBS_TYPE_TI, OBS_TYPE_TI OBS_HEIGHT, OBS_TYPE_TI OBS_WIDTH, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env, const typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const typename rl::environments::reacher::ObservationImage<OBS_TYPE_TI, OBS_HEIGHT, OBS_WIDTH>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using PARAMS = typename SPEC::PARAMETERS;
        static_assert(OBS_SPEC::ROWS == 1);
        static constexpr TI H = OBS_HEIGHT;
        static constexpr TI W = OBS_WIDTH;
        static constexpr TI C = 3;
        static_assert(OBS_SPEC::COLS == H * W * C);
        T sigma_sq = (T)(H * H) / (T)64;
        T agent_px = (state.x + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * H;
        T agent_py = (state.y + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * W;
        T target_px = (state.target_x + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * H;
        T target_py = (state.target_y + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * W;
        for(TI h = 0; h < H; h++){
            for(TI w = 0; w < W; w++){
                T dh_agent = (T)h + (T)0.5 - agent_px;
                T dw_agent = (T)w + (T)0.5 - agent_py;
                T agent_intensity = math::exp(device.math, -(dh_agent * dh_agent + dw_agent * dw_agent) / sigma_sq);
                T dh_target = (T)h + (T)0.5 - target_px;
                T dw_target = (T)w + (T)0.5 - target_py;
                T target_intensity = math::exp(device.math, -(dh_target * dh_target + dw_target * dw_target) / sigma_sq);
                TI base = h * W * C + w * C;
                set(observation, 0, base + 0, target_intensity);
                set(observation, 0, base + 1, (T)0);
                set(observation, 0, base + 2, agent_intensity);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename OBS_TYPE_TI, OBS_TYPE_TI OBS_HEIGHT, OBS_TYPE_TI OBS_WIDTH, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env, const typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC>& state, const typename rl::environments::reacher::ObservationImageFlat<OBS_TYPE_TI, OBS_HEIGHT, OBS_WIDTH>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using PARAMS = typename SPEC::PARAMETERS;
        static_assert(OBS_SPEC::ROWS == 1);
        static constexpr TI H = OBS_HEIGHT;
        static constexpr TI W = OBS_WIDTH;
        static constexpr TI C = 3;
        static_assert(OBS_SPEC::COLS == H * W * C);
        T sigma_sq = (T)(H * H) / (T)64;
        T agent_px = (state.x + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * H;
        T agent_py = (state.y + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * W;
        T target_px = (state.target_x + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * H;
        T target_py = (state.target_y + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * W;
        for(TI h = 0; h < H; h++){
            for(TI w = 0; w < W; w++){
                T dh_agent = (T)h + (T)0.5 - agent_px;
                T dw_agent = (T)w + (T)0.5 - agent_py;
                T agent_intensity = math::exp(device.math, -(dh_agent * dh_agent + dw_agent * dw_agent) / sigma_sq);
                T dh_target = (T)h + (T)0.5 - target_px;
                T dw_target = (T)w + (T)0.5 - target_py;
                T target_intensity = math::exp(device.math, -(dh_target * dh_target + dw_target * dw_target) / sigma_sq);
                TI base = h * W * C + w * C;
                set(observation, 0, base + 0, target_intensity);
                set(observation, 0, base + 1, (T)0);
                set(observation, 0, base + 2, agent_intensity);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::ReacherVisual<SPEC>& env, typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters, const typename rl::environments::reacher::State<STATE_SPEC> state, RNG& rng){
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        T dx = state.x - state.target_x;
        T dy = state.y - state.target_y;
        T distance = math::sqrt(device.math, dx * dx + dy * dy);
        return distance < PARAMS::TARGET_RADIUS;
    }
    // ======================== ReacherVisualMemory operations ========================
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void malloc(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void free(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void init(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env){}
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_parameters(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env, typename rl::environments::ReacherVisualMemory<SPEC>::Parameters& parameters, RNG& rng){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void initial_parameters(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env, typename rl::environments::ReacherVisualMemory<SPEC>::Parameters& parameters){}
    template<typename DEVICE, typename SPEC, typename STATE_SPEC>
    static void initial_state(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env, typename rl::environments::ReacherVisualMemory<SPEC>::Parameters& parameters, typename rl::environments::reacher::StateSequentialTargets<STATE_SPEC>& state){
        using PARAMS = typename SPEC::PARAMETERS;
        state.x = 0;
        state.y = 0;
        state.target1_x = 0.5;
        state.target1_y = 0.5;
        state.target2_x = PARAMS::NUM_TARGETS > 1 ? -0.5 : 0.5;
        state.target2_y = PARAMS::NUM_TARGETS > 1 ? -0.5 : 0.5;
        state.step = 0;
        state.current_target = 0;
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env, typename rl::environments::ReacherVisualMemory<SPEC>::Parameters& parameters, typename rl::environments::reacher::StateSequentialTargets<STATE_SPEC>& state, RNG& rng){
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        state.x = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.y = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.target1_x = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        state.target1_y = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        if(PARAMS::NUM_TARGETS > 1){
            state.target2_x = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
            state.target2_y = random::uniform_real_distribution(device.random, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE, rng);
        }
        else{
            state.target2_x = state.target1_x;
            state.target2_y = state.target1_y;
        }
        state.step = 0;
        state.current_target = 0;
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env, typename rl::environments::ReacherVisualMemory<SPEC>::Parameters& parameters, const typename rl::environments::reacher::StateSequentialTargets<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::reacher::StateSequentialTargets<STATE_SPEC>& next_state, RNG& rng){
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 2);
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using PARAMS = typename SPEC::PARAMETERS;
        T vx = rl::environments::reacher::clip(get(action, 0, 0), (T)-PARAMS::ACTION_LIMIT, (T)PARAMS::ACTION_LIMIT) * PARAMS::MAX_VELOCITY;
        T vy = rl::environments::reacher::clip(get(action, 0, 1), (T)-PARAMS::ACTION_LIMIT, (T)PARAMS::ACTION_LIMIT) * PARAMS::MAX_VELOCITY;
        next_state.x = rl::environments::reacher::clip(state.x + vx * PARAMS::DT, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE);
        next_state.y = rl::environments::reacher::clip(state.y + vy * PARAMS::DT, -PARAMS::ARENA_SIZE, PARAMS::ARENA_SIZE);
        next_state.target1_x = state.target1_x;
        next_state.target1_y = state.target1_y;
        next_state.target2_x = state.target2_x;
        next_state.target2_y = state.target2_y;
        next_state.step = state.step + 1;
        if(state.current_target == 0){
            T dx = next_state.x - next_state.target1_x;
            T dy = next_state.y - next_state.target1_y;
            T distance = math::sqrt(device.math, dx * dx + dy * dy);
            next_state.current_target = distance < PARAMS::TARGET_RADIUS ? 1 : 0;
        }
        else{
            next_state.current_target = state.current_target;
        }
        return PARAMS::DT;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env, typename rl::environments::ReacherVisualMemory<SPEC>::Parameters& parameters, const typename rl::environments::reacher::StateSequentialTargets<STATE_SPEC>& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::reacher::StateSequentialTargets<STATE_SPEC>& next_state, RNG& rng){
        using T = typename SPEC::T;
        T dx, dy;
        if(next_state.current_target == 0){
            dx = next_state.x - next_state.target1_x;
            dy = next_state.y - next_state.target1_y;
        }
        else{
            dx = next_state.x - next_state.target2_x;
            dy = next_state.y - next_state.target2_y;
        }
        T distance = math::sqrt(device.math, dx * dx + dy * dy);
        return -distance;
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env, const typename rl::environments::ReacherVisualMemory<SPEC>::Parameters& parameters, const typename rl::environments::reacher::StateSequentialTargets<STATE_SPEC>& state, const typename rl::environments::reacher::ObservationDenseSequentialTargets<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 7);
        using T = typename SPEC::T;
        set(observation, 0, 0, state.x);
        set(observation, 0, 1, state.y);
        set(observation, 0, 2, state.target1_x);
        set(observation, 0, 3, state.target1_y);
        set(observation, 0, 4, state.target2_x);
        set(observation, 0, 5, state.target2_y);
        set(observation, 0, 6, (T)state.current_target);
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename OBS_TYPE_TI, OBS_TYPE_TI OBS_HEIGHT, OBS_TYPE_TI OBS_WIDTH, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env, const typename rl::environments::ReacherVisualMemory<SPEC>::Parameters& parameters, const typename rl::environments::reacher::StateSequentialTargets<STATE_SPEC>& state, const typename rl::environments::reacher::ObservationImage<OBS_TYPE_TI, OBS_HEIGHT, OBS_WIDTH>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using PARAMS = typename SPEC::PARAMETERS;
        static_assert(OBS_SPEC::ROWS == 1);
        static constexpr TI H = OBS_HEIGHT;
        static constexpr TI W = OBS_WIDTH;
        static constexpr TI C = 3;
        static_assert(OBS_SPEC::COLS == H * W * C);
        T sigma_sq = (T)(H * H) / (T)64;
        T agent_px = (state.x + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * H;
        T agent_py = (state.y + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * W;
        T target1_px = (state.target1_x + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * H;
        T target1_py = (state.target1_y + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * W;
        T target2_px = PARAMS::NUM_TARGETS > 1 ? (state.target2_x + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * H : (T)0;
        T target2_py = PARAMS::NUM_TARGETS > 1 ? (state.target2_y + PARAMS::ARENA_SIZE) / ((T)2 * PARAMS::ARENA_SIZE) * W : (T)0;
        bool show_targets = (state.step == 0);
        for(TI h = 0; h < H; h++){
            for(TI w = 0; w < W; w++){
                T dh_agent = (T)h + (T)0.5 - agent_px;
                T dw_agent = (T)w + (T)0.5 - agent_py;
                T agent_intensity = math::exp(device.math, -(dh_agent * dh_agent + dw_agent * dw_agent) / sigma_sq);
                T target1_intensity = (T)0;
                T target2_intensity = (T)0;
                if(show_targets){
                    T dh_target1 = (T)h + (T)0.5 - target1_px;
                    T dw_target1 = (T)w + (T)0.5 - target1_py;
                    target1_intensity = math::exp(device.math, -(dh_target1 * dh_target1 + dw_target1 * dw_target1) / sigma_sq);
                    if(PARAMS::NUM_TARGETS > 1){
                        T dh_target2 = (T)h + (T)0.5 - target2_px;
                        T dw_target2 = (T)w + (T)0.5 - target2_py;
                        target2_intensity = math::exp(device.math, -(dh_target2 * dh_target2 + dw_target2 * dw_target2) / sigma_sq);
                    }
                }
                TI base = h * W * C + w * C;
                set(observation, 0, base + 0, target1_intensity);
                set(observation, 0, base + 1, target2_intensity);
                set(observation, 0, base + 2, agent_intensity);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename STATE_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::ReacherVisualMemory<SPEC>& env, typename rl::environments::ReacherVisualMemory<SPEC>::Parameters& parameters, const typename rl::environments::reacher::StateSequentialTargets<STATE_SPEC> state, RNG& rng){
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        if(state.current_target == 0){
            return false;
        }
        T dx = state.x - state.target2_x;
        T dy = state.y - state.target2_y;
        T distance = math::sqrt(device.math, dx * dx + dy * dy);
        return distance < PARAMS::TARGET_RADIUS;
    }
    template<typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT typename STATE_SPEC::T abs_diff(DEVICE& device, const rl::environments::reacher::StateSequentialTargets<STATE_SPEC>& s1, const rl::environments::reacher::StateSequentialTargets<STATE_SPEC>& s2){
        using T = typename STATE_SPEC::T;
        T acc = 0;
        acc += math::abs(device.math, s1.x - s2.x);
        acc += math::abs(device.math, s1.y - s2.y);
        acc += math::abs(device.math, s1.target1_x - s2.target1_x);
        acc += math::abs(device.math, s1.target1_y - s2.target1_y);
        acc += math::abs(device.math, s1.target2_x - s2.target2_x);
        acc += math::abs(device.math, s1.target2_y - s2.target2_y);
        acc += (T)(s1.step > s2.step ? s1.step - s2.step : s2.step - s1.step);
        acc += (T)(s1.current_target > s2.current_target ? s1.current_target - s2.current_target : s2.current_target - s1.current_target);
        return acc;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
