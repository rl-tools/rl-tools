#include "environment.h"
#include "visual_environment.h"
#if defined(EMSCRIPTEN) || defined(STDLIB)
#include "environment_helper.h"
#endif
#if defined(EMSCRIPTEN)
#include <emscripten/val.h>
#endif

#include <stdint.h>

template<typename T_ENVIRONMENT>
struct SimulatorState{
    using ENVIRONMENT = T_ENVIRONMENT;
    static constexpr bool VISUAL = ENVIRONMENT::SPEC::STATIC_PARAMETERS::ACTION_INTERFACE == rlt::rl::environments::l2f::parameters::ActionInterface::CTBR;
    DEVICE device;
    RNG rng;
    ENVIRONMENT env;
    typename ENVIRONMENT::Parameters parameters;
    typename ENVIRONMENT::State state, next_state;
    visual_policy::PolicyState<VISUAL> policy;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM, false>> action{};
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM, false>> observation{};
    SimulatorState(uint32_t seed){
        rlt::init(this->device);
        rlt::malloc(this->device, this->rng);
        rlt::init(this->device, this->rng, seed);
        rlt::malloc(this->device, this->env);
        rlt::init(this->device, this->env);
        rlt::sample_initial_parameters(this->device, this->env, this->parameters, this->rng);
        rlt::sample_initial_state(this->device, this->env, this->parameters, this->state, this->rng);
        prepare_policy_observation(true);
    };
    void prepare_policy_observation(bool reset){
        using rlt::get;
        using rlt::set;
        using rlt::view;
        if constexpr(VISUAL){
            auto base = view(device, policy.observation, rlt::matrix::ViewSpec<1, 62>{}, 0, 0);
            for(TI axis = 0; axis < 3; axis++){
                T error = rlt::random::normal_distribution::sample(device.random, T(0), T(0.001), rng);
                if(reset){
                    state.bias_estimate[axis] = state.gyro_bias[axis] + error;
                }
            }
            rlt::observe(device, env, parameters, state, visual_policy::ACTOR_OBSERVATION{}, base, rng);
            if(reset){
                set(base, 0, 40, T(9.81));
            }
            T gyro[3], world_z[3];
            for(TI axis = 0; axis < 3; axis++){
                if(reset){
                    state.gyro[axis] = get(base, 0, 3 + axis);
                }
                else{
                    state.bias_estimate[axis] += (state.gyro[axis] - state.bias_estimate[axis]) * parameters.integration.dt / T(10);
                }
                gyro[axis] = state.gyro[axis] - state.bias_estimate[axis];
                world_z[axis] = get(base, 0, axis);
                set(base, 0, 3 + axis, gyro[axis]);
            }
            if(reset){
                rlt::init(device, policy.inertial, gyro);
            }
            else{
                rlt::step(device, policy.inertial, world_z, state.accelerometer, gyro, parameters.integration.dt);
            }
            for(TI axis = 0; axis < 3; axis++){
                set(policy.observation, 0, 62 + axis, policy.inertial.velocity[axis]);
            }
        }
    };
    void initial_parameters(){
        rlt::initial_parameters(this->device, this->env, this->parameters);
    }
    void initial_state(){
        rlt::initial_state(this->device, this->env, this->parameters, this->state);
        prepare_policy_observation(true);
    };
    void sample_initial_parameters(){
        rlt::sample_initial_parameters(this->device, this->env, this->parameters, this->rng);
    };
    void sample_initial_state(){
        rlt::sample_initial_state(this->device, this->env, this->parameters, this->state, this->rng);
        prepare_policy_observation(true);
    };
    void set_action(uint32_t action_i, T value){
        rlt::set(this->action, 0, action_i, value);
    };
    void observe(){
        rlt::observe(this->device, this->env, this->parameters, this->state, typename ENVIRONMENT::Observation{}, this->observation, this->rng);
    }
    T step(){
        typename ENVIRONMENT::State next_state;
        T dt = rlt::step(this->device, this->env, this->parameters, this->state, this->action, next_state, this->rng);
        this->state = next_state;
        prepare_policy_observation(false);
        return dt;
    };
#if defined(EMSCRIPTEN) || defined(STDLIB)
    std::string get_parameters(){
        return rlt::json(device, this->env, this->parameters);
    }
    std::string get_state(){
        return rlt::json(device, this->env, this->parameters, this->state);
    }
    mutable std::array<float, ENVIRONMENT::Observation::DIM> observation_cache{};
    emscripten::val get_observation() const {
        for(TI i = 0; i < ENVIRONMENT::Observation::DIM; i++){
            observation_cache[i] = rlt::get(this->observation, 0, i);
        }
        return emscripten::val(emscripten::typed_memory_view(ENVIRONMENT::Observation::DIM, observation_cache.data()));
    }
    emscripten::val get_policy_observation() const {
        using rlt::get;
        if constexpr(VISUAL){
            for(TI i = 0; i < 65; i++){
                policy.cache[i] = get(policy.observation, 0, i);
            }
            return emscripten::val(emscripten::typed_memory_view(65, policy.cache));
        }
        else{
            return get_observation();
        }
    }
    mutable std::array<float, ENVIRONMENT::ACTION_DIM> action_cache{};
    emscripten::val get_action() const {
        for(TI i = 0; i < ENVIRONMENT::ACTION_DIM; i++){
            action_cache[i] = rlt::get(this->action, 0, i);
        }
        return emscripten::val(emscripten::typed_memory_view(ENVIRONMENT::ACTION_DIM, action_cache.data()));
    }
    std::string get_ui(){
        return rlt::get_ui(this->device, this->env);
    }

    mutable std::array<float,3> position_cache{};
    emscripten::val get_position() const {
        position_cache[0] = this->state.position[0];
        position_cache[1] = this->state.position[1];
        position_cache[2] = this->state.position[2];
        return emscripten::val(emscripten::typed_memory_view(3, position_cache.data()));
    }
    mutable std::array<float,4> orientation_cache{};
    emscripten::val get_orientation() const {
        orientation_cache[0] = this->state.orientation[0];
        orientation_cache[1] = this->state.orientation[1];
        orientation_cache[2] = this->state.orientation[2];
        orientation_cache[3] = this->state.orientation[3];
        return emscripten::val(emscripten::typed_memory_view(4, orientation_cache.data()));
    }
    void set_parameters(std::string params){
        if constexpr(VISUAL){
            auto merged = nlohmann::json::parse(get_parameters());
            merged.merge_patch(nlohmann::json::parse(params));
            rlt::from_json(device, env, merged, parameters);
        }
        else{
            rlt::from_json(device, this->env, params, this->parameters);
        }
    }
    void set_state(std::string state){
        rlt::from_json(device, this->env, this->parameters, state, this->state);
    }
#endif
};

using State = SimulatorState<ENVIRONMENT>;
using VisualState = SimulatorState<visual_policy::ENVIRONMENT>;

#ifndef EMSCRIPTEN
extern "C" void* memcpy(void* dest, const void* src, TI count) {
    auto* d = static_cast<char*>(dest);
    auto* s = static_cast<const char*>(src);
    for (TI i = 0; i < count; i++) {
        d[i] = s[i];
    }
    return dest;
}
#endif

#ifdef WASM
#define STATE_PTR_TYPE uint32_t
#define GET_STATE(state_ptr) reinterpret_cast<State*>(state_ptr)
#else
#define STATE_PTR_TYPE State*
#define GET_STATE(state_ptr) state_ptr
#endif


extern "C"{
    uint32_t state_size = sizeof(State);
    uint32_t action_dim = ENVIRONMENT::ACTION_DIM;
    uint32_t observation_dim = ENVIRONMENT::Observation::DIM;
    void init(STATE_PTR_TYPE state_ptr, uint32_t seed){
        State& state = *(GET_STATE(state_ptr));
        state = State(seed);
    }
    void initial_parameters(STATE_PTR_TYPE state_ptr){
        State& state = *(GET_STATE(state_ptr));
        state.initial_parameters();
    }
    void initial_state(STATE_PTR_TYPE state_ptr){
        State& state = *(GET_STATE(state_ptr));
        state.initial_state();
    }
    void sample_initial_parameters(STATE_PTR_TYPE state_ptr){
        State& state = *(GET_STATE(state_ptr));
        state.sample_initial_parameters();
    }
    void sample_initial_state(STATE_PTR_TYPE state_ptr){
        State& state = *(GET_STATE(state_ptr));
        state.sample_initial_state();
    }
    void set_action(STATE_PTR_TYPE state_ptr, uint32_t action_i, T value){
        State& state = *(GET_STATE(state_ptr));
        state.set_action(action_i, value);
    }
    void observe(STATE_PTR_TYPE state_ptr){
        State& state = *(GET_STATE(state_ptr));
        state.observe();
    }
    T step(STATE_PTR_TYPE state_ptr){
        State& state = *(GET_STATE(state_ptr));
        return state.step();
    }
}


#ifdef EMSCRIPTEN
#include <emscripten/bind.h>
using namespace emscripten;

template<typename State>
void bind_state(const char* name){
    class_<State>(name)
        .template constructor<uint32_t>()
        .function("initial_parameters", &State::initial_parameters)
        .function("initial_state", &State::initial_state)
        .function("sample_initial_parameters", &State::sample_initial_parameters)
        .function("sample_initial_state", &State::sample_initial_state)
        .function("set_action", &State::set_action)
        .function("observe", &State::observe)
        .function("get_observation", &State::get_observation)
        .function("get_policy_observation", &State::get_policy_observation)
        .function("step", &State::step)
        .function("get_ui", &State::get_ui)
        .function("get_parameters", &State::get_parameters)
        .function("get_state", &State::get_state)
        .function("get_action", &State::get_action)
        .function("get_position", &State::get_position)
        .function("get_orientation", &State::get_orientation)
        .function("set_parameters", &State::set_parameters)
        .function("set_state", &State::set_state)
        .property("action_dim", +[](const State& state) -> uint32_t { return State::ENVIRONMENT::ACTION_DIM; })
        .property("observation_dim", +[](const State& state) -> uint32_t { return State::ENVIRONMENT::Observation::DIM; });
}
EMSCRIPTEN_BINDINGS(state_module){
    bind_state<State>("State");
    bind_state<VisualState>("VisualState");
}
#endif
