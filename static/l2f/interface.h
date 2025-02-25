#include "environment.h"

#include <stdint.h>

struct State{
    DEVICE device;
    RNG rng;
    ENVIRONMENT env;
    ENVIRONMENT::Parameters parameters;
    ENVIRONMENT::State state, next_state;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM, false>> action;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM, false>> observation;
};

extern "C" void* memcpy(void* dest, const void* src, TI count) {
    auto* d = static_cast<char*>(dest);
    auto* s = static_cast<const char*>(src);
    for (TI i = 0; i < count; i++) {
        d[i] = s[i];
    }
    return dest;
}

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
        rlt::init(state.device);
        rlt::malloc(state.device, state.rng);
        rlt::init(state.device, state.rng, seed);
        rlt::malloc(state.device, state.env);
        rlt::sample_initial_parameters(state.device, state.env, state.parameters, state.rng);
        rlt::sample_initial_state(state.device, state.env, state.parameters, state.state, state.rng);
    }

    void sample_initial_parameters(STATE_PTR_TYPE state_ptr){
        State& state = *(GET_STATE(state_ptr));
        rlt::sample_initial_parameters(state.device, state.env, state.parameters, state.rng);
    }
    void sample_initial_state(STATE_PTR_TYPE state_ptr){
        State& state = *(GET_STATE(state_ptr));
        rlt::sample_initial_state(state.device, state.env, state.parameters, state.state, state.rng);
    }
    void set_action(STATE_PTR_TYPE state_ptr, uint32_t action_i, T value){
        State& state = *(GET_STATE(state_ptr));
        rlt::set(state.action, 0, action_i, value);
    }
    T get_observation(STATE_PTR_TYPE state_ptr, uint32_t observation_i){
        State& state = *(GET_STATE(state_ptr));
        rlt::observe(state.device, state.env, state.parameters, state.state, ENVIRONMENT::Observation{}, state.observation, state.rng);
        return rlt::get(state.observation, 0, observation_i);
    }
    void step(STATE_PTR_TYPE state_ptr){
        State& state = *(GET_STATE(state_ptr));
        ENVIRONMENT::State next_state;
        rlt::step(state.device, state.env, state.parameters, state.state, state.action, next_state, state.rng);
        state.state = next_state;
    }
}

