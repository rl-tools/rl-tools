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
    State(uint32_t seed){
        rlt::init(this->device);
        rlt::malloc(this->device, this->rng);
        rlt::init(this->device, this->rng, seed);
        rlt::malloc(this->device, this->env);
        rlt::sample_initial_parameters(this->device, this->env, this->parameters, this->rng);
        rlt::sample_initial_state(this->device, this->env, this->parameters, this->state, this->rng);
    };
    void sample_initial_parameters(){
        rlt::sample_initial_parameters(this->device, this->env, this->parameters, this->rng);
    };
    void sample_initial_state(){
        rlt::sample_initial_state(this->device, this->env, this->parameters, this->state, this->rng);
    };
    void set_action(uint32_t action_i, T value){
        rlt::set(this->action, 0, action_i, value);
    };
    void observe(){
        rlt::observe(this->device, this->env, this->parameters, this->state, ENVIRONMENT::Observation{}, this->observation, this->rng);
    }
    T get_observation(uint32_t observation_i){
        return rlt::get(this->observation, 0, observation_i);
    };
    void step(){
        ENVIRONMENT::State next_state;
        rlt::step(this->device, this->env, this->parameters, this->state, this->action, next_state, this->rng);
        this->state = next_state;
    };

};

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
    T get_observation(STATE_PTR_TYPE state_ptr, uint32_t observation_i){
        State& state = *(GET_STATE(state_ptr));
        return state.get_observation(observation_i);
    }
    void step(STATE_PTR_TYPE state_ptr){
        State& state = *(GET_STATE(state_ptr));
        state.step();
    }
}


#ifdef EMSCRIPTEN
#include <emscripten/bind.h>
using namespace emscripten;

EMSCRIPTEN_BINDINGS(state_module){
    class_<State>("State")                // Expose the State class as "State" in JS
        .constructor<uint32_t>()          // Bind constructor with a uint32_t seed
        .function("sample_initial_parameters", &State::sample_initial_parameters)
        .function("sample_initial_state", &State::sample_initial_state)
        .function("set_action", &State::set_action)
        .function("observe", &State::observe)
        .function("get_observation", &State::get_observation)
        .function("step", &State::step);
}
#endif