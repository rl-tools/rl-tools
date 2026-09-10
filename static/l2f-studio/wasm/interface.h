#include "environment.h"
#if defined(EMSCRIPTEN) || defined(STDLIB)
#include "environment_helper.h"
#endif
#if defined(EMSCRIPTEN)
#include <emscripten/val.h>
#endif

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
        rlt::init(this->device, this->env);
        rlt::sample_initial_parameters(this->device, this->env, this->parameters, this->rng);
        rlt::sample_initial_state(this->device, this->env, this->parameters, this->state, this->rng);
    };
    void initial_parameters(){
        rlt::initial_parameters(this->device, this->env, this->parameters);
    }
    void initial_state(){
        rlt::initial_state(this->device, this->env, this->parameters, this->state);
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
    T step(){
        ENVIRONMENT::State next_state;
        T dt = rlt::step(this->device, this->env, this->parameters, this->state, this->action, next_state, this->rng);
        this->state = next_state;
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
        rlt::from_json(device, this->env, params, this->parameters);
    }
    void set_state(std::string state){
        rlt::from_json(device, this->env, this->parameters, state, this->state);
    }
#endif
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

EMSCRIPTEN_BINDINGS(state_module){
    class_<State>("State")                // Expose the State class as "State" in JS
        .constructor<uint32_t>()          // Bind constructor with a uint32_t seed
        .function("initial_parameters", &State::initial_parameters)
        .function("initial_state", &State::initial_state)
        .function("sample_initial_parameters", &State::sample_initial_parameters)
        .function("sample_initial_state", &State::sample_initial_state)
        .function("set_action", &State::set_action)
        .function("observe", &State::observe)
        .function("get_observation", &State::get_observation)
        .function("step", &State::step)
        .function("get_ui", &State::get_ui)
        .function("get_parameters", &State::get_parameters)
        .function("get_state", &State::get_state)
        .function("get_action", &State::get_action)
        .function("get_position", &State::get_position)
        .function("get_orientation", &State::get_orientation)
        .function("set_parameters", &State::set_parameters)
        .function("set_state", &State::set_state)
        .property("action_dim", +[](const State& state) -> uint32_t { return ENVIRONMENT::ACTION_DIM; })
        .property("observation_dim", +[](const State& state) -> uint32_t { return ENVIRONMENT::Observation::DIM; });
}
#endif