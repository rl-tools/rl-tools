#include "l2f.h"


extern "C"{
  int hello(){
    return 1337;
  }
}


int main(){

  DEVICE device;
  RNG rng;

  TI seed = 0;
  rlt::init(device);
  rlt::malloc(device, rng);
  rlt::init(device, rng, seed);

  ENVIRONMENT env;
  ENVIRONMENT::Parameters parameters;
  rlt::malloc(device, env);
  rlt::sample_initial_parameters(device, env, parameters, rng);
  ENVIRONMENT::State state, next_state;
  rlt::sample_initial_state(device, env, parameters, state, rng);
  rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4, false>> action;
  rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM, false>> observation;
  rlt::observe(device, env, parameters, state, ENVIRONMENT::Observation{}, observation, rng);
  rlt::set_all(device, action, 0.0);
  rlt::step(device, env, parameters, state, action, next_state, rng);


  return 0;
}