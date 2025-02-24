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

  return 0;
}