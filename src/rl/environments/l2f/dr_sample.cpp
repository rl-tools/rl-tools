#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_helper_generic.h>


namespace rlt = rl_tools;


#include <iostream>
#include <chrono>


using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

using T = float;


using ENVIRONMENT = rlt::rl::environments::l2f::parameters::DefaultParameters<T, TI>::ENVIRONMENT;



int main(int argc, char** argv) {
    DEVICE device;
    TI seed;
    if(argc < 2) {
        seed = std::chrono::system_clock::now().time_since_epoch().count();
    }
    else {
        seed = std::stoi(argv[1]);
    }
    auto rng = rlt::random::default_engine(device.random, seed);

    ENVIRONMENT::Parameters nominal_parameters, parameters;
    ENVIRONMENT::State state;
    ENVIRONMENT env;

    rlt::initial_parameters(device, env, nominal_parameters);
    rlt::initial_parameters(device, env, parameters);
    parameters.domain_randomization = {
        1.5, // thrust_to_weight_min;
        5.0, // thrust_to_weight_max;
        0.027, // mass_min;
        5.00, // mass_max;
        1.0, // mass_size_deviation;
        0.0, // motor_time_constant;
        0.0 // rotor_torque_constant;
    };
    rlt::sample_initial_parameters(device, env, parameters, rng, false);



    std::string nominal_parameters_json = rlt::json(device, env, nominal_parameters);
    std::string parameters_json = rlt::json(device, env, parameters);

    rlt::compare_parameters(device, nominal_parameters, parameters);

    std::cout << parameters_json << std::endl;
}