#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>


namespace rlt = rl_tools;


#include <iostream>
#include <chrono>


using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

using T = float;


using ENVIRONMENT = rlt::rl::environments::l2f::parameters::DefaultParameters<T, TI>::ENVIRONMENT;

template <typename OUT, typename T>
void compareDynamicsParameters(OUT& out, const T& nominal, const T& perturbed) {
    auto percentageChange = [](auto nominal_value, auto perturbed_value) {
        return ((perturbed_value - nominal_value) / nominal_value) * 100.0;
    };

    out << std::fixed << std::setprecision(4);

    out << "Mass: " << nominal.mass << " -> " << perturbed.mass
              << " (" << percentageChange(nominal.mass, perturbed.mass) << "%)\n";

    for (int i = 0; i < 3; ++i) {
        out << "J[" << i << "][" << i << "]: " << nominal.J[i][i] << " -> " << perturbed.J[i][i]
                  << " (" << percentageChange(nominal.J[i][i], perturbed.J[i][i]) << "%)\n";
    }

    for (int i = 0; i < 3; ++i) {
        out << "Rotor thrust coefficient[" << i << "]: " << nominal.rotor_thrust_coefficients[i] << " -> "
                  << perturbed.rotor_thrust_coefficients[i] << " (" << percentageChange(nominal.rotor_thrust_coefficients[i], perturbed.rotor_thrust_coefficients[i]) << "%)\n";
    }

    out << "Rotor torque constant: " << nominal.rotor_torque_constant << " -> " << perturbed.rotor_torque_constant
              << " (" << percentageChange(nominal.rotor_torque_constant, perturbed.rotor_torque_constant) << "%)\n";

    out << "Motor time constant: " << nominal.motor_time_constant << " -> " << perturbed.motor_time_constant
              << " (" << percentageChange(nominal.motor_time_constant, perturbed.motor_time_constant) << "%)\n";

    out << "Hovering throttle relative: " << nominal.hovering_throttle_relative << " -> " << perturbed.hovering_throttle_relative
              << " (" << percentageChange(nominal.hovering_throttle_relative, perturbed.hovering_throttle_relative) << "%)\n";
}


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
        1.0, // torque_to_inertia;
        1.0, // mass_size_deviation;
        0.0, // motor_time_constant;
        0.0 // rotor_torque_constant;
    };
    rlt::sample_initial_parameters(device, env, parameters, rng, false);



    std::string nominal_parameters_json = rlt::json(device, env, nominal_parameters);
    std::string parameters_json = rlt::json(device, env, parameters);

    compareDynamicsParameters(std::cerr, nominal_parameters.dynamics, parameters.dynamics);

    std::cout << parameters_json << std::endl;
}