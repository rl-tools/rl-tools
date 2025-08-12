#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>

#include <fstream>
#include <filesystem>
namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = DEVICE::index_t;
using T = double;
constexpr bool DYNAMIC_ALLOCATION = true;

using PARAMETER_FACTORY = rlt::rl::environments::l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI, rlt::rl::environments::l2f::parameters::DEFAULT_DOMAIN_RANDOMIZATION_OPTIONS<true>>;
using ENVIRONMENT = rlt::rl::environments::Multirotor<rlt::rl::environments::l2f::Specification<T, TI,  PARAMETER_FACTORY::STATIC_PARAMETERS>>;

int main(int argc, char** argv){
    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::malloc(device, rng);
    TI seed = 0;
    TI N = 1000;
    rlt::init(device, rng, seed);
    ENVIRONMENT env;
    ENVIRONMENT::Parameters params;
    rlt::init(device, env);
    auto overwrite = [](auto& parameters){
        parameters.mdp.reward = {
            false, // non-negative
            01.00, // scale
            01.50, // constant
            -100.00, // termination penalty
            01.00, // position
            00.00, // position_clip
            00.10, // orientation
            00.00, // linear_velocity
            00.00, // angular_velocity
            00.00, // linear_acceleration
            00.00, // angular_acceleration
            00.00, // action
            01.00, // d_action
            00.00, // position_error_integral
        };
        // parameters.mdp.init.max_position = 0.5;

        parameters.domain_randomization = {
            1.5, // thrust_to_weight_min;
            5.0, // thrust_to_weight_max;
            40, // torque_to_inertia_min;
            1200, // torque_to_inertia_max;
            0.02, // mass_min;
            5.00, // mass_max;
            0.1, // mass_size_deviation;
            0.03, // motor_time_constant_rising_min;
            0.10, // motor_time_constant_rising_max;
            0.03, // motor_time_constant_falling_min;
            0.30, // motor_time_constant_falling_max;
            0.005, // rotor_torque_constant_min;
            0.05, // rotor_torque_constant_max;
            0.0, // orientation_offset_angle_max;
            0.3  // disturbance_force_max;
        };
    };
    overwrite(env.parameters);
    rlt::sample_initial_parameters(device, env, params, rng);
    overwrite(params);
    std::cout << rlt::json(device, env, params) << std::endl;

    std::filesystem::path output_path = "./src/foundation_policy/dynamics_parameters/";
    for (TI set_i=0; set_i<N; ++set_i){
        overwrite(params);
        rlt::sample_initial_parameters(device, env, params, rng);
        overwrite(params);
        std::ofstream output(output_path / (std::to_string(set_i) + ".json"));
        auto params_copy = params;
        // disable domain randomization for pre_training and post_training;
        params_copy.domain_randomization = rlt::rl::environments::l2f::parameters::domain_randomization_disabled<T>;
        output << rlt::json(device, env, params_copy);
        output.close();
    }

    std::filesystem::path output_path_registry = "./src/foundation_policy/registry/";
    if (!std::filesystem::exists(output_path_registry)){
        std::cerr << "Output path does not exist: " << output_path_registry << std::endl;
        std::cerr << "CWD: " << std::filesystem::current_path() << std::endl;
        return 1;
    }
    std::vector<std::tuple<std::string, ENVIRONMENT::Parameters::Dynamics>> registry;
    auto permute_rotors_px4_to_cf = [&device, &env](const auto& dynamics){
        auto copy = dynamics;
        rlt::permute_rotors(device, env, copy, 0, 3, 1, 2);
        return copy;
    };
    registry.emplace_back("crazyflie", rlt::rl::environments::l2f::parameters::dynamics::crazyflie<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>);
    registry.emplace_back("x500", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::x500::real<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));
    registry.emplace_back("mrs", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::mrs<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));
    registry.emplace_back("fs", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::fs::base<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));
    registry.emplace_back("arpl", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::arpl<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));
    registry.emplace_back("flightmare", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::flightmare<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));
    registry.emplace_back("soft", permute_rotors_px4_to_cf(rlt::rl::environments::l2f::parameters::dynamics::soft<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>));

    rlt::initial_parameters(device, env, params);
    overwrite(params);
    for (const auto& [name, dynamics] : registry){
        params.dynamics = dynamics;
        std::ofstream output(output_path_registry / (name + ".json"));
        auto params_copy = params;
        output << rlt::json(device, env, params_copy);
        output.close();
    }

}
