#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>

#include <fstream>
namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = DEVICE::index_t;
using T = float;
constexpr bool DYNAMIC_ALLOCATION = true;


using PARAMETER_FACTORY = rlt::rl::environments::l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI, rlt::rl::environments::l2f::parameters::DEFAULT_DOMAIN_RANDOMIZATION_OPTIONS<true>>;
using ENVIRONMENT = rlt::rl::environments::Multirotor<rlt::rl::environments::l2f::Specification<T, TI,  PARAMETER_FACTORY::STATIC_PARAMETERS>>;

int main(int argc, char** argv){
    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::malloc(device, rng);
    TI seed = 0;
    TI N = 100;
    rlt::init(device, rng, seed);
    ENVIRONMENT env;
    ENVIRONMENT::Parameters params;
    rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    std::cout << rlt::json(device, env, params) << std::endl;

    env.parameters.domain_randomization = {
        1.5, // thrust_to_weight_min;
        5.0, // thrust_to_weight_max;
        0.001, // thrust_to_weight_by_torque_to_inertia_min;
        0.100, // thrust_to_weight_by_torque_to_inertia_max;
        0.02, // mass_min;
        5.00, // mass_max;
        0.1, // mass_size_deviation;
        0.0, // motor_time_constant;
        0.0, // rotor_torque_constant;
        0.0  // orientation_offset_angle_max;
    };
    std::filesystem::path output_path = "./src/foundation_policy/dynamics_parameters/";
    if (!std::filesystem::exists(output_path)){
        std::cerr << "Output path does not exist: " << output_path << std::endl;
        std::cerr << "CWD: " << std::filesystem::current_path() << std::endl;
        return 1;
    }
    for (TI set_i=0; set_i<N; ++set_i){
        rlt::sample_initial_parameters(device, env, params, rng);
        std::ofstream output(output_path / (std::to_string(set_i) + ".json"));
        output << rlt::json(device, env, params);
        output.close();
    }
}