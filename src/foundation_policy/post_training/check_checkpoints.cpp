#include <rl_tools/operations/cpu.h>

#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/utils/evaluation/operations_generic.h>

#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>

#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/nn_analytics/operations_cpu.h>

#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/td3_sampling/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
#include <rl_tools/rl/components/replay_buffer/persist.h>

// #include "../../../logs/2025-03-26_11-06-24/checkpoints/86/checkpoint.h"


namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#include "environment.h"
#include "../pre_training/config.h"
#include "../pre_training/options.h"
// #include "config.h"


// int test(){
//     DEVICE device;
//     RNG rng;
//     rlt::init(device, rng, 100);
//
//     // using ACTOR_TYPE = rl_tools::checkpoint::actor::TYPE::CHANGE_BATCH_SIZE<TI, 1>;
//     using FACTORY = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING, DYNAMIC_ALLOCATION>;
//     using LOOP_CORE_CONFIG = FACTORY::LOOP_CORE_CONFIG;
//     using LOOP_CONFIG = builder::LOOP_ASSEMBLY<LOOP_CORE_CONFIG>::LOOP_CONFIG;
//     using ACTOR_ORIG = LOOP_CONFIG::ACTOR_CRITIC_TYPE::SPEC::ACTOR_NETWORK_TYPE;
//     using ACTOR = ACTOR_ORIG::CHANGE_BATCH_SIZE<TI, 32>::CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;
//     ACTOR::Buffer<false> buffer;
//     // test
//     // rl_tools::checkpoint::actor::TYPE::Buffer<> test_buffer;
//     // rlt::Tensor<rlt::tensor::Specification<T, TI, rl_tools::checkpoint::example::output::SHAPE>> output;
//     // rlt::malloc(device, test_buffer);
//     // rlt::malloc(device, output);
//     // rlt::Mode<rlt::mode::Evaluation<>> mode;
//     // rlt::evaluate(device, rl_tools::checkpoint::actor::module, rl_tools::checkpoint::example::input::container, output, test_buffer, rng, mode);
//     // T abs_diff = rlt::abs_diff(device, rl_tools::checkpoint::example::output::container, output) / rl_tools::checkpoint::example::output::SPEC::SIZE;
//     // auto last_step_output = rlt::view(device, output, rlt::get<0>(rl_tools::checkpoint::example::output::SHAPE{}) - 1);
//     // auto last_step_expected = rlt::view(device, rl_tools::checkpoint::example::output::container, rlt::get<0>(rl_tools::checkpoint::example::output::SHAPE{}) - 1);
//     // std::cout << "last_step_output: " << std::endl;
//     // rlt::print(device, last_step_output);
//     // std::cout << "last_step_expected: " << std::endl;
//     // rlt::print(device, last_step_expected);
//     //
//     // std::cout << "abs_diff to checkpoint example: " << abs_diff << std::endl;
//     // rlt::free(device, test_buffer);
//     // rlt::free(device, output);
//     // return abs_diff < 1e-5 ? 0 : 1;
// }

static constexpr bool DYNAMIC_ALLOCATION = true;
static constexpr TI NUM_EPISODES_EVAL = 100;
using FACTORY = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING, DYNAMIC_ALLOCATION>;
using LOOP_CORE_CONFIG = FACTORY::LOOP_CORE_CONFIG;
using LOOP_CONFIG = builder::LOOP_ASSEMBLY<LOOP_CORE_CONFIG>::LOOP_CONFIG;
using ACTOR_ORIG = LOOP_CONFIG::ACTOR_CRITIC_TYPE::SPEC::ACTOR_NETWORK_TYPE;
using ACTOR = ACTOR_ORIG::CHANGE_BATCH_SIZE<TI, 32>::CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;
using ENVIRONMENT = LOOP_CONFIG::ENVIRONMENT;


int main(){
    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::malloc(device, rng);
    TI seed = 6;
    rlt::init(device, rng, seed);
    using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename ACTOR::template CHANGE_BATCH_SIZE<TI, NUM_EPISODES_EVAL>;
    using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>>;
    rlt::rl::environments::DummyUI ui;
    EVALUATION_ACTOR_TYPE evaluation_actor;
    EVALUATION_ACTOR_TYPE::Buffer<DYNAMIC_ALLOCATION> eval_buffer;
    rlt::malloc(device, evaluation_actor);
    rlt::malloc(device, eval_buffer);

    rlt::utils::extrack::Path checkpoint_path;
    checkpoint_path.experiment = "2025-03-31_21-06-47"; // fails
    // checkpoint_path.experiment = "2025-04-01_13-43-13"; // good
    checkpoint_path.name = "foundation-policy-pre-training";

    std::filesystem::path dynamics_parameters_path = "./src/foundation_policy/dynamics_parameters/";
    std::filesystem::path dynamics_parameter_index = "./src/foundation_policy/checkpoints_2025-03-31_21-06-47.txt";


    std::vector<ENVIRONMENT::Parameters::Dynamics> query_dynamics;

    query_dynamics.emplace_back(rlt::rl::environments::l2f::parameters::dynamics::crazy_flie<ENVIRONMENT::SPEC::T, ENVIRONMENT::SPEC::TI>);

    std::ifstream dynamics_parameter_index_file(dynamics_parameter_index);
    if (!dynamics_parameter_index_file){
        std::cerr << "Failed to open dynamics parameter index file: " << dynamics_parameter_index << std::endl;
        return 1;
    }
    std::string dynamics_parameter_index_line;
    std::vector<std::string> dynamics_parameter_index_lines;
    while (std::getline(dynamics_parameter_index_file, dynamics_parameter_index_line)){
        if (dynamics_parameter_index_line.empty()){
            continue;
        }
        dynamics_parameter_index_lines.push_back(dynamics_parameter_index_line);
    }
    dynamics_parameter_index_file.close();
    const TI num_teachers = dynamics_parameter_index_lines.size();

    for (TI teacher_i=0; teacher_i < num_teachers; ++teacher_i){
        // load actor & critic
        auto cpp_copy = checkpoint_path;
        cpp_copy.attributes["dynamics-id"] = dynamics_parameter_index_lines[dynamics_parameter_index_lines.size() - 1 - teacher_i]; // take from the end because we order by performance and the best are at the end
        rlt::find_latest_run(device, "experiments", cpp_copy);
        auto actor_file = HighFive::File(cpp_copy.checkpoint_path.string(), HighFive::File::ReadOnly);
        rlt::load(device, evaluation_actor, actor_file.getGroup("actor"));

        std::ifstream dynamics_parameter_file = std::ifstream(dynamics_parameters_path / (cpp_copy.attributes["dynamics-id"] + ".json"));
        std::string dynamics_parameter_json((std::istreambuf_iterator<char>(dynamics_parameter_file)), std::istreambuf_iterator<char>());
        dynamics_parameter_file.close();
        ENVIRONMENT env;
        rlt::from_json(device, env, dynamics_parameter_json, env.parameters);

        rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>> result;
        rlt::rl::utils::evaluation::NoData<rlt::rl::utils::evaluation::DataSpecification<decltype(result)::SPEC>> no_data;
        rlt::Mode<rlt::mode::Evaluation<>> mode;
        rlt::evaluate(device, env, ui, evaluation_actor, result, no_data, rng, mode);
        std::cout << "Teacher policy " << cpp_copy.checkpoint_path.string() << " mean return: " << result.returns_mean << " episode length: " << result.episode_length_mean << " share terminated: " << result.share_terminated << std::endl;
    }
    rlt::free(device, evaluation_actor);
    rlt::free(device, eval_buffer);
}
