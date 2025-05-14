#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/multi_agent_wrapper/operations_generic.h>

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

#include "../post_training/environment.h"

#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>

#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/nn_analytics/operations_cpu.h>

#include <rl_tools/rl/utils/evaluation/operations_cpu.h>

#include "../pre_training/config.h"
#include "../pre_training/options.h"
namespace rlt = rl_tools;

#include "../post_training/helper.h"
#include "../blob/checkpoint.h"


using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_PARAMS_DEVICE = rlt::devices::random::Generic<DEVICE::SPEC::MATH>;
using RNG_PARAMS = RNG_PARAMS_DEVICE::ENGINE<>;
using TI = DEVICE::index_t;
using T = float;

#define RL_TOOLS_POST_TRAINING
#include "../post_training/config.h"


// note: make sure that the rng_params is invoked in the exact same way in pre- as in post-training, to make sure the params used to sample parameters to generate data from the trained policy are matching the ones seen by the particular policy for the seed during pretraining

// using EVAL_ACTOR = ACTOR::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;
using EVAL_ACTOR = rlt::checkpoint::actor::TYPE;

int main(int argc, char** argv){
    // declarations
    DEVICE device;
    RNG rng;
    const EVAL_ACTOR& actor = rlt::checkpoint::actor::module;
    EVAL_ACTOR::Buffer<> actor_buffer;
    std::cout << "Input shape: " << std::endl;
    rlt::print(device, EVAL_ACTOR::INPUT_SHAPE{});

    // device init
    rlt::init(device);

    // malloc
    rlt::malloc(device, rng);
    // rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffer);

    // init
    TI seed = argc >= 2 ? std::stoi(argv[1]) : 0;

    // std::string checkpoint_path = "./src/foundation_policy/blob/checkpoint.h5";
    std::string experiment = "2025-04-16_20-10-58";

    rlt::init(device, rng, seed);

    // auto file = HighFive::File(checkpoint_path, HighFive::File::ReadOnly);
    // rlt::load(device, actor, file.getGroup("actor"));

    std::filesystem::path dynamics_parameters_path = "./src/foundation_policy/dynamics_parameters_" + experiment + "/";
    std::filesystem::path dynamics_parameter_index = "./src/foundation_policy/checkpoints_" + experiment + ".txt";

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

    nlohmann::json hidden_states = nlohmann::json::array();
    for (TI drone_i=0; drone_i < dynamics_parameter_index_lines.size(); ++drone_i){
        // load actor & critic
        auto checkpoint_info = dynamics_parameter_index_lines[drone_i];
        auto checkpoint_info_split = split_by_comma(checkpoint_info);
        std::string dynamics_id = checkpoint_info_split[0];
        std::ifstream dynamics_parameter_file = std::ifstream(dynamics_parameters_path / (dynamics_id + ".json"));
        std::string dynamics_parameter_json((std::istreambuf_iterator<char>(dynamics_parameter_file)), std::istreambuf_iterator<char>());
        dynamics_parameter_file.close();
        ENVIRONMENT env;
        rlt::init(device, env);
        auto init_old = env.parameters.mdp.init; // legacy for v3, remove for next run
        rlt::from_json(device, env, dynamics_parameter_json, env.parameters);
        env.parameters.mdp.init = init_old;
        rlt::sample_initial_parameters(device, env, env.parameters, rng);

        using EVAL_SPEC = rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>;
        rlt::rl::utils::evaluation::Result<EVAL_SPEC> result;
        rlt::rl::utils::evaluation::Data<rlt::rl::utils::evaluation::DataSpecification<EVAL_SPEC>> data;
        RNG rng_copy = rng;
        rlt::malloc(device, data);
        rlt::Mode<rlt::mode::Evaluation<rlt::nn::layers::sample_and_squash::mode::DisableEntropy<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Final>>>> mode;
        rlt::rl::environments::DummyUI ui;
        {
            static constexpr bool DYNAMIC_ALLOCATION = true;
            using ADJUSTED_POLICY = typename EVAL_ACTOR::template CHANGE_BATCH_SIZE<TI, EVAL_SPEC::N_EPISODES>;
            typename ADJUSTED_POLICY::template State<DYNAMIC_ALLOCATION> policy_state;
            typename ADJUSTED_POLICY::template Buffer<DYNAMIC_ALLOCATION> policy_evaluation_buffers;
            rlt::rl::utils::evaluation::Buffer<rlt::rl::utils::evaluation::BufferSpecification<EVAL_SPEC, DYNAMIC_ALLOCATION>> evaluation_buffers;
            rlt::malloc(device, policy_state);
            rlt::malloc(device, policy_evaluation_buffers);
            rlt::malloc(device, evaluation_buffers);
            rlt::evaluate(device, env, ui, actor, policy_state, policy_evaluation_buffers, evaluation_buffers, result, data, rng, mode);
            for (TI episode_i = 0; episode_i < EVAL_SPEC::N_EPISODES; ++episode_i) {
                if (result.episode_length[episode_i] == EVAL_SPEC::STEP_LIMIT) {
                    auto hidden_state_view = rlt::view(device, policy_state.content_state.next_content_state.state.state, episode_i);
                    std::array<T, HIDDEN_DIM> hidden_state;
                    for (TI dim_i = 0; dim_i < HIDDEN_DIM; ++dim_i){
                        hidden_state[dim_i] = rlt::get(device, hidden_state_view, dim_i);
                    }
                    nlohmann::json data;
                    data["hidden_state"] = hidden_state;
                    data["return"] = result.returns[episode_i];
                    data["dynamics_id"] = dynamics_id;
                    hidden_states.push_back(data);
                }
            }
            rlt::free(device, policy_state);
            rlt::free(device, policy_evaluation_buffers);
            rlt::free(device, evaluation_buffers);
        }
        rlt::free(device, data);
        std::cout << "Teacher policy mean return: " << result.returns_mean << " episode length: " << result.episode_length_mean << " share terminated: " << result.share_terminated << std::endl;
    }
    std::ofstream o("./src/foundation_policy/hidden_states_" + experiment + ".json");
    o << hidden_states.dump();
    o.close();


    // malloc
    rlt::free(device, rng);
    // rlt::free(device, actor);
    rlt::free(device, actor_buffer);
    return 0;
}
