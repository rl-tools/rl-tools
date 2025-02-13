#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/multi_agent_wrapper/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

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

#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/gru/persist_code.h>
#include <rl_tools/nn/layers/sample_and_squash/persist_code.h>
#include <rl_tools/nn/layers/td3_sampling/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist_code.h>

#include "../pre_training/environment.h"

#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>

#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/nn_analytics/operations_cpu.h>

#include <rl_tools/rl/utils/evaluation/operations_cpu.h>

#include "../pre_training/config.h"

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_PARAMS_DEVICE = rlt::devices::random::Generic<DEVICE::SPEC::MATH>;
using RNG_PARAMS = RNG_PARAMS_DEVICE::ENGINE<>;
using TI = typename DEVICE::index_t;
using T = float;
constexpr bool DYNAMIC_ALLOCATION = true;

struct OPTIONS_PRE_TRAINING{
    static constexpr bool SEQUENTIAL_MODEL = false;
    static constexpr bool MOTOR_DELAY = false;
    static constexpr bool RANDOMIZE_MOTOR_MAPPING = true;
    static constexpr bool RANDOMIZE_THRUST_CURVES = false;
    static constexpr bool OBSERVE_THRASH_MARKOV = false;
    static constexpr bool SAMPLE_INITIAL_PARAMETERS = false;
};
struct OPTIONS_POST_TRAINING: OPTIONS_PRE_TRAINING{
};
using LOOP_CORE_CONFIG_PRE_TRAINING = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING, DYNAMIC_ALLOCATION>::LOOP_CORE_CONFIG;
using LOOP_CORE_CONFIG_POST_TRAINING = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING, DYNAMIC_ALLOCATION>::LOOP_CORE_CONFIG;
struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
    static constexpr T ALPHA = 0.001;
};



// note: make sure that the rng_params is invoked in the exact same way in pre- as in post-training, to make sure the params used to sample parameters to generate data from the trained policy are matching the ones seen by the particular policy for the seed during pretraining

int main(int argc, char** argv){
    // std::vector<std::filesystem::path> checkpoint_paths;
    // for (TI seed = 0; seed < 50; seed++){
    //     std::stringstream ss;
    //     ss << "experiments/2025-02-11_17-03-35/f98ad54_default_default/default/" << std::setfill('0') << std::setw(4) << seed << "/steps/000000002000000/checkpoint.h5";
    //     checkpoint_paths.push_back(ss.str());
    // }

    DEVICE device;
    RNG rng;
    RNG_PARAMS rng_params;
    rlt::init(device);

    TI seed = argc >= 2 ? std::stoi(argv[1]) : 0;

    rlt::utils::extrack::Path checkpoint_path;
    checkpoint_path.experiment = "2025-02-13_11-30-07";
    checkpoint_path.name = "foundation-policy-pre-training";
    checkpoint_path.seed = std::to_string(seed);
    rlt::find_latest_run(device, "experiments", checkpoint_path);
    if (!rlt::find_latest_checkpoint(device, checkpoint_path)){
        std::cerr << "No checkpoint found for " << checkpoint_path.experiment << std::endl;
        return 1;
    }
    rlt::malloc(device, rng_params);
    rlt::init(device, rng_params, seed);
    // warmup
    for(TI i=0; i < (TI)std::stoi(checkpoint_path.attributes["rng-warmup"]); i++){
        rlt::random::uniform_int_distribution(RNG_PARAMS_DEVICE{}, 0, 1, rng_params);
    }


    std::cout << "Found checkpoint: " << checkpoint_path.checkpoint_path << std::endl;

#ifdef RL_TOOLS_ENABLE_TENSORBOARD
    auto timestamp_string = rlt::utils::extrack::get_timestamp_string();
    rlt::init(device, device.logger, "logs/" + timestamp_string);
#endif
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    // typename LOOP_CORE_CONFIG_PRE_TRAINING::template State <LOOP_CORE_CONFIG_PRE_TRAINING> ts;
    // rlt::malloc(device, ts);
    // rlt::init(device, ts, seed);
    LOOP_CORE_CONFIG_PRE_TRAINING::ENVIRONMENT base_env;
    rlt::sample_initial_parameters<DEVICE, LOOP_CORE_CONFIG_PRE_TRAINING::ENVIRONMENT::SPEC, RNG_PARAMS, true>(device, base_env, base_env.parameters, rng_params);

    rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<T, TI, ADAM_PARAMETERS>> actor_optimizer;
    rlt::malloc(device, actor_optimizer);

    LOOP_CORE_CONFIG_POST_TRAINING::NN::ACTOR_TYPE actor;
    decltype(actor)::Buffer<> actor_buffer;
    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffer);
    rlt::init_weights(device, actor, rng);

    auto actor_file = HighFive::File(checkpoint_path.checkpoint_path.string(), HighFive::File::ReadOnly);

    constexpr TI NUM_EPISODES = 100;
    rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, LOOP_CORE_CONFIG_PRE_TRAINING::ENVIRONMENT, NUM_EPISODES, LOOP_CORE_CONFIG_PRE_TRAINING::CORE_PARAMETERS::EPISODE_STEP_LIMIT>> result;
    auto* data = new rlt::rl::utils::evaluation::Data<decltype(result)::SPEC>;
    using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename LOOP_CORE_CONFIG_PRE_TRAINING::NN::ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, NUM_EPISODES>;
    using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<LOOP_CORE_CONFIG_PRE_TRAINING::DYNAMIC_ALLOCATION>>;
    rlt::rl::environments::DummyUI ui;
    EVALUATION_ACTOR_TYPE evaluation_actor;
    EVALUATION_ACTOR_TYPE::Buffer<LOOP_CORE_CONFIG_PRE_TRAINING::DYNAMIC_ALLOCATION> eval_buffer;
    rlt::malloc(device, evaluation_actor);
    rlt::malloc(device, eval_buffer);
    rlt::load(device, evaluation_actor, actor_file.getGroup("actor"));

    typename LOOP_CORE_CONFIG_PRE_TRAINING::ENVIRONMENT_EVALUATION env_eval;
    typename LOOP_CORE_CONFIG_PRE_TRAINING::ENVIRONMENT_EVALUATION::Parameters env_eval_parameters;
    rlt::init(device, env_eval);
    env_eval.parameters = base_env.parameters;
    rlt::initial_parameters(device, env_eval, env_eval_parameters);

    rlt::init(device, rng, seed);
    rlt::Mode<rlt::mode::Evaluation<rlt::nn::layers::sample_and_squash::mode::DisableEntropy<rlt::mode::Final>>> evaluation_mode;
    rlt::evaluate(device, env_eval, env_eval_parameters, ui, evaluation_actor, result, *data, eval_buffer, rng, evaluation_mode, false, true);
    rlt::log(device, device.logger, "Checkpoint ", checkpoint_path.checkpoint_path.string(), ": Mean return: ", result.returns_mean, " Mean episode length: ", result.episode_length_mean);

    using INPUT_SHAPE = decltype(actor)::INPUT_SHAPE;
    using OUTPUT_SHAPE = decltype(actor)::OUTPUT_SHAPE;
    // using INPUT_SHAPE = decltype(evaluation_actor)::INPUT_SHAPE;
    // using OUTPUT_SHAPE = decltype(evaluation_actor)::OUTPUT_SHAPE;
    using INPUT_SHAPE_DATASET = rlt::tensor::Replace<INPUT_SHAPE, NUM_EPISODES * LOOP_CORE_CONFIG_PRE_TRAINING::CORE_PARAMETERS::EPISODE_STEP_LIMIT, 1>;
    using OUTPUT_SHAPE_DATASET = rlt::tensor::Replace<OUTPUT_SHAPE, NUM_EPISODES * LOOP_CORE_CONFIG_PRE_TRAINING::CORE_PARAMETERS::EPISODE_STEP_LIMIT, 1>;

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_DATASET>> dataset_input_3d;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_DATASET>> dataset_output_target_3d;
    rlt::malloc(device, dataset_input_3d);
    rlt::malloc(device, dataset_output_target_3d);
    auto dataset_input = rlt::view(device, dataset_input_3d, 0);
    auto dataset_output_target = rlt::view(device, dataset_output_target_3d, 0);

    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_output;
    rlt::malloc(device, d_output);
    TI current_index = 0;
    for (TI episode_i = 0; episode_i < NUM_EPISODES; episode_i++){
        TI current_step_i;
        for (current_step_i = 0; current_step_i < LOOP_CORE_CONFIG_PRE_TRAINING::CORE_PARAMETERS::EPISODE_STEP_LIMIT; current_step_i++){
            auto observation_tensor = rlt::view(device, dataset_input, current_index + current_step_i);
            auto action_tensor = rlt::view(device, dataset_output_target, current_index + current_step_i);
            auto observation = rlt::matrix_view(device, observation_tensor);
            auto action = rlt::matrix_view(device, action_tensor);
            rlt::observe(device, env_eval, env_eval_parameters, data->states[episode_i][current_step_i], LOOP_CORE_CONFIG_POST_TRAINING::ENVIRONMENT::Observation{}, observation, rng);
            for (TI action_i=0; action_i < OUTPUT_SHAPE::LAST; action_i++){
                rlt::set(action, 0, action_i, data->actions[episode_i][current_step_i][action_i]);
            }
            if (data->terminated[episode_i][current_step_i]){
                break;
            }
        }
        current_index += current_step_i;
    }
    TI N = current_index;
    rlt::reset_optimizer_state(device, actor_optimizer, actor);
    for (TI epoch_i = 0; epoch_i < 100; epoch_i++){
        for (TI sample_i=0; sample_i<N; sample_i++){
            TI target_index = rlt::random::uniform_int_distribution(device.random, sample_i, N - 1, rng);
            auto input = rlt::view(device, dataset_input, sample_i);
            auto input_target = rlt::view(device, dataset_input, target_index);

            rlt::Tensor<rlt::tensor::Specification<T, TI, decltype(input)::SPEC::SHAPE, false>> input_target_temp;
            rlt::copy(device, device, input_target, input_target_temp);
            rlt::copy(device, device, input, input_target);
            rlt::copy(device, device, input_target_temp, input);

            auto output_target = rlt::view(device, dataset_output_target, sample_i);
            rlt::Tensor<rlt::tensor::Specification<T, TI, decltype(output_target)::SPEC::SHAPE, false>> output_target_temp;
            auto output_target_target = rlt::view(device, dataset_output_target, target_index);
            rlt::copy(device, device, output_target_target, output_target_temp);
            rlt::copy(device, device, output_target, output_target_target);
            rlt::copy(device, device, output_target_temp, output_target);
        }
        constexpr TI BATCH_SIZE = INPUT_SHAPE::GET<1>;
        for (TI batch_i = 0; batch_i < N / BATCH_SIZE; batch_i++){
            auto input = rlt::view_range(device, dataset_input_3d, batch_i * BATCH_SIZE, rlt::tensor::ViewSpec<1, BATCH_SIZE>{});
            auto output_target = rlt::view_range(device, dataset_output_target_3d, batch_i * BATCH_SIZE, rlt::tensor::ViewSpec<1, BATCH_SIZE>{});
            rlt::forward(device, actor, input, actor_buffer, rng, evaluation_mode);
            // rlt::evaluate(device, evaluation_actor, input, output, eval_buffer, rng, evaluation_mode);
            auto output_matrix_view = rlt::matrix_view(device, rlt::output(device, actor));
            auto output_target_matrix_view = rlt::matrix_view(device, output_target);
            auto d_output_matrix_view = rlt::matrix_view(device, d_output);
            rlt::nn::loss_functions::mse::gradient(device, output_matrix_view, output_target_matrix_view, d_output_matrix_view);
            T loss = rlt::nn::loss_functions::mse::evaluate(device, output_matrix_view, output_target_matrix_view);
            rlt::set_step(device, device.logger, epoch_i * (N/BATCH_SIZE) + batch_i);
            rlt::add_scalar(device, device.logger, "loss", loss);
            std::cout << "Epoch: " << epoch_i << " Loss: " << loss << std::endl;
            rlt::zero_gradient(device, actor);
            rlt::backward(device, actor, input, d_output, actor_buffer, evaluation_mode);
            rlt::step(device, actor_optimizer, actor);
        }
        // {
        //     auto output_target = rlt::view_range(device, dataset_output_target_3d, (N / BATCH_SIZE - 1) * BATCH_SIZE, rlt::tensor::ViewSpec<1, BATCH_SIZE>{});
        //     std::cout << "Target: " << std::endl;
        //     rlt::print(device, output_target);
        //     std::cout << "Output: " << std::endl;
        //     rlt::print(device, rlt::output(device, ts.actor_critic.actor));
        //     std::cout << "Diff: " << std::endl;
        //     std::cout << rlt::abs_diff(device, rlt::output(device, ts.actor_critic.actor), output_target) << std::endl;;
        // }
        {
            using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename LOOP_CORE_CONFIG_PRE_TRAINING::NN::ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, NUM_EPISODES>;
            using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<LOOP_CORE_CONFIG_PRE_TRAINING::DYNAMIC_ALLOCATION>>;
            rlt::rl::environments::DummyUI ui;
            EVALUATION_ACTOR_TYPE evaluation_actor;
            EVALUATION_ACTOR_TYPE::Buffer<LOOP_CORE_CONFIG_PRE_TRAINING::DYNAMIC_ALLOCATION> eval_buffer;
            rlt::malloc(device, evaluation_actor);
            rlt::malloc(device, eval_buffer);
            rlt::copy(device, device, actor, evaluation_actor);

            rlt::evaluate(device, env_eval, env_eval_parameters, ui, evaluation_actor, result, *data, eval_buffer, rng, evaluation_mode, false, true);
            rlt::add_scalar(device, device.logger, "evaluation/return/mean", result.returns_mean);
            rlt::add_scalar(device, device.logger, "evaluation/return/std", result.returns_std);
            rlt::add_scalar(device, device.logger, "evaluation/episode_length/mean", result.episode_length_mean);
            rlt::add_scalar(device, device.logger, "evaluation/episode_length/std", result.episode_length_std);
            rlt::log(device, device.logger, "Checkpoint ", checkpoint_path.checkpoint_path.string(), ": Mean return: ", result.returns_mean, " Mean episode length: ", result.episode_length_mean);
            rlt::free(device, evaluation_actor);
            rlt::free(device, eval_buffer);
        }
    }
    delete data;
    rlt::free(device, evaluation_actor);
    rlt::free(device, device.logger);
    rlt::free(device);
    return 0;
}
