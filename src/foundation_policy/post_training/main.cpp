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

int main(int argc, char** argv){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    TI seed = argc >= 2 ? std::stoi(argv[1]) : 0;
    rlt::init(device, rng, seed);
    typename LOOP_CORE_CONFIG_PRE_TRAINING::template State <LOOP_CORE_CONFIG_PRE_TRAINING> ts;
    rlt::malloc(device, ts);
    rlt::init(device, ts, seed);
    auto& base_env = rlt::get(ts.off_policy_runner.envs, 0, 0);
    rlt::sample_initial_parameters<DEVICE, LOOP_CORE_CONFIG_PRE_TRAINING::ENVIRONMENT::SPEC, RNG, false>(device, base_env, base_env.parameters, rng);
    for (TI env_i = 1; env_i < LOOP_CORE_CONFIG_PRE_TRAINING::CORE_PARAMETERS::N_ENVIRONMENTS; env_i++){
        auto& env = rlt::get(ts.off_policy_runner.envs, 0, env_i);
        env.parameters = base_env.parameters;
    }

    std::filesystem::path checkpoint_path = "experiments/2025-02-11_17-03-35/f98ad54_default_default/default/0000/steps/000000002000000/checkpoint.h5";
    auto actor_file = HighFive::File(checkpoint_path.string(), HighFive::File::ReadOnly);

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
    rlt::initial_parameters(device, env_eval, env_eval_parameters);

    rlt::init(device, rng, seed);
    rlt::Mode<rlt::mode::Evaluation<>> evaluation_mode;
    rlt::evaluate(device, env_eval, env_eval_parameters, ui, evaluation_actor, result, *data, eval_buffer, rng, evaluation_mode, false, true);
    rlt::log(device, device.logger, "Checkpoint ", checkpoint_path.string(), ": Mean return: ", result.returns_mean, " Mean episode length: ", result.episode_length_mean);

    // using INPUT_SHAPE = decltype(ts.actor_critic.actor)::INPUT_SHAPE;
    // using OUTPUT_SHAPE = decltype(ts.actor_critic.actor)::OUTPUT_SHAPE;
    using INPUT_SHAPE = decltype(evaluation_actor)::INPUT_SHAPE;
    using OUTPUT_SHAPE = decltype(evaluation_actor)::OUTPUT_SHAPE;
    using INPUT_SHAPE_DATASET = rlt::tensor::Replace<INPUT_SHAPE, NUM_EPISODES * LOOP_CORE_CONFIG_PRE_TRAINING::CORE_PARAMETERS::EPISODE_STEP_LIMIT, 1>;
    using OUTPUT_SHAPE_DATASET = rlt::tensor::Replace<OUTPUT_SHAPE, NUM_EPISODES * LOOP_CORE_CONFIG_PRE_TRAINING::CORE_PARAMETERS::EPISODE_STEP_LIMIT, 1>;

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_DATASET>> dataset_input_3d;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_DATASET>> dataset_output_target_3d;
    rlt::malloc(device, dataset_input_3d);
    rlt::malloc(device, dataset_output_target_3d);
    auto dataset_input = rlt::view(device, dataset_input_3d, 0);
    auto dataset_output_target = rlt::view(device, dataset_output_target_3d, 0);

    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output;
    rlt::malloc(device, output);
    TI current_index = 0;
    for (TI episode_i = 0; episode_i < NUM_EPISODES; episode_i++){
        TI current_step_i;
        for (current_step_i = 0; current_step_i < LOOP_CORE_CONFIG_PRE_TRAINING::CORE_PARAMETERS::EPISODE_STEP_LIMIT; current_step_i++){
            auto observation_tensor = rlt::view(device, dataset_input, current_index + current_step_i);
            auto action_tensor = rlt::view(device, dataset_output_target, current_index + current_step_i);
            auto observation = rlt::matrix_view(device, observation_tensor);
            auto action = rlt::matrix_view(device, action_tensor);
            rlt::observe(device, env_eval, env_eval_parameters, data->states[episode_i][current_step_i], decltype(env_eval)::Observation{}, observation, rng);
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
    for (TI epoch_i = 0; epoch_i < 10; epoch_i++){
        for (TI sample_i=0; sample_i<N; sample_i++){
            TI target_index = rlt::random::uniform_int_distribution(device.random, sample_i, N - 1, rng);
            auto input = rlt::view(device, dataset_input, sample_i);
            rlt::Tensor<rlt::tensor::Specification<T, TI, decltype(input)::SPEC::SHAPE, false>> input_target_temp;
            auto input_target = rlt::view(device, dataset_input, target_index);
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
            // rlt::evaluate(device, ts.actor_critic.actor, input, output, ts.actor_buffers[0], rng);
            rlt::evaluate(device, evaluation_actor, input, output, eval_buffer, rng, evaluation_mode);
            T diff = rlt::abs_diff(device, output, output_target);
            rlt::log(device, device.logger, "Diff: ", diff);
        }
    }
    delete data;
    rlt::free(device, evaluation_actor);
    return 0;
}
