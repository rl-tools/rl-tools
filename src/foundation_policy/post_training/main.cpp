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

#include "environment.h"

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

#include "generate_data.h"


using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_PARAMS_DEVICE = rlt::devices::random::Generic<DEVICE::SPEC::MATH>;
using RNG_PARAMS = RNG_PARAMS_DEVICE::ENGINE<>;
using TI = DEVICE::index_t;
using T = float;
constexpr bool DYNAMIC_ALLOCATION = true;

struct OPTIONS_POST_TRAINING: OPTIONS_PRE_TRAINING{
    static constexpr bool OBSERVE_THRASH_MARKOV = false;
    static constexpr bool MOTOR_DELAY = true;
    static constexpr bool ACTION_HISTORY = true;
    static constexpr TI ACTION_HISTORY_LENGTH = 32;
    static constexpr bool OBSERVATION_NOISE = true;
};

struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
    static constexpr T ALPHA = 0.0001;
};
// constants parameters
constexpr TI NUM_EPISODES = 1000;
constexpr TI N_EPOCH = 100;
constexpr TI N_PRE_TRAINING_SEEDS = 1;
constexpr TI SEQUENCE_LENGTH = 1;
constexpr TI BATCH_SIZE = 32;
constexpr T SOLVED_RETURN = 500;
constexpr TI HIDDEN_DIM = 32;

// typedefs
using ENVIRONMENT = typename builder::ENVIRONMENT_FACTORY_POST_TRAINING<DEVICE, T, TI, OPTIONS_POST_TRAINING>::ENVIRONMENT;
struct ENVIRONMENT_PT: ENVIRONMENT{
    using LOOP_CORE_CONFIG_PRE_TRAINING = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING, DYNAMIC_ALLOCATION>::LOOP_CORE_CONFIG;
    using ENV = LOOP_CORE_CONFIG_PRE_TRAINING::ENVIRONMENT;
    using Observation = ENV::Observation;
    using ObservationPrivileged = Observation;
};

using MLP_CONFIG = rlt::nn_models::mlp::Configuration<T, TI, ENVIRONMENT::ACTION_DIM, 3, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::FAST_TANH, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
using MLP = rlt::nn_models::mlp::BindConfiguration<MLP_CONFIG>;

// using INPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
// using INPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
// using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, HIDDEN_DIM>;
// using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;

using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential::OutputModule>
using Module = typename rlt::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
using MODULE_CHAIN = Module<MLP>;
using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, DYNAMIC_ALLOCATION>;
using ACTOR = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
using EVAL_MODE = rlt::Mode<rlt::mode::Default<>>;
using OPTIMIZER = rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<T, TI, ADAM_PARAMETERS>>;
using OUTPUT_SHAPE = ACTOR::OUTPUT_SHAPE;
using RESULT = rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES, ENVIRONMENT::EPISODE_STEP_LIMIT>>;
using DATA = rlt::rl::utils::evaluation::Data<RESULT::SPEC>;

// constants derived
constexpr TI DATASET_SIZE = N_PRE_TRAINING_SEEDS * NUM_EPISODES * ENVIRONMENT::EPISODE_STEP_LIMIT;

template <typename DATA, typename INPUT_SPEC, typename OUTPUT_SPEC, typename TRUNCATED_SPEC, typename PARAMS, typename RNG>
TI add_to_dataset(DEVICE& device, DATA& data, rlt::Tensor<INPUT_SPEC>& input, rlt::Tensor<OUTPUT_SPEC>& output, rlt::Tensor<TRUNCATED_SPEC>& truncated, TI& current_index, PARAMS& base_parameters, RNG& rng){

    TI initial_index = current_index;
    ENVIRONMENT env_eval;
    ENVIRONMENT::Parameters env_eval_parameters;
    rlt::init(device, env_eval);
    env_eval.parameters = base_parameters;
    rlt::initial_parameters(device, env_eval, env_eval_parameters);

    for (TI episode_i = 0; episode_i < DATA::SPEC::N_EPISODES; episode_i++){
        TI current_step_i;
        for (current_step_i = 0; current_step_i < ENVIRONMENT::EPISODE_STEP_LIMIT; current_step_i++){
            auto observation_tensor = rlt::view(device, input, current_index + current_step_i);
            auto action_tensor = rlt::view(device, output, current_index + current_step_i);
            auto observation = rlt::matrix_view(device, observation_tensor);
            auto action = rlt::matrix_view(device, action_tensor);
            rlt::observe(device, env_eval, env_eval_parameters, data.states[episode_i][current_step_i], ENVIRONMENT::Observation{}, observation, rng);
            for (TI action_i=0; action_i < OUTPUT_SPEC::SHAPE::LAST; action_i++){
                rlt::set(action, 0, action_i, data.actions[episode_i][current_step_i][action_i]);
            }
            if (data.terminated[episode_i][current_step_i] || current_step_i == ENVIRONMENT::EPISODE_STEP_LIMIT - 1){

            }
            if (data.terminated[episode_i][current_step_i]){
                break;
            }
        }
        current_index += current_step_i;
        if (current_index >= INPUT_SPEC::SHAPE::FIRST){
            return current_index - initial_index;
        }
    }
    return current_index - initial_index;
}



// note: make sure that the rng_params is invoked in the exact same way in pre- as in post-training, to make sure the params used to sample parameters to generate data from the trained policy are matching the ones seen by the particular policy for the seed during pretraining

int main(int argc, char** argv){
    // declarations
    DEVICE device;
    RNG rng;
    ACTOR actor, best_actor;
    ACTOR::Buffer<> actor_buffer;
    EVAL_MODE evaluation_mode;
    OPTIMIZER actor_optimizer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DATASET_SIZE, ENVIRONMENT::Observation::DIM>>> dataset_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DATASET_SIZE, ENVIRONMENT::ACTION_DIM>>> dataset_output_target;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>> dataset_truncated;
    rlt::Tensor<rlt::tensor::Specification<TI, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>> epoch_indices;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::Observation::DIM>>> batch_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::ACTION_DIM>>> batch_output_target;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_output;
    RESULT* result_memory;
    DATA* data_memory;

    // device init
    rlt::init(device);

    // malloc
    rlt::malloc(device, rng);
    rlt::malloc(device, actor_optimizer);
    rlt::malloc(device, actor);
    rlt::malloc(device, best_actor);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, dataset_input);
    rlt::malloc(device, dataset_output_target);
    rlt::malloc(device, dataset_truncated);
    rlt::malloc(device, epoch_indices);
    rlt::malloc(device, batch_input);
    rlt::malloc(device, batch_output_target);
    rlt::malloc(device, d_output);
    result_memory = new RESULT;
    data_memory = new DATA;
    RESULT& result = *result_memory;
    DATA& data = *data_memory;

    // init
    TI seed = argc >= 2 ? std::stoi(argv[1]) : 0;
    TI current_index = 0;

    T best_return = 0;
    bool best_return_set = false;

#ifdef RL_TOOLS_ENABLE_TENSORBOARD
    auto timestamp_string = rlt::utils::extrack::get_timestamp_string();
    std::filesystem::path run_path = "logs/" + timestamp_string;
    rlt::init(device, device.logger, run_path.string());
#endif
    rlt::init(device, rng, seed);
    rlt::init_weights(device, actor, rng);

    //work
    rlt::utils::extrack::Path checkpoint_path;
    checkpoint_path.experiment = "2025-03-12_13-28-08";
    checkpoint_path.name = "foundation-policy-pre-training";

    for (TI seed_i = 0; seed_i < N_PRE_TRAINING_SEEDS; seed_i++){
        checkpoint_path.seed = std::to_string(seed_i);
        rlt::find_latest_run(device, "experiments", checkpoint_path);
        if (!rlt::find_latest_checkpoint(device, checkpoint_path)){
            std::cerr << "No checkpoint found for " << checkpoint_path.experiment << std::endl;
            return 1;
        }
        std::cout << "Found checkpoint: " << checkpoint_path.checkpoint_path << std::endl;
        auto base_parameters = generate_data<DEVICE, RNG_PARAMS, RNG, RNG_PARAMS_DEVICE, ENVIRONMENT_PT, T, TI, NUM_EPISODES>(device, checkpoint_path, seed_i, result, data);
        base_parameters = ENVIRONMENT::SPEC::STATIC_PARAMETERS::PARAMETER_VALUES;
        if (result.returns_mean < SOLVED_RETURN){
            std::cerr << "Mean return (" << result.returns_mean << ") too low for " << checkpoint_path.checkpoint_path << std::endl;
            return 1;
        }
        rlt::log(device, device.logger, "Checkpoint ", checkpoint_path.checkpoint_path.string(), ": Mean return: ", result.returns_mean, " Mean episode length: ", result.episode_length_mean, " Share terminated: ", result.share_terminated);
        TI num_added = add_to_dataset(device, data, dataset_input, dataset_output_target, dataset_truncated, current_index, base_parameters, rng);
        if (num_added == 0){
            std::cout << "Dataset full after " << seed_i << " seeds" << std::endl;
            break;
        }
    }


    TI N = current_index;
    for (TI i=0; i < N; i++){
        rlt::set(device, epoch_indices, i, i);
    }
    rlt::reset_optimizer_state(device, actor_optimizer, actor);
    for (TI epoch_i = 0; epoch_i < N_EPOCH; epoch_i++){
        for (TI sample_i=0; sample_i<N; sample_i++){
            TI target_index = rlt::random::uniform_int_distribution(device.random, sample_i, N - 1, rng);
            TI target_value = rlt::get(device, epoch_indices, target_index);
            rlt::set(device, epoch_indices, rlt::get(device, epoch_indices, sample_i), target_index);
            rlt::set(device, epoch_indices, target_value, sample_i);
        }
        constexpr TI BATCH_SIZE = INPUT_SHAPE::GET<1>;
        T epoch_loss = 0;
        TI epoch_loss_count = 0;
        for (TI batch_i = 0; batch_i < N / BATCH_SIZE; batch_i++){
            for (TI sample_i=0; sample_i<BATCH_SIZE; sample_i++){
                TI current_epoch_index = batch_i * BATCH_SIZE + sample_i;
                TI current_sample = rlt::get(device, epoch_indices, current_epoch_index);
                auto input = rlt::view(device, dataset_input, current_sample);
                auto input_target_step = rlt::view(device, batch_input, 0);
                auto input_target = rlt::view(device, input_target_step, sample_i);
                rlt::copy(device, device, input, input_target);
                auto output_target = rlt::view(device, dataset_output_target, current_sample);
                auto output_target_step = rlt::view(device, batch_output_target, 0);
                auto output_target_target = rlt::view(device, output_target_step, sample_i);
                rlt::copy(device, device, output_target, output_target_target);
            }

            rlt::forward(device, actor, batch_input, actor_buffer, rng, evaluation_mode);
            // rlt::evaluate(device, evaluation_actor, input, output, eval_buffer, rng, evaluation_mode);
            auto output_matrix_view = rlt::matrix_view(device, rlt::output(device, actor));
            auto output_target_matrix_view = rlt::matrix_view(device, batch_output_target);
            auto d_output_matrix_view = rlt::matrix_view(device, d_output);
            rlt::nn::loss_functions::mse::gradient(device, output_matrix_view, output_target_matrix_view, d_output_matrix_view);
            T loss = rlt::nn::loss_functions::mse::evaluate(device, output_matrix_view, output_target_matrix_view);
            rlt::set_step(device, device.logger, epoch_i * (N/BATCH_SIZE) + batch_i);
            rlt::add_scalar(device, device.logger, "loss", loss);
            epoch_loss += loss;
            epoch_loss_count++;
            rlt::zero_gradient(device, actor);
            rlt::backward(device, actor, batch_input, d_output, actor_buffer, evaluation_mode);
            rlt::step(device, actor_optimizer, actor);
        }
        std::cout << "Epoch: " << epoch_i << " Loss: " << epoch_loss/epoch_loss_count << std::endl;
        {
            using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename ACTOR::template CHANGE_BATCH_SIZE<TI, NUM_EPISODES>;
            using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>>;
            rlt::rl::environments::DummyUI ui;
            EVALUATION_ACTOR_TYPE evaluation_actor;
            EVALUATION_ACTOR_TYPE::Buffer<DYNAMIC_ALLOCATION> eval_buffer;
            rlt::malloc(device, evaluation_actor);
            rlt::malloc(device, eval_buffer);
            rlt::copy(device, device, actor, evaluation_actor);

            ENVIRONMENT env_eval;
            ENVIRONMENT::Parameters env_eval_parameters;
            rlt::init(device, env_eval);
            rlt::sample_initial_parameters(device, env_eval, env_eval_parameters, rng);
            rlt::evaluate(device, env_eval, env_eval_parameters, ui, evaluation_actor, result, data, eval_buffer, rng, evaluation_mode, false, true);
            rlt::add_scalar(device, device.logger, "evaluation/return/mean", result.returns_mean);
            rlt::add_scalar(device, device.logger, "evaluation/return/std", result.returns_std);
            rlt::add_scalar(device, device.logger, "evaluation/episode_length/mean", result.episode_length_mean);
            rlt::add_scalar(device, device.logger, "evaluation/episode_length/std", result.episode_length_std);
            rlt::add_scalar(device, device.logger, "evaluation/share_terminated", result.share_terminated);
            rlt::log(device, device.logger, "Mean return: ", result.returns_mean, " Mean episode length: ", result.episode_length_mean, " Share terminated: ", result.share_terminated * 100, "%");

            if (!best_return_set || result.returns_mean > best_return){
                best_return = result.returns_mean;
                best_return_set = true;
                rlt::copy(device, device, evaluation_actor, best_actor);
                std::cout << "Best return: " << best_return << std::endl;
            }

            rlt::free(device, evaluation_actor);
            rlt::free(device, eval_buffer);
        }
    }

    rlt::rl::loop::steps::checkpoint::save<DYNAMIC_ALLOCATION, ENVIRONMENT>(device, run_path.string(), best_actor, rng);
    // malloc
    rlt::free(device, device.logger);
    rlt::free(device, rng);
    rlt::free(device, actor_optimizer);
    rlt::free(device, actor);
    rlt::free(device, actor_buffer);
    rlt::free(device, dataset_input);
    rlt::free(device, dataset_output_target);
    rlt::free(device, d_output);
    delete result_memory;
    delete data_memory;
    return 0;
}
