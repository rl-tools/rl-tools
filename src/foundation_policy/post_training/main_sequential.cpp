#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
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
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_PARAMS_DEVICE = rlt::devices::random::Generic<DEVICE::SPEC::MATH>;
using RNG_PARAMS = RNG_PARAMS_DEVICE::ENGINE<>;
using TI = typename DEVICE::index_t;
using T = float;
constexpr bool DYNAMIC_ALLOCATION = true;

struct OPTIONS_POST_TRAINING: OPTIONS_PRE_TRAINING{
    static constexpr bool OBSERVE_THRASH_MARKOV = false;
    static constexpr bool MOTOR_DELAY = true;
    static constexpr bool ACTION_HISTORY = true;
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr bool OBSERVATION_NOISE = true;
};

struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T>{
    static constexpr T ALPHA = 0.003;
};
// constants parameters
constexpr TI NUM_EPISODES = 2000;
constexpr TI N_EPOCH = 50000;
constexpr TI N_PRE_TRAINING_SEEDS = 1;
constexpr TI SEQUENCE_LENGTH = 500;
constexpr TI BATCH_SIZE = 8;
constexpr T SOLVED_RETURN = 550;
constexpr TI DMODEL = 32;
constexpr bool RANDOM_START_STEP = false;

// typedefs
using ENVIRONMENT = typename builder::ENVIRONMENT_FACTORY_POST_TRAINING<DEVICE, T, TI, OPTIONS_POST_TRAINING>::ENVIRONMENT;
struct ENVIRONMENT_PT: ENVIRONMENT{
    using LOOP_CORE_CONFIG_PRE_TRAINING = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING, DYNAMIC_ALLOCATION>::LOOP_CORE_CONFIG;
    using ENV = LOOP_CORE_CONFIG_PRE_TRAINING::ENVIRONMENT;
    using Observation = ENV::Observation;
    using ObservationPrivileged = Observation;
};
using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::Observation::DIM>;

using INPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, DMODEL, rlt::nn::activation_functions::RELU>;
using INPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, DMODEL>;
using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
using OUTPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, 4, rlt::nn::activation_functions::IDENTITY>;
using OUTPUT_LAYER = rlt::nn::layers::dense::BindConfiguration<OUTPUT_LAYER_CONFIG>;
template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential::OutputModule>
using Module = typename rlt::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
using MODULE_CHAIN = Module<INPUT_LAYER, Module<GRU, Module<OUTPUT_LAYER>>>;

// using MLP_CONFIG = rlt::nn_models::mlp::Configuration<T, TI, ENVIRONMENT::ACTION_DIM, 3, DMODEL, rlt::nn::activation_functions::ActivationFunction::FAST_TANH, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
// using MLP = rlt::nn_models::mlp::BindConfiguration<MLP_CONFIG>;
// using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
// template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential::OutputModule>
// using Module = typename rlt::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
// using MODULE_CHAIN = Module<MLP>;

using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, DYNAMIC_ALLOCATION>;
using ACTOR = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
using OPTIMIZER = rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<T, TI, ADAM_PARAMETERS>>;
using OUTPUT_SHAPE = ACTOR::OUTPUT_SHAPE;
using RESULT = rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES, ENVIRONMENT::EPISODE_STEP_LIMIT>>;
using DATA = rlt::rl::utils::evaluation::Data<RESULT::SPEC>;

// constants derived
constexpr TI DATASET_SIZE = N_PRE_TRAINING_SEEDS * NUM_EPISODES * ENVIRONMENT::EPISODE_STEP_LIMIT;

// typedefs derived
using INPUT_SHAPE_DATASET = rlt::tensor::Shape<TI, DATASET_SIZE, ENVIRONMENT::Observation::DIM>;
using OUTPUT_SHAPE_DATASET = rlt::tensor::Shape<TI, DATASET_SIZE, ENVIRONMENT::ACTION_DIM>;
template <typename DATA, typename INPUT_SPEC, typename OUTPUT_SPEC, typename PARAMS, typename RNG>
TI add_to_dataset(DEVICE& device, DATA& data, rlt::Tensor<INPUT_SPEC>& input, rlt::Tensor<OUTPUT_SPEC>& output, TI& current_index, std::vector<TI>& episode_start_indices, PARAMS& base_parameters, RNG& rng){
    TI initial_index = current_index;
    ENVIRONMENT env_eval;
    ENVIRONMENT::Parameters env_eval_parameters;
    rlt::init(device, env_eval);
    env_eval.parameters = base_parameters;
    rlt::initial_parameters(device, env_eval, env_eval_parameters);

    for (TI episode_i = 0; episode_i < DATA::SPEC::N_EPISODES; episode_i++){
        episode_start_indices.push_back(current_index);
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
            if (data.terminated[episode_i][current_step_i]){
                current_step_i += 1; // +1 because we terminate with current_step_i characterizing the number of steps added to the buffer and the increment does not happen in this case
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
    ACTOR actor;
    ACTOR::Buffer<> actor_buffer;
    OPTIMIZER actor_optimizer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 1>>> reset;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_target, d_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_DATASET>> dataset_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_DATASET>> dataset_output_target;
    RESULT result;
    DATA* data;

    // device init
    rlt::init(device);

    // malloc
    rlt::malloc(device, rng);
    rlt::malloc(device, actor_optimizer);
    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, reset);
    rlt::malloc(device, output_target);
    rlt::malloc(device, d_output);
    rlt::malloc(device, dataset_input);
    rlt::malloc(device, dataset_output_target);
    data = new DATA;

    // init
    TI seed = argc >= 2 ? std::stoi(argv[1]) : 0;
    TI current_index = 0;

#ifdef RL_TOOLS_ENABLE_TENSORBOARD
    auto timestamp_string = rlt::utils::extrack::get_timestamp_string();
    std::filesystem::path run_path = "logs/" + timestamp_string;
    rlt::init(device, device.logger, run_path.string());
#endif
    rlt::init(device, rng, seed);
    rlt::init_weights(device, actor, rng);

    //work
    rlt::utils::extrack::Path checkpoint_path;
    checkpoint_path.experiment = "2025-02-20_15-25-14";
    checkpoint_path.name = "foundation-policy-pre-training";

    std::vector<TI> episode_start_indices;
    for (TI seed_i = 0; seed_i < N_PRE_TRAINING_SEEDS; seed_i++){
        checkpoint_path.seed = std::to_string(seed_i);
        rlt::find_latest_run(device, "experiments", checkpoint_path);
        if (!rlt::find_latest_checkpoint(device, checkpoint_path)){
            std::cerr << "No checkpoint found for " << checkpoint_path.experiment << std::endl;
            return 1;
        }
        std::cout << "Found checkpoint: " << checkpoint_path.checkpoint_path << std::endl;
        auto base_parameters = generate_data<DEVICE, RNG_PARAMS, RNG, RNG_PARAMS_DEVICE, ENVIRONMENT_PT, T, TI, NUM_EPISODES>(device, checkpoint_path, seed_i, result, data);
        if (result.returns_mean < SOLVED_RETURN){
            std::cerr << "Mean return (" << result.returns_mean << ") too low for " << checkpoint_path.checkpoint_path << std::endl;
            return 1;
        }
        rlt::log(device, device.logger, "Checkpoint ", checkpoint_path.checkpoint_path.string(), ": Mean return: ", result.returns_mean, " Mean episode length: ", result.episode_length_mean);
        TI num_added = add_to_dataset(device, *data, dataset_input, dataset_output_target, current_index, episode_start_indices, base_parameters, rng);
        if (num_added == 0){
            std::cout << "Dataset full after " << seed_i << " seeds" << std::endl;
            break;
        }
    }
    TI N = current_index;

    std::vector<TI> episode_end_indices(episode_start_indices.begin() + 1, episode_start_indices.end());
    episode_end_indices.push_back(N-1);

    std::vector<TI> epoch_start_index_indices(episode_start_indices.size());
    std::iota(epoch_start_index_indices.begin(), epoch_start_index_indices.end(), 0);
    rlt::reset_optimizer_state(device, actor_optimizer, actor);
    for (TI epoch_i = 0; epoch_i < N_EPOCH; epoch_i++){
        std::shuffle(epoch_start_index_indices.begin(), epoch_start_index_indices.end(), rng.engine);
        constexpr TI BATCH_SIZE = INPUT_SHAPE::GET<1>;
        T epoch_loss = 0;
        TI epoch_loss_count = 0;
        TI current_episode_index = 0;
        for (TI batch_i = 0; batch_i < N / BATCH_SIZE; batch_i++){
            for (TI batch_sample_i = 0; batch_sample_i < BATCH_SIZE; batch_sample_i++){
                TI sequence_i = 0;
                while (sequence_i < SEQUENCE_LENGTH){
                    TI current_start_index = episode_start_indices[epoch_start_index_indices[current_episode_index]];
                    TI current_end_index = episode_end_indices[epoch_start_index_indices[current_episode_index]];
                    if (RANDOM_START_STEP) {
                        current_start_index = rlt::random::uniform_int_distribution(device.random, current_start_index, current_end_index, rng);
                    }
                    for (TI episode_step_i = 0; episode_step_i < SEQUENCE_LENGTH; episode_step_i++){
                        if (episode_step_i == 0){
                            rlt::set(device, reset, true, sequence_i, batch_sample_i, 0);
                        }
                        else{
                            rlt::set(device, reset, false, sequence_i, batch_sample_i, 0);
                        }
                        auto input_step_target = rlt::view(device, input, sequence_i);
                        auto output_target_step_target = rlt::view(device, output_target, sequence_i);

                        auto input_target = rlt::view(device, input_step_target, batch_sample_i);
                        auto output_target_target = rlt::view(device, output_target_step_target, batch_sample_i);

                        auto input_source = rlt::view(device, dataset_input, current_start_index + episode_step_i);
                        auto output_target_source = rlt::view(device, dataset_output_target, current_start_index + episode_step_i);
                        rlt::copy(device, device, input_source, input_target);
                        rlt::copy(device, device, output_target_source, output_target_target);

                        if (current_start_index + episode_step_i == current_end_index){
                            if ((current_episode_index + 1) >= epoch_start_index_indices.size()){
                                std::cout << "current_episode_index: " << current_episode_index << " epoch_start_index_indices.size(): " << epoch_start_index_indices.size() << std::endl;
                                goto end_epoch;
                            }
                            break;
                        }
                        sequence_i++;
                        if (sequence_i >= SEQUENCE_LENGTH){
                            break;
                        }
                    }
                    current_episode_index++;
                    if (current_episode_index >= epoch_start_index_indices.size()){
                        std::cout << "current_episode_index: " << current_episode_index << " epoch_start_index_indices.size(): " << epoch_start_index_indices.size() << std::endl;
                        goto end_epoch;
                    }
                }
            }
            using EVAL_MODE = rlt::Mode<rlt::nn::layers::gru::ResetMode<rlt::mode::Default<>, rlt::nn::layers::gru::ResetModeSpecification<TI, decltype(reset)>>>;
            EVAL_MODE mode;
            mode.reset_container = reset;
            // rlt::set_all(device, reset, true);
            rlt::forward(device, actor, input, actor_buffer, rng, mode);
            auto output_matrix_view = rlt::matrix_view(device, rlt::output(device, actor));
            auto output_target_matrix_view = rlt::matrix_view(device, output_target);
            auto d_output_matrix_view = rlt::matrix_view(device, d_output);
            rlt::nn::loss_functions::mse::gradient(device, output_matrix_view, output_target_matrix_view, d_output_matrix_view);
            T loss = rlt::nn::loss_functions::mse::evaluate(device, output_matrix_view, output_target_matrix_view);
            rlt::set_step(device, device.logger, device.logger.step + 1);
            rlt::add_scalar(device, device.logger, "loss", loss);
            epoch_loss += loss;
            epoch_loss_count++;
            rlt::zero_gradient(device, actor);
            rlt::backward(device, actor, input, d_output, actor_buffer, mode);
            rlt::step(device, actor_optimizer, actor);
        }
        end_epoch:
        std::cout << "Epoch: " << epoch_i << " Loss: " << epoch_loss/epoch_loss_count << std::endl;
        if (epoch_i % 50 == 0){
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
            using EVAL_MODE = rlt::Mode<rlt::mode::Default<>>;
            EVAL_MODE mode;
            rlt::evaluate(device, env_eval, env_eval_parameters, ui, evaluation_actor, result, *data, eval_buffer, rng, mode, false, true);
            rlt::add_scalar(device, device.logger, "evaluation/return/mean", result.returns_mean);
            rlt::add_scalar(device, device.logger, "evaluation/return/std", result.returns_std);
            rlt::add_scalar(device, device.logger, "evaluation/episode_length/mean", result.episode_length_mean);
            rlt::add_scalar(device, device.logger, "evaluation/episode_length/std", result.episode_length_std);
            rlt::add_scalar(device, device.logger, "evaluation/share_terminated", result.share_terminated);
            rlt::log(device, device.logger, "Mean return: ", result.returns_mean, " Mean episode length: ", result.episode_length_mean, " Share terminated: ", result.share_terminated * 100, "%");

            rlt::free(device, evaluation_actor);
            rlt::free(device, eval_buffer);
        }
    }

    rlt::rl::loop::steps::checkpoint::save<DYNAMIC_ALLOCATION, ENVIRONMENT>(device, run_path.string(), actor, rng);
    // malloc
    rlt::free(device, device.logger);
    rlt::free(device, rng);
    rlt::free(device, actor_optimizer);
    rlt::free(device, actor);
    rlt::free(device, actor_buffer);
    rlt::free(device, dataset_input);
    rlt::free(device, dataset_output_target);
    rlt::free(device, d_output);
    delete data;
    return 0;
}
