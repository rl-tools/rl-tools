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
constexpr bool SHUFFLE = false;

#define RL_TOOLS_POST_TRAINING
#include "config.h"

// constants derived
constexpr TI DATASET_SIZE = N_PRE_TRAINING_SEEDS * NUM_EPISODES * ENVIRONMENT::EPISODE_STEP_LIMIT;

template <typename ENVIRONMENT, typename OBSERVATION_TEACHER, typename TEACHER_ORIG, typename DATA, typename INPUT_SPEC, typename OUTPUT_SPEC, typename TRUNCATED_SPEC, typename RESET_SPEC, typename RNG>
TI add_to_dataset(DEVICE& device, DATA& data, TEACHER_ORIG& teacher, rlt::Tensor<INPUT_SPEC>& input_student, rlt::Tensor<OUTPUT_SPEC>& output, rlt::Tensor<TRUNCATED_SPEC>& truncated, rlt::Tensor<RESET_SPEC>& reset, TI& current_index, RNG& rng){
    TI initial_index = current_index;
    ENVIRONMENT env_eval;
    rlt::init(device, env_eval);
    static constexpr TI DATASET_SIZE = INPUT_SPEC::SHAPE::FIRST;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DATASET_SIZE, OBSERVATION_TEACHER::DIM>>> input_teacher;
    rlt::malloc(device, input_teacher);
    bool reset_flag = true;
    for (TI episode_i = 0; episode_i < DATA::SPEC::N_EPISODES; episode_i++){
        typename ENVIRONMENT::Parameters env_eval_parameters = data.parameters[episode_i];
        TI current_step_i;
        for (current_step_i = 0; current_step_i < ENVIRONMENT::EPISODE_STEP_LIMIT; current_step_i++){
            auto observation_student_tensor = rlt::view(device, input_student, current_index + current_step_i);
            auto observation_teacher_tensor = rlt::view(device, input_teacher, current_index + current_step_i);
            auto observation_student = rlt::matrix_view(device, observation_student_tensor);
            auto observation_teacher = rlt::matrix_view(device, observation_teacher_tensor);
            rlt::observe(device, env_eval, env_eval_parameters, data.states[episode_i][current_step_i], OBSERVATION_TEACHER{}, observation_teacher, rng);
            rlt::observe(device, env_eval, env_eval_parameters, data.states[episode_i][current_step_i], typename ENVIRONMENT::Observation{}, observation_student, rng);
            bool truncated_flag = data.terminated[episode_i][current_step_i] || current_step_i == (ENVIRONMENT::EPISODE_STEP_LIMIT - 1);
            rlt::set(device, truncated, truncated_flag, current_index + current_step_i);
            rlt::set(device, reset, reset_flag, current_index + current_step_i);
            if (data.terminated[episode_i][current_step_i]){
                reset_flag = true;
                break;
            }
        }
        current_index += current_step_i;
        if (current_index >= INPUT_SPEC::SHAPE::FIRST){
            break;
        }
    }

    static constexpr TI BATCH_SIZE = 1;
    using TEACHER = typename TEACHER_ORIG::template CHANGE_BATCH_SIZE<TI, BATCH_SIZE>::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;
    typename TEACHER::template Buffer<true> policy_teacher_buffer;
    rlt::malloc(device, policy_teacher_buffer);
    typename TEACHER::template State<true> teacher_state;
    rlt::malloc(device, teacher_state);
    rlt::reset(device, teacher, teacher_state, rng);
    for(TI step_i=initial_index; step_i < current_index; ++step_i){
        static_assert(BATCH_SIZE == 1, "Batch size needs to be one for sequential state tracking (reset / evaluate_step)");
        auto input_chunk = rlt::view_range(device, input_teacher, step_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
        auto output_chunk = rlt::view_range(device, output, step_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
        auto reset_chunk = rlt::view_range(device, reset, step_i * BATCH_SIZE, rlt::tensor::ViewSpec<0, BATCH_SIZE>{});
        if (rlt::get(device, reset_chunk, 0)){
            rlt::reset(device, teacher, teacher_state, rng);
        }
        rlt::Mode<rlt::mode::Evaluation<>> mode;
        rlt::evaluate_step(device, teacher, input_chunk, teacher_state, output_chunk, policy_teacher_buffer, rng, mode);
    }

    rlt::free(device, input_teacher);
    rlt::free(device, teacher_state);
    rlt::free(device, policy_teacher_buffer);
    return current_index - initial_index;
}


template <typename DEVICE, typename STUDENT, typename TEACHER, typename DS_INPUT_SPEC, typename DS_OUTPUT_SPEC, typename DS_TRUNCATED_SPEC, typename DS_RESET_SPEC, typename TI, typename RNG>
void gather_epoch(DEVICE& device, TEACHER& teacher, STUDENT& student, rlt::Tensor<DS_INPUT_SPEC>& dataset_input_student, rlt::Tensor<DS_OUTPUT_SPEC>& dataset_output_target, rlt::Tensor<DS_TRUNCATED_SPEC>& dataset_truncated, rlt::Tensor<DS_RESET_SPEC>& dataset_reset, TI& current_index, RNG& rng){
    RESULT result;
    DATA* data_memory;
    data_memory = new DATA;
    DATA& data = *data_memory;
    sample_trajectories<ENVIRONMENT>(device, student, result, data, rng);
    add_to_dataset<ENVIRONMENT, ENVIRONMENT_PT::Observation>(device, data, teacher, dataset_input_student, dataset_output_target, dataset_truncated, dataset_reset, current_index, rng);
    delete data_memory;
}

using FACTORY = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING, DYNAMIC_ALLOCATION>;
using LOOP_CORE_CONFIG = FACTORY::LOOP_CORE_CONFIG;
using LOOP_CONFIG = builder::LOOP_ASSEMBLY<LOOP_CORE_CONFIG>::LOOP_CONFIG;
using CRITIC_ORIG = LOOP_CONFIG::ACTOR_CRITIC_TYPE::SPEC::CRITIC_NETWORK_TYPE;
using CRITIC_BS = CRITIC_ORIG::CHANGE_BATCH_SIZE<TI, BATCH_SIZE>;
using CRITIC = CRITIC_BS::CHANGE_SEQUENCE_LENGTH<TI, SEQUENCE_LENGTH>;
using CRITIC_TEMP = CRITIC::CHANGE_CAPABILITY<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>>;
using ACTOR_ORIG = LOOP_CONFIG::ACTOR_CRITIC_TYPE::SPEC::ACTOR_NETWORK_TYPE;
using ACTOR_TEACHER = ACTOR_ORIG::CHANGE_BATCH_SIZE<TI, 32>::CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;


// note: make sure that the rng_params is invoked in the exact same way in pre- as in post-training, to make sure the params used to sample parameters to generate data from the trained policy are matching the ones seen by the particular policy for the seed during pretraining

int main(int argc, char** argv){
    // declarations
    DEVICE device;
    RNG rng;
    ACTOR_TEACHER actor_teacher;
    typename ACTOR_TEACHER::Buffer<> actor_teacher_buffer;
    ACTOR actor, best_actor;
    ACTOR::Buffer<> actor_buffer;
    OPTIMIZER actor_optimizer;
    std::cout << "Input shape: " << std::endl;
    rlt::print(device, ACTOR::INPUT_SHAPE{});
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DATASET_SIZE, ENVIRONMENT::Observation::DIM>>> dataset_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DATASET_SIZE, ENVIRONMENT::ACTION_DIM>>> dataset_output_target;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>> dataset_truncated;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>> dataset_reset;
    rlt::Tensor<rlt::tensor::Specification<TI, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>> epoch_indices;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::Observation::DIM>>> batch_input;
    static_assert(CRITIC::INPUT_SHAPE::template GET<2> == ENVIRONMENT_PT::Observation::DIM + ENVIRONMENT::ACTION_DIM);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::ACTION_DIM>>> batch_output_target;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 1>>> batch_reset;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_output;

    // device init
    rlt::init(device);

    // malloc
    rlt::malloc(device, rng);
    rlt::malloc(device, actor_optimizer);
    rlt::malloc(device, actor_teacher);
    rlt::malloc(device, actor_teacher_buffer);
    rlt::malloc(device, actor);
    rlt::malloc(device, best_actor);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, dataset_input);
    rlt::malloc(device, dataset_output_target);
    rlt::malloc(device, dataset_truncated);
    rlt::malloc(device, dataset_reset);
    rlt::malloc(device, epoch_indices);
    rlt::malloc(device, batch_input);
    rlt::malloc(device, batch_output_target);
    rlt::malloc(device, batch_reset);
    rlt::malloc(device, d_output);

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
    checkpoint_path.experiment = "2025-03-17_12-34-45";
    checkpoint_path.name = "foundation-policy-pre-training";

    {
        // load actor & critic
        auto cpp_copy = checkpoint_path;
        rlt::find_latest_run(device, "experiments", cpp_copy);
        auto actor_file = HighFive::File((cpp_copy.checkpoint_path.parent_path() / "checkpoint.h5").string(), HighFive::File::ReadOnly);
        rlt::load(device, actor_teacher, actor_file.getGroup("actor"));

        RESULT result;
        DATA_EVAL no_data;
        RNG rng_copy = rng;
        sample_trajectories<ENVIRONMENT_PT>(device, actor_teacher, result, no_data, rng_copy);
        std::cout << "Teacher policy mean return: " << result.returns_mean << " episode length: " << result.episode_length_mean << " share terminated: " << result.share_terminated << std::endl;
        if (result.returns_mean < SOLVED_RETURN){
            std::cerr << "Mean return (" << result.returns_mean << ") too low for " << checkpoint_path.checkpoint_path << std::endl;
            return 1;
        }
    }

    for (TI i=0; i < DATASET_SIZE; i++){
        rlt::set(device, epoch_indices, i, i);
    }

    rlt::reset_optimizer_state(device, actor_optimizer, actor);
    for (TI epoch_i = 0; epoch_i < N_EPOCH; epoch_i++){
        current_index = 0;
        gather_epoch(device, actor_teacher, actor, dataset_input, dataset_output_target, dataset_truncated, dataset_reset, current_index, rng);
        TI N = current_index;

        if constexpr(SHUFFLE){
            for (TI i=0; i < N; i++){
                rlt::set(device, epoch_indices, i, i);
            }
            for (TI sample_i=0; sample_i<N; sample_i++){
                TI target_index = rlt::random::uniform_int_distribution(device.random, sample_i, N - 1, rng);
                TI target_value = rlt::get(device, epoch_indices, target_index);
                rlt::set(device, epoch_indices, rlt::get(device, epoch_indices, sample_i), target_index);
                rlt::set(device, epoch_indices, target_value, sample_i);
            }
        }
        constexpr TI BATCH_SIZE = INPUT_SHAPE::GET<1>;
        T epoch_loss = 0;
        TI epoch_loss_count = 0;
        for (TI batch_i = 0; batch_i < N / BATCH_SIZE; batch_i++){
            for (TI sample_i=0; sample_i<BATCH_SIZE; sample_i++){
                TI current_epoch_index = batch_i * BATCH_SIZE + sample_i;
                TI current_sample = rlt::get(device, epoch_indices, current_epoch_index);
                bool reset = false;
                for (TI step_i=0; step_i < SEQUENCE_LENGTH; step_i++){
                    auto input = rlt::view(device, dataset_input, current_sample);
                    auto input_target_step = rlt::view(device, batch_input, step_i);
                    auto input_target = rlt::view(device, input_target_step, sample_i);
                    rlt::copy(device, device, input, input_target);
                    auto output_target = rlt::view(device, dataset_output_target, current_sample);
                    auto output_target_step = rlt::view(device, batch_output_target, step_i);
                    auto output_target_target = rlt::view(device, output_target_step, sample_i);
                    rlt::copy(device, device, output_target, output_target_target);
                    rlt::set(device, batch_reset, reset, step_i, sample_i, 0);
                    current_sample = (current_sample + 1) % N;
                    reset = current_sample == 0 || rlt::get(device, dataset_truncated, current_sample);
                }
            }

            rlt::Mode<rlt::nn::layers::gru::ResetMode<rlt::mode::Default<>, rlt::nn::layers::gru::ResetModeSpecification<TI, decltype(batch_reset)>>> mode;
            mode.reset_container = batch_reset;
            rlt::forward(device, actor, batch_input, actor_buffer, rng, mode);
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
            rlt::backward(device, actor, batch_input, d_output, actor_buffer, mode);
            rlt::step(device, actor_optimizer, actor);
        }
        std::cout << "Epoch: " << epoch_i << " Loss: " << epoch_loss/epoch_loss_count << std::endl;
        auto target_path = run_path / "checkpoints" / std::to_string(epoch_i);
        if (!std::filesystem::exists(target_path)){
            std::filesystem::create_directories(target_path);
        }
        rlt::rl::loop::steps::checkpoint::save<DYNAMIC_ALLOCATION, ENVIRONMENT>(device, target_path.string(), actor, rng);
        {
            using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename ACTOR::template CHANGE_BATCH_SIZE<TI, NUM_EPISODES_EVAL>;
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
            rlt::Mode<rlt::mode::Default<>> mode;
            RESULT_EVAL result_eval;
            DATA_EVAL data_eval;
            rlt::evaluate(device, env_eval, env_eval_parameters, ui, evaluation_actor, result_eval, data_eval, eval_buffer, rng, mode, false, true);
            rlt::add_scalar(device, device.logger, "evaluation/return/mean", result_eval.returns_mean);
            rlt::add_scalar(device, device.logger, "evaluation/return/std", result_eval.returns_std);
            rlt::add_scalar(device, device.logger, "evaluation/episode_length/mean", result_eval.episode_length_mean);
            rlt::add_scalar(device, device.logger, "evaluation/episode_length/std", result_eval.episode_length_std);
            rlt::add_scalar(device, device.logger, "evaluation/share_terminated", result_eval.share_terminated);
            rlt::log(device, device.logger, "Mean return: ", result_eval.returns_mean, " Mean episode length: ", result_eval.episode_length_mean, " Share terminated: ", result_eval.share_terminated * 100, "%");

            if (!best_return_set || result_eval.returns_mean > best_return){
                best_return = result_eval.returns_mean;
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
    rlt::malloc(device, rng);
    rlt::malloc(device, actor_optimizer);
    rlt::malloc(device, actor_teacher);
    rlt::malloc(device, actor_teacher_buffer);
    rlt::malloc(device, actor);
    rlt::malloc(device, best_actor);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, dataset_input);
    rlt::malloc(device, dataset_output_target);
    rlt::malloc(device, dataset_truncated);
    rlt::malloc(device, dataset_reset);
    rlt::malloc(device, epoch_indices);
    rlt::malloc(device, batch_input);
    rlt::malloc(device, batch_output_target);
    rlt::malloc(device, batch_reset);
    rlt::malloc(device, d_output);
    return 0;
}
