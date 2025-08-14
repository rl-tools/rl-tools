#ifndef RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
// #define RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
#endif
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
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

#include "helper.h"


using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_PARAMS_DEVICE = rlt::devices::random::Generic<DEVICE::SPEC::MATH>;
using RNG_PARAMS = RNG_PARAMS_DEVICE::ENGINE<>;
using TI = DEVICE::index_t;
using T = float;

#define RL_TOOLS_POST_TRAINING
#include "config.h"

// constants derived
constexpr TI DATASET_SIZE = (ON_POLICY ? 1 : N_EPOCH) * (1 + TEACHER_STUDENT_MIX) * NUM_TEACHERS * NUM_EPISODES * ENVIRONMENT::EPISODE_STEP_LIMIT;


using FACTORY = builder::FACTORY<DEVICE, T, TI, RNG, OPTIONS_PRE_TRAINING, DYNAMIC_ALLOCATION>;
using LOOP_CORE_CONFIG = FACTORY::LOOP_CORE_CONFIG;
using LOOP_CONFIG = builder::LOOP_ASSEMBLY<LOOP_CORE_CONFIG>::LOOP_CONFIG;
using CRITIC_ORIG = LOOP_CONFIG::ACTOR_CRITIC_TYPE::SPEC::CRITIC_NETWORK_TYPE;
using CRITIC_BS = CRITIC_ORIG::CHANGE_BATCH_SIZE<TI, BATCH_SIZE>;
using CRITIC = CRITIC_BS::CHANGE_SEQUENCE_LENGTH<TI, SEQUENCE_LENGTH>;
using CRITIC_TEMP = CRITIC::CHANGE_CAPABILITY<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>>;
using ACTOR_ORIG = LOOP_CONFIG::ACTOR_CRITIC_TYPE::SPEC::ACTOR_NETWORK_TYPE;
using ACTOR_TEACHER = ACTOR_ORIG::CHANGE_BATCH_SIZE<TI, 32>::CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;

struct CHECKPOINT_PARAMETERS{
    static constexpr TI TEST_INPUT_BATCH_SIZE = 2;
};


// note: make sure that the rng_params is invoked in the exact same way in pre- as in post-training, to make sure the params used to sample parameters to generate data from the trained policy are matching the ones seen by the particular policy for the seed during pretraining

int main(int argc, char** argv){

    // declarations
    DEVICE device;
    RNG rng;
    ACTOR_TEACHER actor_teacher[NUM_TEACHERS];
    ENVIRONMENT_TEACHER::Parameters teacher_parameters[NUM_TEACHERS];
    typename ACTOR_TEACHER::Buffer<> actor_teacher_buffer;
    ACTOR actor, best_actor;
    ACTOR::Buffer<> actor_buffer;
    OPTIMIZER actor_optimizer;
    std::cout << "Input shape: " << std::endl;
    rlt::print(device, ACTOR::INPUT_SHAPE{});
    std::cout << "Dataset size: " << DATASET_SIZE << std::endl;
    rlt::Tensor<rlt::tensor::Specification<TeacherMeta<T>, TI, rlt::tensor::Shape<TI, NUM_TEACHERS>>> teacher_metas;
    rlt::Tensor<rlt::tensor::Specification<TI, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>> dataset_episode_start_indices;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DATASET_SIZE, ENVIRONMENT::Observation::DIM>>> dataset_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, DATASET_SIZE, ENVIRONMENT::ACTION_DIM>>> dataset_output_target;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>> dataset_truncated;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>> dataset_reset;
    rlt::Tensor<rlt::tensor::Specification<TI, TI, rlt::tensor::Shape<TI, DATASET_SIZE>>> epoch_indices;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::Observation::DIM>>> batch_input;
    static_assert(CRITIC::INPUT_SHAPE::template GET<2> == ENVIRONMENT_TEACHER::Observation::DIM + ENVIRONMENT::ACTION_DIM);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, ENVIRONMENT::ACTION_DIM>>> batch_output_target;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 1>>> batch_reset;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_output;

    ENVIRONMENT env_eval;
    auto permute_rotors_px4_to_cf = [&device, &env_eval](const auto& dynamics){
        auto copy = dynamics;
        rlt::permute_rotors(device, env_eval, copy, 0, 3, 1, 2);
        return copy;
    };

    // std::map<std::string, ENVIRONMENT::Parameters::Dynamics> env_eval_parameters = {
    //     {"crazyflie", rlt::rl::environments::l2f::parameters::dynamics::registry<rlt::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie, typename ENVIRONMENT::SPEC>},
    //     {"arpl", rlt::rl::environments::l2f::parameters::dynamics::registry<rlt::rl::environments::l2f::parameters::dynamics::REGISTRY::arpl, typename ENVIRONMENT::SPEC>},
    //     {"flightmare", rlt::rl::environments::l2f::parameters::dynamics::registry<rlt::rl::environments::l2f::parameters::dynamics::REGISTRY::flightmare, typename ENVIRONMENT::SPEC>},
    //     {"x500", rlt::rl::environments::l2f::parameters::dynamics::registry<rlt::rl::environments::l2f::parameters::dynamics::REGISTRY::x500_real, typename ENVIRONMENT::SPEC>}
    // };

    // device init
    rlt::init(device);

    // malloc
    rlt::malloc(device, rng);
    rlt::malloc(device, actor_optimizer);
    for (TI teacher_i=0; teacher_i < NUM_TEACHERS; ++teacher_i){
        rlt::malloc(device, actor_teacher[teacher_i]);
    }
    rlt::malloc(device, actor_teacher_buffer);
    rlt::malloc(device, actor);
    rlt::malloc(device, best_actor);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, teacher_metas);
    rlt::malloc(device, dataset_episode_start_indices);
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
    std::cout << "SEED: " << seed << std::endl;
    std::cout << "HIDDEN_DIM: " << HIDDEN_DIM << std::endl;
    std::cout << "NUM_EPISODES: " << NUM_EPISODES << std::endl;
    std::cout << "NUM_TEACHERS: " << NUM_TEACHERS << std::endl;
    TI current_episode = 0;
    TI current_index = 0;

    T best_return = 0;
    bool best_return_set = false;

    auto timestamp_string = rlt::utils::extrack::get_timestamp_string();
    // check env vars for RL_TOOLS_RUN_PATH
    const char* run_path_env = std::getenv("RL_TOOLS_RUN_PATH");
    std::filesystem::path run_path = "logs/" + timestamp_string;
    if (run_path_env != nullptr) {
        run_path = run_path_env;
        std::cout << "Using run path from environment variable: RL_TOOLS_RUN_PATH=" << run_path << std::endl;
    }


#ifdef RL_TOOLS_ENABLE_TENSORBOARD
    rlt::init(device, device.logger, run_path.string());
#endif
    std::ofstream test_stats_file(run_path / "test_stats.csv");
    // header
    test_stats_file << "epoch,global_batch,model,teacher_selection,return_mean,return_std,episode_length_mean,episode_length_std,share_terminated" << std::endl;
    rlt::init(device, rng, seed);
    rlt::init_weights(device, actor, rng);

    //work
    std::filesystem::path registry_path = "./src/foundation_policy/registry";
    rlt::utils::extrack::Path checkpoint_path;
    // checkpoint_path.experiment = "2025-03-31_21-06-47"; // fails
    // checkpoint_path.experiment = "2025-04-01_13-43-13"; // good
    // checkpoint_path.experiment = "2025-04-03_21-30-10";
    // checkpoint_path.experiment = "2025-04-04_17-00-11";
    // checkpoint_path.experiment = "2025-04-07_23-12-07";
    // checkpoint_path.experiment = "2025-04-08_23-23-52";
    checkpoint_path.experiment = "2025-04-16_20-10-58";
    checkpoint_path.name = "foundation-policy-pre-training";



    std::filesystem::path dynamics_parameters_path = "./src/foundation_policy/dynamics_parameters_" + checkpoint_path.experiment + "/";
    std::filesystem::path dynamics_parameter_index = "./src/foundation_policy/checkpoints_" + checkpoint_path.experiment + ".txt";
    // std::filesystem::path dynamics_parameter_index = "./src/foundation_policy/checkpoints_debug.txt";

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
    if (dynamics_parameter_index_lines.size() < NUM_TEACHERS){
        std::cerr << "Dynamic parameter index file is too small: " << dynamics_parameter_index << " " << dynamics_parameter_index_lines.size() << "/" << NUM_TEACHERS << std::endl;
        return 1;
    }
    if (TEACHER_SELECTION == TEACHER_SELECTION_MODE::ALL && dynamics_parameter_index_lines.size() != NUM_TEACHERS){
        std::cerr << "Expected " << NUM_TEACHERS << " dynamics parameters, but found " << dynamics_parameter_index_lines.size() << " in " << dynamics_parameter_index << std::endl;
        return 1;
    }
    if (TEACHER_SELECTION == TEACHER_SELECTION_MODE::RANDOM) {
        std::shuffle(dynamics_parameter_index_lines.begin(), dynamics_parameter_index_lines.end(), rng.engine);
    }

    for (TI teacher_i=0; teacher_i < NUM_TEACHERS; ++teacher_i){
        // load actor & critic
        std::string checkpoint_info;
        switch (TEACHER_SELECTION) {
            case TEACHER_SELECTION_MODE::ALL:
            case TEACHER_SELECTION_MODE::BEST:
            case TEACHER_SELECTION_MODE::RANDOM:
                checkpoint_info = dynamics_parameter_index_lines[dynamics_parameter_index_lines.size() - 1 - teacher_i];
                break;
            case TEACHER_SELECTION_MODE::WORST:
                checkpoint_info = dynamics_parameter_index_lines[teacher_i];
                break;
        }
        auto checkpoint_info_split = split_by_comma(checkpoint_info);
        auto cpp_copy = checkpoint_path;
        cpp_copy.attributes["dynamics-id"] = checkpoint_info_split[0]; // take from the end because we order by performance and the best are at the end
        cpp_copy.step = checkpoint_info_split[1];
        rlt::find_latest_run(device, "1k-experiments", cpp_copy);
        auto actor_file = HighFive::File(cpp_copy.checkpoint_path.string(), HighFive::File::ReadOnly);
        rlt::load(device, actor_teacher[teacher_i], actor_file.getGroup("actor"));

        std::ifstream dynamics_parameter_file = std::ifstream(dynamics_parameters_path / (cpp_copy.attributes["dynamics-id"] + ".json"));
        std::string dynamics_parameter_json((std::istreambuf_iterator<char>(dynamics_parameter_file)), std::istreambuf_iterator<char>());
        dynamics_parameter_file.close();
        ENVIRONMENT_TEACHER env;
        rlt::from_json(device, env, dynamics_parameter_json, teacher_parameters[teacher_i]);

        rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT_TEACHER, NUM_EPISODES_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>> result;
        rlt::rl::utils::evaluation::Data<rlt::rl::utils::evaluation::DataSpecification<decltype(result)::SPEC>> data;
        RNG rng_copy = rng;
        rlt::malloc(device, data);
        sample_trajectories<ENVIRONMENT_TEACHER>(device, actor_teacher[teacher_i], teacher_parameters[teacher_i], result, data, rng_copy);

        T mean_position[3] = {0, 0, 0};
        TI num_positions = 0;
        for (TI episode_i=0; episode_i < NUM_EPISODES_EVAL; episode_i++){
            for (TI step_i=STEADY_STATE_POSITION_OFFSET_ESTIMATION_START; step_i < STEADY_STATE_POSITION_OFFSET_ESTIMATION_END; step_i++){
                const auto& state = rlt::get(device, data.states, episode_i, step_i);
                if (STEADY_STATE_POSITION_CORRECTION && state.trajectory.type == rlt::rl::environments::l2f::POSITION){
                    for (TI dim_i=0; dim_i < 3; dim_i++){
                        mean_position[dim_i] += state.position[dim_i];
                    }
                    num_positions++;
                }
            }
        }
        for (TI dim_i=0; dim_i < 3; dim_i++){
            if(num_positions > 0){
                mean_position[dim_i] /= num_positions;
            }
            auto teacher_meta = rlt::get(device, teacher_metas, teacher_i);
            teacher_meta.steady_state_position_offset[dim_i] = mean_position[dim_i];
            rlt::set(device, teacher_metas, teacher_meta, teacher_i);
        }
        rlt::free(device, data);
        std::cout << "Teacher policy (" << cpp_copy.checkpoint_path.string() << ") mean return: " << result.returns_mean << " episode length: " << result.episode_length_mean << " share terminated: " << result.share_terminated << " steady state pos correction: " << mean_position[0] << "," << mean_position[1] << "," << mean_position[2] << std::endl;
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
        current_episode = ON_POLICY ? 0 : current_episode;
        current_index = ON_POLICY ? 0 : current_index; // reset dataset if ON_POLICY
        RESULT average_result;
        average_result.returns_mean = 0;
        average_result.returns_std = 0;
        average_result.episode_length_mean = 0;
        average_result.episode_length_std = 0;
        average_result.share_terminated = 0;
        TI NUM_AVG = 0;

        if (epoch_i < EPOCH_TEACHER_FORCING || TEACHER_STUDENT_MIX > 0){ // start with behavioral cloning (data gathering using teacher)
            for (TI teacher_i=0; teacher_i < NUM_TEACHERS; teacher_i++){
                auto teacher_meta = rlt::get(device, teacher_metas, teacher_i);
                constexpr TI TEACHER_EPOCHS = (TEACHER_STUDENT_MIX > 0 ? TEACHER_STUDENT_MIX : 1);
                for (TI teacher_epoch_i = 0; teacher_epoch_i < TEACHER_EPOCHS; teacher_epoch_i++){
                    static_assert(DATASET_SIZE >= NUM_TEACHERS * TEACHER_EPOCHS * NUM_EPISODES * ENVIRONMENT::EPISODE_STEP_LIMIT);
                    auto result = gather_epoch<ENVIRONMENT_TEACHER, ENVIRONMENT_TEACHER::Observation, ENVIRONMENT::Observation, NUM_EPISODES, TEACHER_DETERMINISTIC>(device, actor_teacher[teacher_i], teacher_meta, teacher_parameters[teacher_i], actor_teacher[teacher_i], dataset_episode_start_indices, dataset_input, dataset_output_target, dataset_truncated, dataset_reset, current_episode, current_index, rng);
                    if (epoch_i < EPOCH_TEACHER_FORCING){
                        NUM_AVG++;
                        average_result.returns_mean += result.returns_mean;
                        average_result.returns_std += result.returns_mean * result.returns_mean;
                        average_result.episode_length_mean += result.episode_length_mean;
                        average_result.episode_length_std += result.episode_length_mean * result.episode_length_mean;
                        average_result.share_terminated += result.share_terminated;
                    }
                }
            }
        }
        if (epoch_i >= EPOCH_TEACHER_FORCING){
            using RESULT = rlt::rl::utils::evaluation::Result<rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES, ENVIRONMENT::EPISODE_STEP_LIMIT>>;
            RESULT results[NUM_TEACHERS];
            std::vector<std::tuple<TI, T>> active_teachers;
            rlt::rl::utils::evaluation::Data<rlt::rl::utils::evaluation::DataSpecification<RESULT::SPEC>> datas[NUM_TEACHERS];
            for (TI teacher_i=0; teacher_i < NUM_TEACHERS; teacher_i++){
                rlt::malloc(device, datas[teacher_i]);
                auto& result = results[teacher_i];
                auto& data = datas[teacher_i];
                // auto result = gather_epoch<ENVIRONMENT, ENVIRONMENT_TEACHER::Observation, ENVIRONMENT::Observation, NUM_EPISODES, TEACHER_DETERMINISTIC>(device, actor_teacher[teacher_i], teacher_parameters[teacher_i], actor, dataset_episode_start_indices, dataset_input, dataset_output_target, dataset_truncated, dataset_reset, current_episode, current_index, rng);
                sample_trajectories<ENVIRONMENT>(device, actor, teacher_parameters[teacher_i], result, data, rng);
                NUM_AVG++;
                average_result.returns_mean += result.returns_mean;
                average_result.returns_std += result.returns_mean * result.returns_mean;
                average_result.episode_length_mean += result.episode_length_mean;
                average_result.episode_length_std += result.episode_length_mean * result.episode_length_mean;
                average_result.share_terminated += result.share_terminated;
                active_teachers.emplace_back(teacher_i, result.returns_mean);
            }
            auto argsort = [](const auto& v, auto comp) {
                std::vector<std::size_t> idx(v.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&](auto i, auto j){return comp(v[i], v[j]);});
                return idx;
            };
            auto indices = argsort(active_teachers, [](const auto& a, const auto& b) { return std::get<1>(a) < std::get<1>(b); }); // ascending order
            for (TI teacher_i=0; teacher_i < NUM_TEACHERS; teacher_i++){
                if (indices[teacher_i] < NUM_ACTIVE_TEACHERS){
                    auto teacher_meta = rlt::get(device, teacher_metas, teacher_i);
                    add_to_dataset<ENVIRONMENT, ENVIRONMENT_TEACHER::Observation, ENVIRONMENT::Observation, TEACHER_DETERMINISTIC>(device, datas[teacher_i], actor_teacher[teacher_i], teacher_meta, dataset_episode_start_indices, dataset_input, dataset_output_target, dataset_truncated, dataset_reset, current_episode, current_index, rng);
                }
                rlt::free(device, datas[teacher_i]);
            }
        }

        average_result.returns_mean /= NUM_AVG;
        average_result.returns_std /= NUM_AVG;
        average_result.returns_std = std::sqrt(average_result.returns_std - average_result.returns_mean * average_result.returns_mean);
        average_result.episode_length_mean /= NUM_AVG;
        average_result.episode_length_std /= NUM_AVG;
        average_result.episode_length_std = std::sqrt(average_result.episode_length_std - average_result.episode_length_mean * average_result.episode_length_mean);
        average_result.share_terminated /= NUM_AVG;
        rlt::add_scalar(device, device.logger, "evaluation/return/mean", average_result.returns_mean);
        rlt::add_scalar(device, device.logger, "evaluation/return/std", average_result.returns_std);
        rlt::add_scalar(device, device.logger, "evaluation/episode_length/mean", average_result.episode_length_mean);
        rlt::add_scalar(device, device.logger, "evaluation/episode_length/std", average_result.episode_length_std);
        rlt::add_scalar(device, device.logger, "evaluation/share_terminated", average_result.share_terminated);
        rlt::log(device, device.logger, (epoch_i >= EPOCH_TEACHER_FORCING ? "Student" : "Teacher"), " Mean return: ", average_result.returns_mean, " Mean episode length: ", average_result.episode_length_mean, " Share terminated: ", average_result.share_terminated * 100, "%");

        if (epoch_i >= EPOCH_TEACHER_FORCING && (!best_return_set || average_result.returns_mean > best_return)){
            best_return = average_result.returns_mean;
            best_return_set = true;
            rlt::copy(device, device, actor, best_actor);
            std::cout << "Best return: " << best_return << std::endl;
        }

        TI N = current_index;
        TI N_EPISODE = current_episode;

        for (TI i=0; i < N_EPISODE; i++){
            rlt::set(device, epoch_indices, i, i);
        }
        if constexpr(SHUFFLE){
            for (TI sample_i=0; sample_i<N_EPISODE; sample_i++){
                TI target_index = rlt::random::uniform_int_distribution(device.random, sample_i, N_EPISODE - 1, rng);
                TI target_value = rlt::get(device, epoch_indices, target_index);
                rlt::set(device, epoch_indices, rlt::get(device, epoch_indices, sample_i), target_index);
                rlt::set(device, epoch_indices, target_value, sample_i);
            }
        }
        constexpr TI BATCH_SIZE = INPUT_SHAPE::GET<1>;
        T epoch_loss = 0;
        TI epoch_loss_count = 0;
        TI epoch_episode_index = 0;
        std::cout << "Epoch: " << epoch_i << " has " << N << " samples and " << N_EPISODE << " episodes" << std::endl;
        TI batch_i = 0;
        TI epoch_episode = rlt::get(device, epoch_indices, epoch_episode_index);
        TI current_sample = rlt::get(device, dataset_episode_start_indices, epoch_episode);
        TI global_batch = 0;
        while(true){
            global_batch = epoch_i * (N/BATCH_SIZE) + batch_i;
            for (TI sample_i=0; sample_i<BATCH_SIZE; sample_i++){
                // TI current_epoch_index = batch_i * BATCH_SIZE + sample_i;
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
                    TI next_sample = (current_sample + 1) % N;
                    reset = next_sample == 0 || rlt::get(device, dataset_truncated, current_sample);
                    if(!reset){
                        current_sample = next_sample;
                    }
                    else{
                        if(++epoch_episode_index >= N_EPISODE){
                            std::cerr << "Epoch episode index exceeded after " << batch_i << " batches" << std::endl;
                            goto end_of_epoch;
                        }
                        current_sample = rlt::get(device, dataset_episode_start_indices, rlt::get(device, epoch_indices, epoch_episode_index));
                    }
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
            rlt::set_step(device, device.logger, global_batch);
            rlt::add_scalar(device, device.logger, "loss", loss);
            epoch_loss += loss;
            epoch_loss_count++;
            rlt::zero_gradient(device, actor);
            rlt::backward(device, actor, batch_input, d_output, actor_buffer, mode);
            rlt::step(device, actor_optimizer, actor);
            batch_i++;
        }
        end_of_epoch:
        std::cout << "Epoch: " << epoch_i << " #Batches: " << batch_i << " Loss: " << epoch_loss/epoch_loss_count << std::endl;
        #ifndef RL_TOOLS_DISABLE_INTERMEDIATE_CHECKPOINTS
        auto target_path = run_path / "checkpoints" / std::to_string(epoch_i);
        if (!std::filesystem::exists(target_path)){
            std::filesystem::create_directories(target_path);
        }
        rlt::rl::loop::steps::checkpoint::save<DYNAMIC_ALLOCATION, ENVIRONMENT, CHECKPOINT_PARAMETERS>(device, target_path.string(), actor, rng);
        #endif
        for (const auto& entry : std::filesystem::directory_iterator(registry_path)) {
            if (entry.is_regular_file()){
                std::string file_name_without_extension = entry.path().stem().string();
                if (file_name_without_extension == ".gitignore") {
                    continue;
                }
                std::ifstream parameters_file(entry.path());
                if (!parameters_file){
                    std::cerr << "Failed to open parameters file: " << entry.path() << std::endl;
                    continue;
                }
                std::stringstream parameters_json;
                parameters_json << parameters_file.rdbuf();
                parameters_file.close();
                using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename ACTOR::template CHANGE_BATCH_SIZE<TI, NUM_EPISODES_EVAL>;
                using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>>;
                rlt::rl::environments::DummyUI ui;
                EVALUATION_ACTOR_TYPE evaluation_actor;
                EVALUATION_ACTOR_TYPE::Buffer<DYNAMIC_ALLOCATION> eval_buffer;
                rlt::malloc(device, evaluation_actor);
                rlt::malloc(device, eval_buffer);
                rlt::copy(device, device, actor, evaluation_actor);

                ENVIRONMENT::Parameters env_eval_parameters;
                rlt::from_json(device, env_eval, parameters_json.str(), env_eval_parameters);
                rlt::init(device, env_eval);
                env_eval.parameters.dynamics = env_eval_parameters.dynamics;
                rlt::sample_initial_parameters(device, env_eval, env_eval_parameters, rng);
                rlt::Mode<rlt::mode::Default<>> mode;
                RESULT_EVAL result_eval;
                DATA_EVAL data_eval;
                rlt::evaluate(device, env_eval, ui, evaluation_actor, result_eval, data_eval, rng, mode);
                rlt::add_scalar(device, device.logger, "test/" + file_name_without_extension + "/return/mean", result_eval.returns_mean);
                rlt::add_scalar(device, device.logger, "test/" + file_name_without_extension + "/return/std", result_eval.returns_std);
                rlt::add_scalar(device, device.logger, "test/" + file_name_without_extension + "/episode_length/mean", result_eval.episode_length_mean);
                rlt::add_scalar(device, device.logger, "test/" + file_name_without_extension + "/episode_length/std", result_eval.episode_length_std);
                rlt::add_scalar(device, device.logger, "test/" + file_name_without_extension + "/share_terminated", result_eval.share_terminated);
                rlt::log(device, device.logger, file_name_without_extension + ": Mean return: ", result_eval.returns_mean, " Mean episode length: ", result_eval.episode_length_mean, " Share terminated: ", result_eval.share_terminated * 100, "%");
                std::string teacher_selection;
                switch (TEACHER_SELECTION) {
                    case TEACHER_SELECTION_MODE::ALL:
                        teacher_selection = "all";
                        break;
                    case TEACHER_SELECTION_MODE::BEST:
                        teacher_selection = "best";
                        break;
                    case TEACHER_SELECTION_MODE::WORST:
                        teacher_selection = "worst";
                        break;
                    case TEACHER_SELECTION_MODE::RANDOM:
                        teacher_selection = "random";
                        break;
                    default:
                        teacher_selection = "unknown";
                }
                test_stats_file << epoch_i << "," << global_batch << "," << file_name_without_extension << "," << teacher_selection << "," << result_eval.returns_mean << "," << result_eval.returns_std << "," << result_eval.episode_length_mean << "," << result_eval.episode_length_std << "," << result_eval.share_terminated << std::endl;
                rlt::free(device, evaluation_actor);
                rlt::free(device, eval_buffer);
            }
        }
    }

    rlt::rl::loop::steps::checkpoint::save<DYNAMIC_ALLOCATION, ENVIRONMENT, CHECKPOINT_PARAMETERS>(device, run_path.string(), best_actor, rng);
    // malloc
    rlt::free(device, rng);
    rlt::free(device, actor_optimizer);
    for (TI teacher_i=0; teacher_i < NUM_TEACHERS; ++teacher_i){
        rlt::free(device, actor_teacher[teacher_i]);
    }
    rlt::free(device, actor_teacher_buffer);
    rlt::free(device, actor);
    rlt::free(device, best_actor);
    rlt::free(device, actor_buffer);
    rlt::free(device, teacher_metas);
    rlt::free(device, dataset_episode_start_indices);
    rlt::free(device, dataset_input);
    rlt::free(device, dataset_output_target);
    rlt::free(device, dataset_truncated);
    rlt::free(device, dataset_reset);
    rlt::free(device, epoch_indices);
    rlt::free(device, batch_input);
    rlt::free(device, batch_output_target);
    rlt::free(device, batch_reset);
    rlt::free(device, d_output);
    return 0;
}
