#include "training.h"
#include <cassert>

template <typename T_ABLATION_SPEC>
void train(typename multirotor_training::config::Config<T_ABLATION_SPEC>::TI seed = 0){
    using namespace multirotor_training::config;

    using CONFIG = multirotor_training::config::Config<T_ABLATION_SPEC>;
    using T = typename CONFIG::T;
    using TI = typename CONFIG::TI;

    std::cout << "Seed " << seed << "\n";
    multirotor_training::operations::TrainingState<CONFIG> ts;
    multirotor_training::operations::init(ts, seed);
    for(TI step_i=0; step_i < CONFIG::STEP_LIMIT; step_i++){
        multirotor_training::operations::step(ts);
    }
    {
        std::string DATA_FILE_PATH = std::string("learning_curves_") + ts.run_name + ".h5";
        auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::Overwrite);
        std::vector<TI> step;
        std::vector<T> returns_mean, returns_std, episode_length_mean, episode_length_std;
        for(TI eval_i = 0; eval_i < decltype(ts)::N_EVALUATIONS; eval_i++){
            step.push_back(eval_i * CONFIG::EVALUATION_INTERVAL);
            returns_mean.push_back(ts.evaluation_results[eval_i].returns_mean);
            returns_std.push_back(ts.evaluation_results[eval_i].returns_std);
            episode_length_mean.push_back(ts.evaluation_results[eval_i].episode_length_mean);
            episode_length_std.push_back(ts.evaluation_results[eval_i].episode_length_std);
        }
        data_file.createDataSet("step", step);
        data_file.createDataSet("returns_mean", returns_mean);
        data_file.createDataSet("returns_std", returns_std);
        data_file.createDataSet("episode_length_mean", episode_length_mean);
        data_file.createDataSet("episode_length_std", episode_length_std);

    }

    multirotor_training::operations::destroy(ts);
}

template <typename TI>
struct AblationSpecBase: parameters::DefaultAblationSpec{
    static constexpr bool DISTURBANCE = true;
    static constexpr bool OBSERVATION_NOISE = true;
    static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
    static constexpr bool ROTOR_DELAY = true;
    static constexpr bool ACTION_HISTORY = true;
    static constexpr bool ENABLE_CURRICULUM = true;
    static constexpr bool USE_INITIAL_REWARD_FUNCTION = true;
    static constexpr bool RECALCULATE_REWARDS = true;
    static constexpr bool EXPLORATION_NOISE_DECAY = true;
    static constexpr TI NUM_RUNS = 50;
};

template <typename TI, bool T_DISTURBANCE, bool T_OBSERVATION_NOISE, bool T_ASYMMETRIC_ACTOR_CRITIC, bool T_ROTOR_DELAY, bool T_ACTION_HISTORY, bool T_ENABLE_CURRICULUM, bool T_USE_INITIAL_REWARD_FUNCTION, bool T_RECALCULATE_REWARDS, bool T_EXPLORATION_NOISE_DECAY>
struct AblationSpecTemplate: AblationSpecBase<TI> {
    static constexpr bool DISTURBANCE = T_DISTURBANCE;
    static constexpr bool OBSERVATION_NOISE = T_OBSERVATION_NOISE;
    static constexpr bool ASYMMETRIC_ACTOR_CRITIC = T_ASYMMETRIC_ACTOR_CRITIC;
    static constexpr bool ROTOR_DELAY = T_ROTOR_DELAY;
    static constexpr bool ACTION_HISTORY = T_ACTION_HISTORY;
    static constexpr bool ENABLE_CURRICULUM = T_ENABLE_CURRICULUM;
    static constexpr bool USE_INITIAL_REWARD_FUNCTION = T_USE_INITIAL_REWARD_FUNCTION;
    static constexpr bool RECALCULATE_REWARDS = T_RECALCULATE_REWARDS;
    static constexpr bool EXPLORATION_NOISE_DECAY = T_EXPLORATION_NOISE_DECAY;
    static_assert(!ACTION_HISTORY || ROTOR_DELAY); // action history implies rotor delay
    static_assert(!ENABLE_CURRICULUM || USE_INITIAL_REWARD_FUNCTION); // curriculum implies initial reward function
};

int main(int argc, char** argv){
    std::cout << "Running the ablation study using BackpropTools: " BACKPROP_TOOLS_STRINGIFY(BACKPROP_TOOLS_COMMIT_HASH) << std::endl;

    using TI = int;
    using BASE_CONFIG = multirotor_training::config::Config<AblationSpecBase<TI>>;

    TI job_array_id;
    assert(argc == 1 || argc == 2);
    if(argc == 2){
        job_array_id = std::stoi(argv[1]);
    }
    else{
        job_array_id = 0;
    }
    TI ablation_id = job_array_id / AblationSpecBase<TI>::NUM_RUNS;
    TI run_id = job_array_id % AblationSpecBase<TI>::NUM_RUNS;
    // T_DISTURBANCE T_OBSERVATION_NOISE T_ASYMMETRIC_ACTOR_CRITIC T_ROTOR_DELAY T_ACTION_HISTORY T_ENABLE_CURRICULUM T_USE_INITIAL_REWARD_FUNCTION T_RECALCULATE_REWARDS T_EXPLORATION_NOISE_DECAY

    using ABLATION_SPEC_00 = AblationSpecTemplate<TI, true,  true,  true,  true,  true,  true,  true,  true, true>;
    using ABLATION_SPEC_01 = AblationSpecTemplate<TI, true,  true,  true,  true,  true, false,  true,  true, true>;
    using ABLATION_SPEC_02 = AblationSpecTemplate<TI, true,  true,  true,  true,  true, false, false,  true, true>;
    using ABLATION_SPEC_03 = AblationSpecTemplate<TI, true,  true,  true,  true, false,  true,  true,  true, true>;
    using ABLATION_SPEC_04 = AblationSpecTemplate<TI, true,  true,  true, false, false,  true,  true,  true, true>;
    using ABLATION_SPEC_05 = AblationSpecTemplate<TI, true,  true, false,  true,  true,  true,  true,  true, true>;
    using ABLATION_SPEC_06 = AblationSpecTemplate<TI, true, false,  true,  true,  true,  true,  true,  true, true>;
    using ABLATION_SPEC_07 = AblationSpecTemplate<TI,false,  true,  true,  true,  true,  true,  true,  true, true>;
    using ABLATION_SPEC_08 = AblationSpecTemplate<TI, true,  true,  true,  true,  true,  true,  true, false, true>;
    using ABLATION_SPEC_09 = AblationSpecTemplate<TI, true,  true, false,  true,  true, false,  true,  true, true>;
    using ABLATION_SPEC_10 = AblationSpecTemplate<TI, true,  true, false,  true,  true, false, false,  true, true>;
    using ABLATION_SPEC_11 = AblationSpecTemplate<TI, true,  true,  true,  true,  true,  true,  true,  true, false>;

    switch(ablation_id){
        case 0:
            train<ABLATION_SPEC_00>(run_id);
            break;
        case 1:
            train<ABLATION_SPEC_01>(run_id);
            break;
        case 2:
            train<ABLATION_SPEC_02>(run_id);
            break;
        case 3:
            train<ABLATION_SPEC_03>(run_id);
            break;
        case 4:
            train<ABLATION_SPEC_04>(run_id);
            break;
        case 5:
            train<ABLATION_SPEC_05>(run_id);
            break;
        case 6:
            train<ABLATION_SPEC_06>(run_id);
            break;
        case 7:
            train<ABLATION_SPEC_07>(run_id);
            break;
        case 8:
            train<ABLATION_SPEC_08>(run_id);
            break;
        case 9:
            train<ABLATION_SPEC_09>(run_id);
            break;
        case 10:
            train<ABLATION_SPEC_10>(run_id);
            break;
        case 11:
            train<ABLATION_SPEC_11>(run_id);
            break;
        default:
            std::cout << "Invalid ablation id: " << ablation_id << std::endl;
            return 1;
    }

    return 0;
}
