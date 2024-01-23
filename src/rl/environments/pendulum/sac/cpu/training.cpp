#include "config.h"
#ifdef RL_TOOLS_ENABLE_HDF5
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#endif

#include <chrono>

using TrainingConfig = training_config::TrainingConfig;

int main(){
    using T = typename TrainingConfig::T;
    using TI = typename TrainingConfig::TI;
    using DEVICE = typename TrainingConfig::DEVICE;
    TI NUM_RUNS = 10;
#ifdef RL_TOOLS_ENABLE_HDF5
    std::string DATA_FILE_PATH = "rl_environments_pendulum_sac_learning_curves.h5";
    auto data_file = HighFive::File(DATA_FILE_PATH, HighFive::File::Overwrite);
#endif

    for(TI run_i = 0; run_i < NUM_RUNS; run_i++){
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Run: " << run_i << std::endl;
        rlt::rl::algorithms::sac::loop::TrainingState<TrainingConfig> ts;
        TI seed = run_i;
        rlt::rl::algorithms::sac::loop::init(ts, seed);
        ts.off_policy_runner.parameters.exploration_noise = 0;
#ifdef RL_TOOLS_ENABLE_HDF5
        auto run_group = data_file.createGroup(std::to_string(run_i));
#endif
        for(TI step_i=0; step_i < TrainingConfig::STEP_LIMIT; step_i++){
            rlt::rl::algorithms::sac::loop::step(ts);
//            if(step_i % 1000 == 0){
////                std::cout << "alpha: " << rlt::math::exp(DEVICE::SPEC::MATH{}, rlt::get(ts.actor_critic.log_alpha.parameters, 0, 0)) << std::endl;
//            }
        }
        std::vector<size_t> dims{decltype(ts)::N_EVALUATIONS};
        std::vector<T> mean_returns;
        for(TI i=0; i < decltype(ts)::N_EVALUATIONS; i++){
            mean_returns.push_back(ts.evaluation_results[i].returns_mean);
        }

#ifdef RL_TOOLS_ENABLE_HDF5
        run_group.createDataSet("episode_returns", mean_returns);
#endif

        rlt::rl::algorithms::sac::loop::destroy(ts);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Run time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0 << "s" << std::endl;
    }
    return 0;
}
