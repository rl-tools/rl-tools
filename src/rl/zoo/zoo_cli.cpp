#ifdef RL_TOOLS_ENABLE_CLI11
#include <CLI/CLI.hpp>
#endif
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <rl_tools/utils/extrack/extrack.h>
int zoo(int, int, std::string, std::string, std::string, std::string);
int main(int argc, char** argv) {
    int seed = 0;
    int n_seeds = 1;
    int n_jobs = 1;
    std::string extrack_base_path = "";
    std::string extrack_experiment = rl_tools::utils::extrack::get_timestamp_string();
    std::string extrack_experiment_path = "";
    std::string config_path = "";
    #ifdef RL_TOOLS_ENABLE_CLI11
        CLI::App app{"rl_zoo"};
        app.add_option("-s,--seed", seed, "seed");
        app.add_option("-n,--n_seeds", n_seeds, "n_seeds");
        app.add_option("-j,--jobs", n_jobs, "number of parallel threads");
        app.add_option("--e,--extrack", extrack_base_path, "extrack");
        app.add_option("--ee,--extrack-experiment", extrack_experiment, "extrack-experiment");
        app.add_option("--eep,--extrack-experiment-path", extrack_experiment_path, "extrack-experiment-path");
        app.add_option("-c,--config", config_path, "config");
        CLI11_PARSE(app, argc, argv);
    #else
        if(argc > 1){
            seed = std::stoi(argv[1]);
            n_seeds = seed + 1;
        }
    #endif
    if(seed >= n_seeds){
        n_seeds = seed + 1;
    }
    int total_seeds = n_seeds - seed;
    if(n_jobs <= 1 || total_seeds <= 1){
        return zoo(seed, n_seeds, extrack_base_path, extrack_experiment, extrack_experiment_path, config_path);
    }
    n_jobs = std::min(n_jobs, total_seeds);
    std::vector<std::thread> threads;
    std::atomic<int> result{0};
    for(int j = 0; j < n_jobs; j++){
        int job_seed = seed + j;
        threads.emplace_back([=, &result](){
            for(int s = job_seed; s < n_seeds; s += n_jobs){
                int r = zoo(s, s + 1, extrack_base_path, extrack_experiment, extrack_experiment_path, config_path);
                if(r != 0) result.store(r);
            }
        });
    }
    for(auto& t : threads) t.join();
    return result.load();
}