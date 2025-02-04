#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_EXTRACK_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_EXTRACK_OPERATIONS_CPU_H


#include "extrack.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <filesystem>

#include <cstdlib>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename TI>
    void init(DEVICE& device, utils::extrack::Config<TI>& config, utils::extrack::Paths& paths, typename DEVICE::index_t seed = 0){
        if(paths.experiment.empty()){
            utils::assert_exit(device, !config.base_path.empty(), "Extrack base path (-e,--extrack) must be set if the Extrack experiment path (--ee,--extrack-experiment) is not set.");
            if(config.experiment.empty()){

                auto environment_extrack_experiment = std::getenv("RL_TOOLS_EXTRACK_EXPERIMENT");
                if(environment_extrack_experiment != nullptr){
                    config.experiment = std::string(environment_extrack_experiment);
                } else {
                    config.experiment = utils::extrack::get_timestamp_string();
                }
            }
            paths.experiment = config.base_path / config.experiment;
        }
        {
            std::string commit_hash = RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH);
            std::string setup_name = commit_hash.substr(0, 7) + "_" + config.name;
            if(!config.population_variates.empty()){
                setup_name = setup_name + "_" + config.population_variates;
            }
            paths.setup = paths.experiment / setup_name;
            paths.config = paths.setup / config.population_values;
        }
        {
            std::stringstream padded_seed_ss;
            padded_seed_ss << std::setw(4) << std::setfill('0') << seed;
            paths.seed = paths.config / padded_seed_ss.str();
        }
        std::cerr << "Seed: " << seed << std::endl;
        std::cerr << "Extrack Experiment: " << paths.seed << std::endl;
#ifdef RL_TOOLS_EXTRACK_GIT_DIFF
        std::filesystem::path git_path = paths.seed / "git";
        std::filesystem::create_directories(git_path);
        {
            std::ofstream diff_file(git_path / "commit.txt");
            diff_file << rl_tools::utils::extrack::git::commit;
            diff_file.close();
        }
        {
            std::ofstream diff_file(git_path / "diff.txt");
            diff_file << rl_tools::utils::extrack::git::diff;
            diff_file.close();
        }
        {
            std::ofstream diff_file(git_path / "diff_color.txt");
            diff_file << rl_tools::utils::extrack::git::diff_color;
            diff_file.close();
        }
        {
            std::ofstream diff_file(git_path / "word_diff.txt");
            diff_file << rl_tools::utils::extrack::git::word_diff;
            diff_file.close();
        }
        {
            std::ofstream diff_file(git_path / "word_diff_color.txt");
            diff_file << rl_tools::utils::extrack::git::word_diff_color;
            diff_file.close();
        }
        {
            std::ofstream diff_file(git_path / "diff_staged.txt");
            diff_file << rl_tools::utils::extrack::git::diff_staged;
            diff_file.close();
        }
        {
            std::ofstream diff_file(git_path / "diff_staged_color.txt");
            diff_file << rl_tools::utils::extrack::git::diff_staged_color;
            diff_file.close();
        }
        {
            std::ofstream diff_file(git_path / "word_diff_staged.txt");
            diff_file << rl_tools::utils::extrack::git::word_diff_staged;
            diff_file.close();
        }
        {
            std::ofstream diff_file(git_path / "word_diff_staged_color.txt");
            diff_file << rl_tools::utils::extrack::git::word_diff_staged_color;
            diff_file.close();
        }
#endif

    }
    template <typename DEVICE, typename TI>
    std::filesystem::path get_step_folder(DEVICE& device, utils::extrack::Config<TI>& config, utils::extrack::Paths& paths, typename DEVICE::index_t step){
        std::stringstream step_ss;
        step_ss << std::setw(config.step_width) << std::setfill('0') << step;
        std::filesystem::path step_folder = paths.seed / "steps" / step_ss.str();
        std::filesystem::create_directories(step_folder);
        return step_folder;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
