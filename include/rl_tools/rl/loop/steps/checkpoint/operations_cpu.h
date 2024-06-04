#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_CHECKPOINT_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_CHECKPOINT_OPERATIONS_GENERIC_H

#include "config.h"

#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/containers/persist.h>
#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#endif

#include <rl_tools/containers/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>

#include <filesystem>
#include <iostream>
#include <fstream>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG>
    void init(DEVICE& device, rl::loop::steps::checkpoint::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using STATE = rl::loop::steps::checkpoint::State<T_CONFIG>;
        init(device, static_cast<typename STATE::NEXT&>(ts), seed);
    }

    template <typename DEVICE, typename T_CONFIG>
    void free(DEVICE& device, rl::loop::steps::checkpoint::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::checkpoint::State<T_CONFIG>;
        free(device, static_cast<typename STATE::NEXT&>(ts));
    }

    template <typename DEVICE, typename CONFIG>
    bool step(DEVICE& device, rl::loop::steps::checkpoint::State<CONFIG>& ts){
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using STATE = rl::loop::steps::checkpoint::State<CONFIG>;
        if(ts.step % CONFIG::CHECKPOINT_PARAMETERS::CHECKPOINT_INTERVAL == 0){
            std::stringstream step_ss;
            step_ss << std::setw(15) << std::setfill('0') << ts.step;
            std::filesystem::path step_folder = ts.extrack_seed_path / "steps" / step_ss.str();
            std::filesystem::create_directories(step_folder);
            std::filesystem::path checkpoint_path = step_folder / "checkpoint.h5";
            std::cerr << "Checkpointing at step: " << ts.step << " to: " << checkpoint_path << std::endl;
            auto& actor = get_actor(ts);
            using ACTOR_TYPE = typename CONFIG::NN::ACTOR_TYPE;
            using ACTOR_FORWARD_TYPE = typename ACTOR_TYPE::template CHANGE_CAPABILITY<nn::layer_capability::Forward>;
//            using ACTOR_FORWARD_TYPE = ACTOR_TYPE;
            ACTOR_FORWARD_TYPE actor_forward;
            malloc(device, actor_forward);
#if defined(RL_TOOLS_ENABLE_HDF5) and !defined(RL_TOOLS_DISABLE_HDF5)
            try{
                auto actor_file = HighFive::File(checkpoint_path.string(), HighFive::File::Overwrite);
                copy(device, device, actor, actor_forward);
                save(device, actor_forward, actor_file.createGroup("actor"));
            }
            catch(HighFive::Exception& e){
                std::cerr << "Error while saving actor at " + checkpoint_path.string() + ": " << e.what() << std::endl;
            }
#endif
            typename ACTOR_TYPE::template Buffer<1> actor_buffer;
            malloc(device, actor_buffer);
            copy(device, device, actor, actor_forward);
            std::filesystem::path checkpoint_code_path = step_folder / "checkpoint.h";
            auto actor_weights = save_code(device, actor_forward, std::string("rl_tools::checkpoint::actor"), true);
            std::ofstream actor_output_file(checkpoint_code_path);
            actor_output_file << actor_weights;
            {
                MatrixStatic<matrix::Specification<T, TI, 1, ACTOR_TYPE::INPUT_DIM>> input;
                MatrixStatic<matrix::Specification<T, TI, 1, ACTOR_TYPE::OUTPUT_DIM>> output;
                auto rng_copy = ts.rng;
                randn(device, input, rng_copy);
                evaluate(device, actor, input, output, actor_buffer, rng_copy);
                actor_output_file << "\n" << save_code(device, input, std::string("rl_tools::checkpoint::example::input"), true);
                actor_output_file << "\n" << save_code(device, output, std::string("rl_tools::checkpoint::example::output"), true);
                free(device, input);
                free(device, output);
            }
            free(device, actor_buffer);
            free(device, actor_forward);

            actor_output_file << "\n" << "namespace rl_tools::checkpoint::meta{";
            actor_output_file << "\n" << "   " << "char name[] = \"" << step_folder.string() << "\";";
            actor_output_file << "\n" << "   " << "char commit_hash[] = \"" << RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH) << "\";";
            actor_output_file << "\n" << "}";

        }
        bool finished = step(device, static_cast<typename STATE::NEXT&>(ts));
        return finished;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
