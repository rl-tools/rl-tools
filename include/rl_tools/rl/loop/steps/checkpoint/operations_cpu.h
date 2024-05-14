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
        using TI = typename CONFIG::TI;
        using STATE = rl::loop::steps::checkpoint::State<CONFIG>;
        if(ts.step % CONFIG::PARAMETERS::CHECKPOINT_INTERVAL == 0){
            std::stringstream step_ss;
            step_ss << std::setw(15) << std::setfill('0') << ts.step;
            std::filesystem::path step_folder = ts.extrack_seed_path / "steps" / step_ss.str();
            std::cerr << "Checkpointing at step: " << ts.step << " to: " << step_folder << std::endl;
            std::filesystem::create_directories(step_folder);
            std::filesystem::path checkpoint_filename = step_folder / "checkpoint.h5";
#if defined(RL_TOOLS_ENABLE_HDF5) and !defined(RL_TOOLS_DISABLE_HDF5)
            try{
                auto actor_file = HighFive::File(checkpoint_filename.string(), HighFive::File::Overwrite);
                save(device, get_actor(ts), actor_file.createGroup("actor"));
            }
            catch(HighFive::Exception& e){
                std::cerr << "Error while saving actor at " + checkpoint_filename.string() + ": " << e.what() << std::endl;
            }
#endif
        }
        bool finished = step(device, static_cast<typename STATE::NEXT&>(ts));
        return finished;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
