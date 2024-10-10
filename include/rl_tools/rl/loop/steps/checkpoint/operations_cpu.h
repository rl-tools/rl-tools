#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_CHECKPOINT_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_CHECKPOINT_OPERATIONS_GENERIC_H

#include "config.h"

#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/containers/matrix/persist.h>
#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#endif

#ifdef RL_TOOLS_ENABLE_ZLIB
#include <zlib.h>
#include <cstring>
#endif

#include "../../../../containers/matrix/persist_code.h"
#include "../../../../containers/tensor/persist_code.h"
#include "../../../../nn/optimizers/adam/instance/persist_code.h"
#include "../../../../nn/parameters/persist_code.h"
#include "../../../../nn/layers/dense/persist_code.h"
#include "../../../../nn/layers/gru/persist_code.h"
#include "../../../../nn/layers/sample_and_squash/persist_code.h"
#include "../../../../nn_models/mlp/persist_code.h"
#include "../../../../nn_models/sequential/persist_code.h"

#include "../../../../utils/zlib/operations_cpu.h"

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

    namespace rl::loop::steps::checkpoint{
            template <auto BATCH_SIZE, typename DEVICE, typename CONFIG, typename ACTOR_TYPE, bool STORE_UNCOMPRESSED_ANYWAYS=true>
            void save_code(DEVICE& device, rl::loop::steps::checkpoint::State<CONFIG>& ts, std::string step_folder, ACTOR_TYPE& actor_forward){
                using T = typename CONFIG::T;
                using TI = typename DEVICE::index_t;
                typename ACTOR_TYPE::template Buffer<> actor_buffer;
                malloc(device, actor_buffer);
                auto actor_weights = rl_tools::save_code(device, actor_forward, std::string("rl_tools::checkpoint::actor"), true);
                std::stringstream output_ss;
                output_ss << actor_weights;
                {
                    Tensor<tensor::Specification<T, TI, tensor::Replace<typename ACTOR_TYPE::INPUT_SHAPE, BATCH_SIZE, 1>>> input;
                    Tensor<tensor::Specification<T, TI, tensor::Replace<typename ACTOR_TYPE::OUTPUT_SHAPE, BATCH_SIZE, 1>>> output;
                    malloc(device, input);
                    malloc(device, output);
                    auto rng_copy = ts.rng;
                    randn(device, input, rng_copy);
                    Mode<mode::Evaluation<>> mode;
                    evaluate(device, actor_forward, input, output, actor_buffer, rng_copy, mode);
                    output_ss << "\n" << save_code(device, input, std::string("rl_tools::checkpoint::example::input"), true);
                    output_ss << "\n" << save_code(device, output, std::string("rl_tools::checkpoint::example::output"), true);
                    free(device, input);
                    free(device, output);
                }
                output_ss << "\n" << "namespace rl_tools::checkpoint::meta{";
                output_ss << "\n" << "   " << "char name[] = \"" << step_folder << "\";";
                output_ss << "\n" << "   " << "char commit_hash[] = \"" << RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH) << "\";";
                output_ss << "\n" << "}";
                std::string output_string = output_ss.str();
                bool stored_compressed = false;
#ifdef RL_TOOLS_ENABLE_ZLIB
                {
                    std::filesystem::path checkpoint_code_path = std::filesystem::path(step_folder) / "checkpoint.h.gz";
                    std::vector<uint8_t> checkpoint_output;
                    if(!compress_zlib(output_string, checkpoint_output)){
                        std::cerr << "Error while compressing trajectories." << std::endl;
                        return;
                    }
                    std::ofstream actor_output_file(checkpoint_code_path, std::ios::binary);
                    actor_output_file.write(reinterpret_cast<const char*>(checkpoint_output.data()), checkpoint_output.size());
                    actor_output_file.close();
                    stored_compressed = true;
                };
#endif
                if(!stored_compressed || STORE_UNCOMPRESSED_ANYWAYS){
                    std::filesystem::path checkpoint_code_path = std::filesystem::path(step_folder) / "checkpoint.h";
                    std::cerr << "Checkpointing at step: " << ts.step << " to: " << checkpoint_code_path << std::endl;
                    std::ofstream actor_output_file(checkpoint_code_path);
                    actor_output_file << output_string;
                    actor_output_file.close();
                }

                free(device, actor_buffer);
            }
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
            auto& actor = get_actor(ts);
            using ACTOR_TYPE = typename CONFIG::NN::ACTOR_TYPE;
            static constexpr TI BATCH_SIZE = 13;
            using INPUT_SHAPE = tensor::Replace<typename ACTOR_TYPE::INPUT_SHAPE, BATCH_SIZE, 1>;
            using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename CONFIG::NN::ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, BATCH_SIZE>;
            using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<nn::capability::Forward<>>;
            EVALUATION_ACTOR_TYPE evaluation_actor;
            malloc(device, evaluation_actor);
            copy(device, device, actor, evaluation_actor);
//            using ACTOR_FORWARD_TYPE = nn_models::sequential::Build<nn::capability::Forward<>, typename ACTOR_TYPE::SPEC::ORIGINAL_ROOT, INPUT_SHAPE>;
//            using ACTOR_FORWARD_TYPE = ACTOR_TYPE;
//            ACTOR_FORWARD_TYPE actor_forward;
//            malloc(device, actor_forward);
//            copy(device, device, actor, actor_forward);
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
            std::cerr << "Checkpointing at step: " << ts.step << " to: " << checkpoint_path << std::endl;
            try{
                auto actor_file = HighFive::File(checkpoint_path.string(), HighFive::File::Overwrite);
                save(device, evaluation_actor, actor_file.createGroup("actor"));
            }
            catch(HighFive::Exception& e){
                std::cerr << "Error while saving actor at " + checkpoint_path.string() + ": " << e.what() << std::endl;
            }
#endif
            rl::loop::steps::checkpoint::save_code<BATCH_SIZE>(device, ts, step_folder.string(), evaluation_actor);
            free(device, evaluation_actor);

        }
        bool finished = step(device, static_cast<typename STATE::NEXT&>(ts));
        return finished;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
