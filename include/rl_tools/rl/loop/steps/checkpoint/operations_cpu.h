#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_CHECKPOINT_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_CHECKPOINT_OPERATIONS_GENERIC_H

#include "config.h"

#ifdef RL_TOOLS_ENABLE_ZLIB
#include <zlib.h>
#include <cstring>
#endif

#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
#include "../../../../persist/backends/hdf5/hdf5.h"
#else
#include "../../../../persist/backends/tar/operations_cpu.h"
#endif

#include "../../../../numeric_types/persist_code.h"
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
#include "../../../../utils/extrack/operations_cpu.h"

#include <filesystem>
#include <iostream>
#include <fstream>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG>
    void malloc(DEVICE& device, rl::loop::steps::checkpoint::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::checkpoint::State<T_CONFIG>;
        malloc(device, static_cast<typename STATE::NEXT&>(ts));
        malloc(device, ts.rng_checkpoint);
    }

    template <typename DEVICE, typename T_CONFIG>
    void free(DEVICE& device, rl::loop::steps::checkpoint::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::checkpoint::State<T_CONFIG>;
        free(device, static_cast<typename STATE::NEXT&>(ts));
        free(device, ts.rng_checkpoint);
    }
    template <typename DEVICE, typename T_CONFIG>
    void init(DEVICE& device, rl::loop::steps::checkpoint::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using STATE = rl::loop::steps::checkpoint::State<T_CONFIG>;
        init(device, static_cast<typename STATE::NEXT&>(ts), seed);
        init(device, ts.rng_checkpoint, seed);
        ts.checkpoint_this_step = false;
    }

    namespace rl::loop::steps::checkpoint{
        template <bool DYNAMIC_ALLOCATION, typename ENVIRONMENT, typename DEVICE, typename ACTOR_TYPE, typename RNG, bool STORE_UNCOMPRESSED_ANYWAYS=true>
        void save_code(DEVICE& device, const std::string step_folder, const std::filesystem::path& latest_folder, typename DEVICE::index_t step, ACTOR_TYPE& actor_forward, RNG& rng){
            using T = typename ACTOR_TYPE::TYPE_POLICY::DEFAULT;
            using TI = typename DEVICE::index_t;
            std::filesystem::path step_folder_path = step_folder;
            auto actor_weights = rl_tools::save_code(device, actor_forward, std::string("rl_tools::checkpoint::actor"), true);
            std::stringstream output_ss;
            output_ss << actor_weights;
            { // note: this is duplicated for hdf5 (see futher down) and code
                Tensor<tensor::Specification<T, TI, typename ACTOR_TYPE::INPUT_SHAPE, DYNAMIC_ALLOCATION>> input;
                Tensor<tensor::Specification<T, TI, typename ACTOR_TYPE::OUTPUT_SHAPE, DYNAMIC_ALLOCATION>> output;
                typename ACTOR_TYPE::template Buffer<DYNAMIC_ALLOCATION> actor_buffer;
                malloc(device, input);
                malloc(device, output);
                malloc(device, actor_buffer);
                randn(device, input, rng);
                Mode<mode::Evaluation<>> mode;
                evaluate(device, actor_forward, input, output, actor_buffer, rng, mode);
                output_ss << "\n" << "namespace rl_tools::checkpoint::example::inputs{";
                output_ss << "\n" << save_code(device, input, std::string("_0"), true);
                output_ss << "\n" << "}";
                output_ss << "\n" << "namespace rl_tools::checkpoint::example::outputs{";
                output_ss << "\n" << save_code(device, output, std::string("_0"), true);
                output_ss << "\n" << "}";
                free(device, input);
                free(device, output);
                free(device, actor_buffer);
            }
            output_ss << "\n" << "namespace rl_tools::checkpoint::meta{";
            output_ss << "\n" << "   " << "char name[] = \"" << step_folder << "\";";
            output_ss << "\n" << "   " << "char commit_hash[] = \"" << RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH) << "\";";
            output_ss << "\n" << "}";
            { // Save environment
                ENVIRONMENT env;
                typename ENVIRONMENT::Parameters env_parameters;
                malloc(device, env);
                init(device, env);
                initial_parameters(device, env, env_parameters);
                auto env_info = save_code_env(device, env, env_parameters, "rl_tools::checkpoint::meta::environment");
                free(device, env);
                output_ss << "\n" << env_info;
            }
            std::string output_string = output_ss.str();
            bool stored_compressed = false;
#ifdef RL_TOOLS_ENABLE_ZLIB
            {
                std::filesystem::path checkpoint_code_path = step_folder_path / "checkpoint.h.gz";
                std::vector<uint8_t> checkpoint_output;
                if(!compress_zlib(output_string, checkpoint_output)){
                    std::cerr << "Error while compressing trajectories." << std::endl;
                    return;
                }
                std::ofstream actor_output_file(checkpoint_code_path, std::ios::binary);
                actor_output_file.write(reinterpret_cast<const char*>(checkpoint_output.data()), checkpoint_output.size());
                actor_output_file.close();
                link_latest_artifact(device, latest_folder, checkpoint_code_path, step_folder_path, step);
                stored_compressed = true;
            };
#endif
            if(!stored_compressed || STORE_UNCOMPRESSED_ANYWAYS){
                std::filesystem::create_directories(step_folder_path);
                std::filesystem::path checkpoint_code_path = step_folder_path / "checkpoint.h";
                std::cerr << "Checkpointing to: " << checkpoint_code_path << std::endl;
                std::ofstream actor_output_file(checkpoint_code_path);
                actor_output_file << output_string;
                actor_output_file.close();
                link_latest_artifact(device, latest_folder, checkpoint_code_path, step_folder_path, step);
            }

        }
        template <bool DYNAMIC_ALLOCATION, typename ENVIRONMENT, typename DEVICE, typename ACTOR_TYPE, typename RNG, bool STORE_UNCOMPRESSED_ANYWAYS=true>
        void save_code(DEVICE& device, const std::string step_folder, ACTOR_TYPE& actor_forward, RNG& rng){
            save_code<DYNAMIC_ALLOCATION, ENVIRONMENT, DEVICE, ACTOR_TYPE, RNG, STORE_UNCOMPRESSED_ANYWAYS>(device, step_folder, std::filesystem::path{}, 0, actor_forward, rng);
        }
        template <bool DYNAMIC_ALLOCATION, typename ENVIRONMENT, typename CHECKPOINT_PARAMETERS, typename DEVICE, typename ACTOR, typename RNG>
        void save(DEVICE& device, const std::string step_folder, const std::filesystem::path& latest_folder, typename DEVICE::index_t step, ACTOR& actor, RNG& rng){
            using TI = typename DEVICE::index_t;
            static constexpr TI BATCH_SIZE = CHECKPOINT_PARAMETERS::TEST_INPUT_BATCH_SIZE;
            using INPUT_SHAPE = tensor::Replace<typename ACTOR::INPUT_SHAPE, BATCH_SIZE, 1>;
            using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename ACTOR::template CHANGE_BATCH_SIZE<TI, BATCH_SIZE>;
            using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<nn::capability::Forward<DYNAMIC_ALLOCATION>>;
            EVALUATION_ACTOR_TYPE evaluation_actor;
            malloc(device, evaluation_actor);
            copy(device, device, actor, evaluation_actor);
            std::filesystem::path step_folder_path = step_folder;
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
            std::filesystem::path checkpoint_path = step_folder_path / "checkpoint.h5";
#else
            std::filesystem::path checkpoint_path = step_folder_path / "checkpoint.tar";
#endif
            auto save_checkpoint_contents = [&](auto& root, auto& actor_group){
                std::cerr << "Checkpointing to: " << checkpoint_path << std::endl;
                set_attribute(device, actor_group, "checkpoint_name", step_folder.c_str());
                ENVIRONMENT environment;
                auto env_description = json(device, environment);
                std::string meta = "{\"environment\": " + env_description + "}";
                set_attribute(device, actor_group, "meta", meta.c_str());
                rl_tools::save(device, evaluation_actor, actor_group);
                using T = typename EVALUATION_ACTOR_TYPE::TYPE_POLICY::DEFAULT;
                Tensor<tensor::Specification<T, TI, typename EVALUATION_ACTOR_TYPE::INPUT_SHAPE, DYNAMIC_ALLOCATION>> input;
                Tensor<tensor::Specification<T, TI, typename EVALUATION_ACTOR_TYPE::OUTPUT_SHAPE, DYNAMIC_ALLOCATION>> output;
                typename EVALUATION_ACTOR_TYPE::template Buffer<DYNAMIC_ALLOCATION> actor_buffer;
                malloc(device, input);
                malloc(device, output);
                malloc(device, actor_buffer);
                randn(device, input, rng);
                Mode<mode::Evaluation<>> mode;
                evaluate(device, actor, input, output, actor_buffer, rng, mode);
                auto example_group = create_group(device, root, "example");
                auto inputs_group = create_group(device, example_group, "inputs");
                save(device, input, inputs_group, "0");
                auto outputs_group = create_group(device, example_group, "outputs");
                save(device, output, outputs_group, "0");
                free(device, input);
                free(device, output);
                free(device, actor_buffer);
            };
            try{
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
                {
                    std::lock_guard<std::mutex> lock(persist::backends::hdf5::global_mutex());
                    persist::backends::hdf5::File root_file(checkpoint_path.string(), persist::backends::hdf5::Mode::WRITE);
                    auto actor_group = create_group(device, root_file, "actor");
                    save_checkpoint_contents(root_file, actor_group);
                }
#else
                persist::backends::tar::Writer writer;
                persist::backends::tar::WriterGroup<persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> root_group{"", &writer};
                auto actor_group = create_group(device, root_group, "actor");
                save_checkpoint_contents(root_group, actor_group);
                persist::backends::tar::finalize(device, writer);
                auto actor_file = std::ofstream(checkpoint_path, std::ios::binary);
                actor_file.write(writer.buffer.data(), writer.buffer.size());
                actor_file.close();
#endif
                link_latest_artifact(device, latest_folder, checkpoint_path, step_folder_path, step);
            }
            catch(std::exception& e){
                std::cerr << "Error while saving actor at " + checkpoint_path.string() + ": " << e.what() << std::endl;
            }
            rl::loop::steps::checkpoint::save_code<DYNAMIC_ALLOCATION, ENVIRONMENT>(device, step_folder, latest_folder, step, evaluation_actor, rng);
            free(device, evaluation_actor);

        }
        template <bool DYNAMIC_ALLOCATION, typename ENVIRONMENT, typename CHECKPOINT_PARAMETERS, typename DEVICE, typename ACTOR, typename RNG>
        void save(DEVICE& device, const std::string step_folder, ACTOR& actor, RNG& rng){
            save<DYNAMIC_ALLOCATION, ENVIRONMENT, CHECKPOINT_PARAMETERS>(device, step_folder, std::filesystem::path{}, 0, actor, rng);
        }
    }

    template <typename DEVICE, typename CONFIG>
    bool step(DEVICE& device, rl::loop::steps::checkpoint::State<CONFIG>& ts){
        using TYPE_POLICY = typename CONFIG::TYPE_POLICY;
        using TI = typename CONFIG::TI;
        using STATE = rl::loop::steps::checkpoint::State<CONFIG>;
        if(ts.step % CONFIG::CHECKPOINT_PARAMETERS::CHECKPOINT_INTERVAL == 0 || ts.checkpoint_this_step){
            ts.checkpoint_this_step = false;
            auto step_folder = get_step_folder(device, ts.extrack_config, ts.extrack_paths, ts.step);
            auto latest_folder = get_latest_folder(device, ts.extrack_paths);
            auto& actor = get_actor(ts);
            rl::loop::steps::checkpoint::save<CONFIG::DYNAMIC_ALLOCATION, typename CONFIG::ENVIRONMENT, typename CONFIG::CHECKPOINT_PARAMETERS>(device, step_folder.string(), latest_folder, ts.step, actor, ts.rng_checkpoint);
        }
        bool finished = step(device, static_cast<typename STATE::NEXT&>(ts));
        return finished;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
