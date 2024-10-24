#define MUX
#ifdef RL_TOOLS_ENABLE_TRACY
#include "Tracy.hpp"
#endif
//#define RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
#ifdef MUX
#include <rl_tools/operations/cpu_mux.h>
#else
#include <rl_tools/operations/cpu.h>
#endif
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/embedding/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#ifdef MUX
#include <rl_tools/nn/operations_cpu_mux.h>
#else
#include <rl_tools/nn/operations_cpu.h>
#endif
#include <rl_tools/nn/loss_functions/categorical_cross_entropy/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/optimizers/adam/instance/persist.h>
#include <rl_tools/nn/layers/embedding/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include "dataset.h"

namespace rlt = rl_tools;

#include "gru_model.h"


#include <chrono>
#include <filesystem>


#ifdef MUX
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
#else
using DEVICE = rlt::devices::DefaultCPU;
#endif
using TI = typename DEVICE::index_t;
using T = float;

using CONFIG = Config<T, TI>;

int main(){
    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);

    std::string data_path, file_name;
    bool use_enwik8 = true;
    if(use_enwik8){
        file_name = "enwik8.small.zip";
    }
    else{
        file_name = "00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json.small.gzip";
    }
    if(std::filesystem::exists("/Users")) {
        data_path = "/Users/jonas/Downloads/" + file_name;
    }
    else{
        if(std::filesystem::exists("/mnt/c/Users")) {
            data_path = "/mnt/c/Users/Jonas/Downloads/" + file_name;
        }
        else {
            data_path = "/home/jonas/Downloads/" + file_name;
        }
    }
    if(!std::filesystem::exists(data_path)){
        std::cerr << "Data path does not exist: " << data_path << std::endl;
        return 1;
    }
    std::string dataset_string;
    if(use_enwik8){
        dataset_string = load_dataset_enwik8<TI>(data_path);
    }
    else{
        dataset_string = load_dataset_wikipedia_plaintext<TI>(data_path);
    }

    std::vector<std::tuple<std::string, std::string>> dataset;
    for(TI offset=0; offset < dataset_string.size() - CONFIG::PARAMS::SEQUENCE_LENGTH - 1; offset++){
        auto input = dataset_string.substr(offset, CONFIG::PARAMS::SEQUENCE_LENGTH);
        auto output = dataset_string.substr(offset+1, CONFIG::PARAMS::SEQUENCE_LENGTH);
        dataset.emplace_back(std::tuple(input, output));
    }
    std::shuffle(dataset.begin(), dataset.end(), rng);
    std::cout << "Dataset size: " << dataset.size() << std::endl;
    std::cout << "Dataset sample: " << std::endl;
    for(TI i=0; i < 10; i++){
        std::cout << std::get<0>(dataset[i]) << " -> " << std::get<1>(dataset[i]) << std::endl;
    }

    typename CONFIG::MODEL model;
    typename CONFIG::MODEL::Buffer<> buffer;
    typename CONFIG::ADAM optimizer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, CONFIG::MODEL::INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, CONFIG::MODEL::OUTPUT_SHAPE>> d_output;
    rlt::Tensor<typename CONFIG::OUTPUT_TARGET_SPEC> output_target;
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, d_output);
    rlt::malloc(device, output_target);
    rlt::init_weights(device, model, rng);
    rlt::reset_optimizer_state(device, optimizer, model);
    std::cout << "INPUT SHAPE";
    rlt::print(device, decltype(input)::SPEC::SHAPE{});
    std::cout << std::endl;
    for(TI epoch_i=0; epoch_i < 1000; epoch_i++){
        std::shuffle(dataset.begin(), dataset.end(), rng);
        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_print = start_time;
        for(TI sample_i=0; sample_i < dataset.size() - CONFIG::PARAMS::BATCH_SIZE; sample_i += CONFIG::PARAMS::BATCH_SIZE){
#ifdef RL_TOOLS_ENABLE_TRACY
            FrameMark;
#endif
            if(sample_i % 10000 == 0){
                //checkpoint
                std::filesystem::path FILE_PATH = "model_checkpoint.h5";
                {
                    std::cout << "Checkpointing" << std::endl;
                    auto file = HighFive::File(FILE_PATH, HighFive::File::Overwrite);
                    rlt::zero_gradient(device, model);
                    rlt::reset_forward_state(device, model);
                    rlt::save(device, model, file.createGroup("checkpoint"));
                }
                if(sample_i == 0 || sample_i == CONFIG::PARAMS::BATCH_SIZE){ // reload check
                    auto file = HighFive::File(FILE_PATH, HighFive::File::ReadOnly);
                    CONFIG::MODEL model_copy;
                    rlt::malloc(device, model_copy);
                    rlt::load(device, model_copy, file.getGroup("checkpoint"));
                    T abs_diff = rlt::abs_diff(device, model, model_copy);
                    rlt::utils::assert_exit(device, abs_diff < 1e-6, "Checkpoint failed");
                    rlt::free(device, model_copy);
                }
            }

            for(TI batch_i = 0; batch_i < CONFIG::PARAMS::BATCH_SIZE; batch_i++){
                for(TI sequence_i = 0; sequence_i < CONFIG::PARAMS::SEQUENCE_LENGTH; sequence_i++){
                    rlt::set(device, input, std::get<0>(dataset[sample_i + batch_i])[sequence_i], sequence_i, batch_i, 0);
                    rlt::set(device, output_target, std::get<1>(dataset[sample_i + batch_i])[sequence_i], sequence_i, batch_i, 0);
                }
            }
            {
#ifdef RL_TOOLS_ENABLE_TRACY
                ZoneScopedN("forward");
#endif
                rlt::forward(device, model, input, buffer, rng);
            }
            auto output_logits = rlt::output(device, model);
            auto output_logits_matrix_view = rlt::matrix_view(device, output_logits);
            auto output_target_matrix_view = rlt::matrix_view(device, output_target);
            auto d_output_matrix_view = rlt::matrix_view(device, d_output);
            {
#ifdef RL_TOOLS_ENABLE_TRACY
                ZoneScopedN("loss_gradient");
#endif
                rlt::nn::loss_functions::categorical_cross_entropy::gradient_tiled(device, output_logits_matrix_view, output_target_matrix_view, d_output_matrix_view);
            }
            T elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
            T elapsed_print = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - last_print).count() / 1000.0;
            if(elapsed_print > 0.2 || sample_i % 10000 == 0){
                T loss = rlt::nn::loss_functions::categorical_cross_entropy::evaluate(device, output_logits_matrix_view, output_target_matrix_view);
                last_print = std::chrono::high_resolution_clock::now();
                std::cout << "Epoch: " << epoch_i << " Sample: " << sample_i << " Batch: " << sample_i/CONFIG::PARAMS::BATCH_SIZE << " (" << sample_i/CONFIG::PARAMS::BATCH_SIZE/elapsed << " batch/s)" << " Loss: " << loss << std::endl;
            }
            rlt::zero_gradient(device, model);
            {
#ifdef RL_TOOLS_ENABLE_TRACY
                ZoneScopedN("backward");
#endif
                rlt::backward(device, model, input, d_output, buffer);
            }
            rlt::step(device, optimizer, model);
        }
    }
    return 0;
}
