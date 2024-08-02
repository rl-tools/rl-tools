#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/embedding/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/loss_functions/categorical_cross_entropy/operations_generic.h>
#include <rl_tools/nn_models/sequential_v2/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/optimizers/adam/instance/persist.h>
#include <rl_tools/nn/layers/embedding/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/sequential_v2/persist.h>
#include "dataset.h"

namespace rlt = rl_tools;

#include "gru_model.h"


#include <chrono>

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = typename DEVICE::index_t;
using T = float;
using CONFIG = Config<T, TI>;



int main() {
    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);



    typename CONFIG::MODEL model;
    typename CONFIG::MODEL::Buffer<CONFIG::BATCH_SIZE> buffer;
    rlt::Tensor<typename CONFIG::INPUT_SPEC> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename CONFIG::OUTPUT_SHAPE>> output;

    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    std::filesystem::path FILE_PATH = "model_checkpoint.h5";

    {
        auto file = HighFive::File(FILE_PATH, HighFive::File::ReadOnly);
        rlt::load(device, model, file.getGroup("checkpoint"));
    }

    std::string input_string;
//    std::getline(std::cin, input_string);
//    std::cout << "Input: " << input_string << std::endl;
    input_string = "The car is on the str";
    std::cout << input_string << std::flush;
    while(true){
        if(input_string.size() > CONFIG::SEQUENCE_LENGTH){
            input_string = input_string.substr(input_string.size() - CONFIG::SEQUENCE_LENGTH, CONFIG::SEQUENCE_LENGTH);
        }
        for(TI batch_i = 0; batch_i < CONFIG::BATCH_SIZE; batch_i++){
            for(TI sequence_i = 0; sequence_i < CONFIG::SEQUENCE_LENGTH; sequence_i++){
                if(sequence_i < input_string.size()) {
                    rlt::set(device, input, input_string[sequence_i], sequence_i, batch_i);
                }
            }
        }

        rlt::forward(device, model, input, buffer, rng);
        auto output_matrix = rlt::output(model);
        output._data = output_matrix._data;
        auto sequence_step = rlt::view(device, output, input_string.size()-1);
        auto logits = rlt::view(device, sequence_step, 0);
        T temperature = 0.5;
        rlt::scale(device, logits, 1/temperature);
        rlt::exp(device, logits);
        T sum = rlt::sum(device, logits);
        T comulative_prob = 0;
        T random_number = rlt::random::uniform_real_distribution(device.random, (T)0, (T)1, rng);
        for(TI i=0; i < CONFIG::NUM_CLASSES; i++){
            T prob = rlt::get(device, logits, i) / sum;
            if(random_number < comulative_prob + prob){
                input_string += (char)i;
                std::cout << (char)i << std::flush;
                break;
            }
            comulative_prob += prob;
            if(i == CONFIG::NUM_CLASSES - 1){
                input_string += (char)i;
                std::cout << (char)0 << std::flush;
            }
        }
//        rlt::print(device, logits);
//        auto logit_matrix = rlt::matrix_view(device, logits);
//        char next_token = rlt::argmax_row(device, logit_matrix);
//        input_string += next_token;
//        std::cout << input_string << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }



    return 0;
}
