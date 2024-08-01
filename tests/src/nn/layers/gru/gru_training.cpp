#ifdef RL_TOOLS_ENABLE_TRACY
#include "Tracy.hpp"
#endif
#define RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/embedding/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/loss_functions/categorical_cross_entropy/operations_generic.h>
#include <rl_tools/nn_models/sequential_v2/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include "dataset.h"


#include <chrono>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = typename DEVICE::index_t;
using T = float;

constexpr TI NUM_CLASSES = 2<<7;
constexpr TI EMBEDDING_DIM = 32;
constexpr TI BATCH_SIZE = 32;
constexpr TI SEQUENCE_LENGTH = 64;
constexpr TI HIDDEN_DIM = 64;
constexpr TI OUTPUT_DIM = NUM_CLASSES;

template <TI BATCH_SIZE>
using INPUT_SHAPE_TEMPLATE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE>;

int main() {
    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);

    std::string data_path = "/Users/jonas/Downloads/00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json.gzip";
    std::string dataset_string = load_dataset<TI>(data_path);
    std::vector<std::tuple<std::string, std::string>> dataset;
    for(TI offset=0; offset < dataset_string.size() - SEQUENCE_LENGTH - 1; offset++){
        auto input = dataset_string.substr(offset, SEQUENCE_LENGTH);
        auto output = dataset_string.substr(offset+1, SEQUENCE_LENGTH);
        dataset.emplace_back(std::tuple(input, output));
    }
    std::shuffle(dataset.begin(), dataset.end(), rng);
    std::cout << "Dataset size: " << dataset.size() << std::endl;
    std::cout << "Dataset sample: " << std::endl;
    for(TI i=0; i < 10; i++){
        std::cout << std::get<0>(dataset[i]) << " -> " << std::get<1>(dataset[i]) << std::endl;
    }


    using INPUT_SHAPE = INPUT_SHAPE_TEMPLATE<BATCH_SIZE>;
    using INPUT_SPEC = rlt::tensor::Specification<unsigned char, TI, INPUT_SHAPE>;

    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;

    using EMBEDDING_LAYER_SPEC = rlt::nn::layers::embedding::Specification<T, TI, NUM_CLASSES, EMBEDDING_DIM, INPUT_SHAPE_TEMPLATE>;
    using EMBEDDING_LAYER_TEMPLATE = rlt::nn::layers::embedding::BindSpecification<EMBEDDING_LAYER_SPEC>;

//    EMBEDDING_LAYER embedding_layer;
//    EMBEDDING_LAYER::Buffer<BATCH_SIZE> embedding_buffer;

    using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, EMBEDDING_DIM, HIDDEN_DIM, rlt::nn::parameters::Gradient>;
    using GRU_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU_SPEC>;
//    decltype(gru)::Buffer<BATCH_SIZE> gru_buffer;

    using DENSE_LAYER_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, OUTPUT_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, rlt::nn::parameters::groups::Normal, rlt::MatrixDynamicTag, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
    using DENSE_LAYER_TEMPLATE = rlt::nn::layers::dense::BindSpecification<DENSE_LAYER_SPEC>;
//    DENSE_LAYER dense_layer;
//    DENSE_LAYER::Buffer<BATCH_SIZE> dense_buffer;

    using IF = rlt::nn_models::sequential_v2::Interface<CAPABILITY>;
    using MODEL = IF::Module<EMBEDDING_LAYER_TEMPLATE::Layer, IF::Module<GRU_TEMPLATE::Layer, IF::Module<DENSE_LAYER_TEMPLATE::Layer>>>;
    MODEL model;
    MODEL::Buffer<BATCH_SIZE> buffer;


//    using EMBEDDING_OUTPUT_SPEC = rlt::tensor::Specification<T, TI, decltype(embedding_layer)::OUTPUT_SHAPE>;
    using OUTPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, OUTPUT_DIM>;
    using OUTPUT_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>;
//    using GRU_OUTPUT_SPEC = rlt::tensor::Specification<T, TI, decltype(gru)::OUTPUT_SHAPE>;
    using OUTPUT_TARGET_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, 1>;
    using OUTPUT_TARGET_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_TARGET_SHAPE>;
    using ADAM_SPEC = rlt::nn::optimizers::adam::Specification<T, TI>;
    using ADAM = rlt::nn::optimizers::Adam<ADAM_SPEC>;



    ADAM optimizer;
    rlt::Tensor<INPUT_SPEC> input;
//    rlt::Tensor<EMBEDDING_OUTPUT_SPEC> d_embedding_output;
    rlt::Tensor<OUTPUT_SPEC> d_output;
//    rlt::Tensor<GRU_OUTPUT_SPEC> d_gru_output;
    rlt::Tensor<OUTPUT_TARGET_SPEC> output_target;
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
//    rlt::malloc(device, embedding_layer);
//    rlt::malloc(device, embedding_buffer);
//    rlt::malloc(device, gru);
//    rlt::malloc(device, gru_buffer);
//    rlt::malloc(device, dense_layer);
//    rlt::malloc(device, dense_buffer);
    rlt::malloc(device, input);
//    rlt::malloc(device, d_embedding_output);
    rlt::malloc(device, d_output);
//    rlt::malloc(device, d_gru_output);
    rlt::malloc(device, output_target);
//    rlt::init_weights(device, embedding_layer, rng);
//    rlt::init_weights(device, gru, rng);
//    rlt::init_weights(device, dense_layer, rng);
    rlt::init_weights(device, model, rng);
//    rlt::reset_optimizer_state(device, optimizer, embedding_layer);
//    rlt::reset_optimizer_state(device, optimizer, gru);
    rlt::reset_optimizer_state(device, optimizer, model);
//    rlt::print(device, embedding_layer.weights.parameters);
    for(TI epoch_i=0; epoch_i < 10; epoch_i++){
        std::shuffle(dataset.begin(), dataset.end(), rng);
        auto start_time = std::chrono::high_resolution_clock::now();
        for(TI sample_i=0; sample_i < dataset.size(); sample_i += BATCH_SIZE){
            FrameMark;
            for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                for(TI sequence_i = 0; sequence_i < SEQUENCE_LENGTH; sequence_i++){
                    rlt::set(device, input, std::get<0>(dataset[sample_i + batch_i])[sequence_i], sequence_i, batch_i);
                    rlt::set(device, output_target, std::get<1>(dataset[sample_i + batch_i])[sequence_i], sequence_i, batch_i, 0);
                }
            }
//            rlt::forward(device, embedding_layer, input, embedding_buffer, rng);
//            rlt::forward(device, gru, rlt::output(embedding_layer), gru_buffer, rng);
//            auto hidden_state = rlt::matrix_view(device, rlt::output(gru));
//            rlt::forward(device, dense_layer, hidden_state, dense_buffer, rng);
            {
                ZoneScopedN("forward");
                rlt::forward(device, model, input, buffer, rng);
            }
            auto output_logits = rlt::output(model);
//            auto output_logits_matrix_view = rlt::matrix_view(device, output_logits);
            auto output_target_matrix_view = rlt::matrix_view(device, output_target);
            auto d_output_matrix_view = rlt::matrix_view(device, d_output);
            rlt::nn::loss_functions::categorical_cross_entropy::gradient(device, output_logits, output_target_matrix_view, d_output_matrix_view);
            T loss = rlt::nn::loss_functions::categorical_cross_entropy::evaluate(device, output_logits, output_target_matrix_view);
            T elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
            std::cout << "Sample: " << sample_i << " Batch: " << sample_i/BATCH_SIZE << "(" << sample_i/BATCH_SIZE/elapsed << " batch/s)" << " Loss: " << loss << std::endl;
            rlt::zero_gradient(device, model);
//            rlt::zero_gradient(device, gru);
//            rlt::zero_gradient(device, dense_layer);
            rlt::zero_gradient(device, model);
//            auto d_gru_output_matrix_view = rlt::matrix_view(device, d_gru_output);
//            rlt::backward_full(device, dense_layer, hidden_state, d_output_matrix_view, d_gru_output_matrix_view, dense_buffer);
//            rlt::backward_full(device, gru, rlt::output(embedding_layer), d_gru_output, d_embedding_output, gru_buffer);
            {
                ZoneScopedN("backward");
                rlt::backward(device, model, input, d_output, buffer);
            }
//            rlt::step(device, optimizer, embedding_layer);
//            rlt::step(device, optimizer, gru);
            rlt::step(device, optimizer, model);
        }
    }

    return 0;
}
