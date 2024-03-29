#include <gtest/gtest.h>
#include <rl_tools/operations/cpu.h>

#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>
#include <rl_tools/containers/tensor/persist.h>

#include <rl_tools/nn/layers/gru/operations_generic.h>

//#include <rl_tools/nn/layers/gru/layer.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>

namespace rlt = rl_tools;


#include "../../../utils/utils.h"


TEST(RL_TOOLS_NN_LAYERS_GRU, LOAD_GRU){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = DEVICE::index_t;
    DEVICE device;
    constexpr T EPSILON = 1e-6;
    constexpr TI SEQUENCE_LENGTH = 50;
    constexpr TI BATCH_SIZE = 128;
    constexpr TI INPUT_DIM = 1;
    constexpr TI OUTPUT_DIM = 1;
    constexpr TI HIDDEN_DIM = 16;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, INPUT_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    using GRU_OUTPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU_OUTPUT_SHAPE>> output;
    using OUTPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, OUTPUT_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_target;
    using WOUT_SHAPE = rlt::tensor::Shape<TI, OUTPUT_DIM, HIDDEN_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, WOUT_SHAPE>> weight_out, weight_out_grad;
    using BOUT_SHAPE = rlt::tensor::Shape<TI, OUTPUT_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, BOUT_SHAPE>> bias_out, bias_out_grad;

    using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, INPUT_DIM, HIDDEN_DIM, rlt::nn::parameters::Plain, BATCH_SIZE>;
    rlt::nn::layers::gru::LayerBackward<GRU_SPEC> gru;
    rlt::malloc(device, gru);

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, output_target);
    rlt::malloc(device, weight_out);
    rlt::malloc(device, weight_out_grad);
    rlt::malloc(device, bias_out);
    rlt::malloc(device, bias_out_grad);

    std::string DATA_FILE_NAME = "gru_training_trace.h5";
    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TESTS_DATA_PATH);
    std::string DATA_FILE_PATH = std::string(data_path_stub) + "/" + DATA_FILE_NAME;
    std::cout << "DATA_FILE_PATH: " << DATA_FILE_PATH << std::endl;
    auto output_file = HighFive::File(std::string(DATA_FILE_PATH), HighFive::File::ReadOnly);
    for(auto epoch_group_name : output_file.listObjectNames()){
        auto epoch_group = output_file.getGroup(epoch_group_name);
        for(auto batch_group_name: epoch_group.listObjectNames()){
            auto batch_group = epoch_group.getGroup(batch_group_name);
            auto input_ds = batch_group.getDataSet("input");
            auto output_ds = batch_group.getDataSet("output");
            rlt::load(device, input_ds, input);
            rlt::load(device, output_ds, output_target);
            auto weight_group = batch_group.getGroup("weights");
            auto gradient_group = batch_group.getGroup("gradient");
            auto W_ir_ds = weight_group.getDataSet("W_ir");
            auto W_iz_ds = weight_group.getDataSet("W_iz");
            auto W_in_ds = weight_group.getDataSet("W_in");
            auto W_hr_ds = weight_group.getDataSet("W_hr");
            auto W_hz_ds = weight_group.getDataSet("W_hz");
            auto W_hn_ds = weight_group.getDataSet("W_hn");
            auto b_ir_ds = weight_group.getDataSet("b_ir");
            auto b_iz_ds = weight_group.getDataSet("b_iz");
            auto b_in_ds = weight_group.getDataSet("b_in");
            auto b_hr_ds = weight_group.getDataSet("b_hr");
            auto b_hz_ds = weight_group.getDataSet("b_hz");
            auto b_hn_ds = weight_group.getDataSet("b_hn");
            auto W_out_ds = weight_group.getDataSet("W_out");
            auto b_out_ds = weight_group.getDataSet("b_out");
            rlt::load(device, W_ir_ds, gru.W_ir);
            rlt::load(device, W_iz_ds, gru.W_iz);
            rlt::load(device, W_in_ds, gru.W_in);
            rlt::load(device, W_hr_ds, gru.W_hr);
            rlt::load(device, W_hz_ds, gru.W_hz);
            rlt::load(device, W_hn_ds, gru.W_hn);
            rlt::load(device, b_ir_ds, gru.b_ir);
            rlt::load(device, b_iz_ds, gru.b_iz);
            rlt::load(device, b_in_ds, gru.b_in);
            rlt::load(device, b_hr_ds, gru.b_hr);
            rlt::load(device, b_hz_ds, gru.b_hz);
            rlt::load(device, b_hn_ds, gru.b_hn);
            rlt::load(device, W_out_ds, weight_out);
            rlt::load(device, b_out_ds, bias_out);
            rlt::forward(device, gru, input);
        }
    }
    rlt::free(device, input);
    rlt::free(device, output);
    rlt::free(device, weight_out);
    rlt::free(device, weight_out_grad);
    rlt::free(device, bias_out);
    rlt::free(device, bias_out_grad);
    rlt::free(device, gru);
}
