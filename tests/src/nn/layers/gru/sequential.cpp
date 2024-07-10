#include <gtest/gtest.h>
#include <rl_tools/operations/cpu.h>

#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>
#include <rl_tools/containers/tensor/persist.h>

#include <rl_tools/nn/layers/gru/operations_generic.h>

#include <rl_tools/nn_models/sequential_v2/operations_generic.h>

//#include <rl_tools/nn/layers/gru/layer.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>

namespace rlt = rl_tools;


#include "../../../utils/utils.h"

TEST(RL_TOOLS_NN_LAYERS_GRU, SEQUENTIAL_V2){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = DEVICE::index_t;
    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);
    constexpr T EPSILON = 1e-6;
    constexpr TI SEQUENCE_LENGTH = 50;
    constexpr TI BATCH_SIZE = 128;
    constexpr TI INPUT_DIM = 1;
    constexpr TI OUTPUT_DIM = 1;
    constexpr TI HIDDEN_DIM = 16;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, INPUT_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input, dinput;
    using GRU_OUTPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU_OUTPUT_SHAPE>> gru_output, gru_output_evaluate, dloss_dgru_output;
    using GRU_OUTPUT_STEP_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, HIDDEN_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU_OUTPUT_STEP_SHAPE>> dloss_dgru_output_step;
    using OUTPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, OUTPUT_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output_target;
    using WOUT_SHAPE = rlt::tensor::Shape<TI, OUTPUT_DIM, HIDDEN_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, WOUT_SHAPE>> weight_out, weight_out_grad;
    using BOUT_SHAPE = rlt::tensor::Shape<TI, OUTPUT_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, BOUT_SHAPE>> bias_out, bias_out_grad;
    using GRU_INPUT_WEIGHT_SHAPE = rlt::tensor::Shape<TI, HIDDEN_DIM, INPUT_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU_INPUT_WEIGHT_SHAPE>> grad_W_ir, grad_W_iz, grad_W_in;
    using GRU_INPUT_BIAS_SHAPE = rlt::tensor::Shape<TI, HIDDEN_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU_INPUT_BIAS_SHAPE>> grad_b_ir, grad_b_iz, grad_b_in;
    using GRU_HIDDEN_WEIGHT_SHAPE = rlt::tensor::Shape<TI, HIDDEN_DIM, HIDDEN_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU_HIDDEN_WEIGHT_SHAPE>> grad_W_hr, grad_W_hz, grad_W_hn;
    using GRU_HIDDEN_BIAS_SHAPE = rlt::tensor::Shape<TI, HIDDEN_DIM>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU_HIDDEN_BIAS_SHAPE>> grad_b_hr, grad_b_hz, grad_b_hn;

    using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, INPUT_DIM, HIDDEN_DIM, rlt::nn::parameters::Gradient>;
    using GRU_LAYER = rlt::nn::layers::gru::BindSpecification<GRU_SPEC>;
    using namespace rlt::nn_models::sequential_v2;
    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
    using MODEL = Module<CAPABILITY, GRU_LAYER::Layer>;
    MODEL sequential;
    MODEL::Buffer<BATCH_SIZE> buffers;

    rlt::malloc(device, sequential);
    rlt::malloc(device, buffers);

    auto gru = rlt::get_layer<0>(sequential);


    rlt::malloc(device, input);
    rlt::malloc(device, dinput);
    rlt::malloc(device, gru_output);
    rlt::malloc(device, gru_output_evaluate);
    rlt::malloc(device, dloss_dgru_output);
    rlt::malloc(device, output_target);
    rlt::malloc(device, weight_out);
    rlt::malloc(device, weight_out_grad);
    rlt::malloc(device, bias_out);
    rlt::malloc(device, bias_out_grad);
    rlt::malloc(device, grad_W_ir);
    rlt::malloc(device, grad_W_iz);
    rlt::malloc(device, grad_W_in);
    rlt::malloc(device, grad_b_ir);
    rlt::malloc(device, grad_b_iz);
    rlt::malloc(device, grad_b_in);
    rlt::malloc(device, grad_W_hr);
    rlt::malloc(device, grad_W_hz);
    rlt::malloc(device, grad_W_hn);
    rlt::malloc(device, grad_b_hr);
    rlt::malloc(device, grad_b_hz);
    rlt::malloc(device, grad_b_hn);
    rlt::malloc(device, dloss_dgru_output_step);

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
            auto gru_output_ds = batch_group.getDataSet("gru_output");
            rlt::load(device, input_ds, input);
            rlt::load(device, output_ds, output_target);
            rlt::load(device, gru_output_ds, gru_output);
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
            auto dloss_dgru_output_ds = batch_group.getDataSet("d_loss_d_y_pred_gru");
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
            rlt::load(device, dloss_dgru_output_ds, dloss_dgru_output);
//            {
//                rlt::forward(device, gru, input);
//                rlt::print(device, gru.output);
//                T abs_diff = rlt::absolute_difference(device, gru_output, gru.output) / (decltype(gru_output)::SPEC::SIZE);
//                std::cout << "abs_diff: " << abs_diff << std::endl;
//                ASSERT_LT(abs_diff, 1e-15);
//            }
            {
                rlt::evaluate(device, sequential, input, gru_output_evaluate, buffers, rng);
//                rlt::print(device, gru_output_evaluate);
                T abs_diff = rlt::absolute_difference(device, gru_output, gru_output_evaluate) / (decltype(gru_output)::SPEC::SIZE);
                ASSERT_LT(abs_diff, 1e-15);
            }
            //            auto dloss_dgru_output_view = rlt::view(device, dloss_dgru_output, 0, rlt::tensor::ViewSpec<1>{});
            //            rlt::print(device, dloss_dgru_output_view);
            rlt::zero_gradient(device, gru);
            rlt::backward_full(device, sequential, input, dloss_dgru_output, dinput, buffers);
            auto grad_W_ir_ds = gradient_group.getDataSet("W_ir");
            auto grad_W_iz_ds = gradient_group.getDataSet("W_iz");
            auto grad_W_in_ds = gradient_group.getDataSet("W_in");
            auto grad_W_hr_ds = gradient_group.getDataSet("W_hr");
            auto grad_W_hz_ds = gradient_group.getDataSet("W_hz");
            auto grad_W_hn_ds = gradient_group.getDataSet("W_hn");
            auto grad_b_ir_ds = gradient_group.getDataSet("b_ir");
            auto grad_b_iz_ds = gradient_group.getDataSet("b_iz");
            auto grad_b_in_ds = gradient_group.getDataSet("b_in");
            auto grad_b_hr_ds = gradient_group.getDataSet("b_hr");
            auto grad_b_hz_ds = gradient_group.getDataSet("b_hz");
            auto grad_b_hn_ds = gradient_group.getDataSet("b_hn");
            rlt::load(device, grad_W_ir_ds, grad_W_ir);
            rlt::load(device, grad_W_iz_ds, grad_W_iz);
            rlt::load(device, grad_W_in_ds, grad_W_in);
            rlt::load(device, grad_W_hr_ds, grad_W_hr);
            rlt::load(device, grad_W_hz_ds, grad_W_hz);
            rlt::load(device, grad_W_hn_ds, grad_W_hn);
            rlt::load(device, grad_b_ir_ds, grad_b_ir);
            rlt::load(device, grad_b_iz_ds, grad_b_iz);
            rlt::load(device, grad_b_in_ds, grad_b_in);
            rlt::load(device, grad_b_hr_ds, grad_b_hr);
            rlt::load(device, grad_b_hz_ds, grad_b_hz);
            rlt::load(device, grad_b_hn_ds, grad_b_hn);
            auto grad_W_hr_view = rlt::view_range(device, gru.weights_hidden.gradient, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_hr = rlt::absolute_difference(device, grad_W_hr_view, grad_W_hr)/decltype(grad_W_hr)::SPEC::SIZE;
            std::cout << "abs_diff_W_hr: " << abs_diff_W_hr << std::endl;

            auto grad_b_hr_view = rlt::view_range(device, gru.biases_hidden.gradient, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_hr = rlt::absolute_difference(device, grad_b_hr_view, grad_b_hr)/decltype(grad_b_hr)::SPEC::SIZE;
            std::cout << "abs_diff_b_hr: " << abs_diff_b_hr << std::endl;

            auto grad_W_hz_view = rlt::view_range(device, gru.weights_hidden.gradient, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_hz = rlt::absolute_difference(device, grad_W_hz_view, grad_W_hz)/decltype(grad_W_hz)::SPEC::SIZE;
            std::cout << "abs_diff_W_hz: " << abs_diff_W_hz << std::endl;

            auto grad_b_hz_view = rlt::view_range(device, gru.biases_hidden.gradient, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_hz = rlt::absolute_difference(device, grad_b_hz_view, grad_b_hz)/decltype(grad_b_hz)::SPEC::SIZE;
            std::cout << "abs_diff_b_hz: " << abs_diff_b_hz << std::endl;

            auto grad_W_hn_view = rlt::view_range(device, gru.weights_hidden.gradient, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_hn = rlt::absolute_difference(device, grad_W_hn_view, grad_W_hn)/decltype(grad_W_hn)::SPEC::SIZE;
            std::cout << "abs_diff_W_hn: " << abs_diff_W_hn << std::endl;

            auto grad_b_hn_view = rlt::view_range(device, gru.biases_hidden.gradient, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_hn = rlt::absolute_difference(device, grad_b_hn_view, grad_b_hn)/decltype(grad_b_hn)::SPEC::SIZE;
            std::cout << "abs_diff_b_hn: " << abs_diff_b_hn << std::endl;

            auto grad_W_ir_view = rlt::view_range(device, gru.weights_input.gradient, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_ir = rlt::absolute_difference(device, grad_W_ir_view, grad_W_ir)/decltype(grad_W_ir)::SPEC::SIZE;
            std::cout << "abs_diff_W_ir: " << abs_diff_W_ir << std::endl;

            auto grad_b_ir_view = rlt::view_range(device, gru.biases_input.gradient, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_ir = rlt::absolute_difference(device, grad_b_ir_view, grad_b_hr)/decltype(grad_b_ir)::SPEC::SIZE;
            std::cout << "abs_diff_b_ir: " << abs_diff_b_ir << std::endl;

            auto grad_W_iz_view = rlt::view_range(device, gru.weights_input.gradient, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_iz = rlt::absolute_difference(device, grad_W_iz_view, grad_W_iz)/decltype(grad_W_iz)::SPEC::SIZE;
            std::cout << "abs_diff_W_iz: " << abs_diff_W_iz << std::endl;

            auto grad_b_iz_view = rlt::view_range(device, gru.biases_input.gradient, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_iz = rlt::absolute_difference(device, grad_b_iz_view, grad_b_iz)/decltype(grad_b_iz)::SPEC::SIZE;
            std::cout << "abs_diff_b_iz: " << abs_diff_b_iz << std::endl;

            auto grad_W_in_view = rlt::view_range(device, gru.weights_input.gradient, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_in = rlt::absolute_difference(device, grad_W_in_view, grad_W_in)/decltype(grad_W_in)::SPEC::SIZE;
            std::cout << "abs_diff_W_in: " << abs_diff_W_in << std::endl;

            auto grad_b_in_view = rlt::view_range(device, gru.biases_input.gradient, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_in = rlt::absolute_difference(device, grad_b_in_view, grad_b_in)/decltype(grad_b_in)::SPEC::SIZE;
            std::cout << "abs_diff_b_in: " << abs_diff_b_in << std::endl;


            ASSERT_LT(abs_diff_W_hr, EPSILON);
            ASSERT_LT(abs_diff_b_hr, EPSILON);
            ASSERT_LT(abs_diff_W_hz, EPSILON);
            ASSERT_LT(abs_diff_b_hz, EPSILON);
            ASSERT_LT(abs_diff_W_hn, EPSILON);
            ASSERT_LT(abs_diff_b_hn, EPSILON);

            ASSERT_LT(abs_diff_W_ir, EPSILON);
            ASSERT_LT(abs_diff_b_ir, EPSILON);
            ASSERT_LT(abs_diff_W_iz, EPSILON);
            ASSERT_LT(abs_diff_b_iz, EPSILON);
            ASSERT_LT(abs_diff_W_in, EPSILON);
            ASSERT_LT(abs_diff_b_in, EPSILON);
        }
    }
    rlt::free(device, input);
    rlt::free(device, gru_output);
    rlt::free(device, output_target);
    rlt::free(device, weight_out);
    rlt::free(device, weight_out_grad);
    rlt::free(device, bias_out);
    rlt::free(device, bias_out_grad);
    rlt::free(device, gru);
}
