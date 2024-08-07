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

#include <gtest/gtest.h>

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


    auto gru = rlt::get_layer<0>(sequential);



    std::string DATA_FILE_NAME = "gru_training_trace.h5";
    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TESTS_DATA_PATH);
    std::string DATA_FILE_PATH = std::string(data_path_stub) + "/" + DATA_FILE_NAME;
    std::cout << "DATA_FILE_PATH: " << DATA_FILE_PATH << std::endl;
    auto output_file = HighFive::File(std::string(DATA_FILE_PATH), HighFive::File::ReadOnly);
    for(auto epoch_group_name : output_file.listObjectNames()){
        auto epoch_group = output_file.getGroup(epoch_group_name);
        for(auto batch_group_name: epoch_group.listObjectNames()){
            auto batch_group = epoch_group.getGroup(batch_group_name);
            rlt::load(device, input, batch_group, "input");
            rlt::load(device, output_target, batch_group, "output");
            rlt::load(device, gru_output, batch_group, "gru_output");
            auto weight_group = batch_group.getGroup("weights");
            auto gradient_group = batch_group.getGroup("gradient");
            auto dloss_dgru_output_ds = batch_group.getDataSet("d_loss_d_y_pred_gru");
            rlt::load(device, gru.W_ir, weight_group, "W_ir");
            rlt::load(device, gru.W_iz, weight_group, "W_iz");
            rlt::load(device, gru.W_in, weight_group, "W_in");
            rlt::load(device, gru.W_hr, weight_group, "W_hr");
            rlt::load(device, gru.W_hz, weight_group, "W_hz");
            rlt::load(device, gru.W_hn, weight_group, "W_hn");
            rlt::load(device, gru.b_ir, weight_group, "b_ir");
            rlt::load(device, gru.b_iz, weight_group, "b_iz");
            rlt::load(device, gru.b_in, weight_group, "b_in");
            rlt::load(device, gru.b_hr, weight_group, "b_hr");
            rlt::load(device, gru.b_hz, weight_group, "b_hz");
            rlt::load(device, gru.b_hn, weight_group, "b_hn");
            rlt::load(device, weight_out, weight_group, "W_out");
            rlt::load(device, bias_out, weight_group, "b_out");
            rlt::load(device, dloss_dgru_output, batch_group, "d_loss_d_y_pred_gru");
            {
                rlt::forward(device, sequential, input, buffers, rng);
//                rlt::print(device, gru.output);
                T abs_diff = rlt::abs_diff(device, gru_output, gru.output) / (decltype(gru_output)::SPEC::SIZE);
                std::cout << "abs_diff: " << abs_diff << std::endl;
                ASSERT_LT(abs_diff, 1e-15);
            }
            {
                rlt::evaluate(device, sequential, input, gru_output_evaluate, buffers, rng);
//                rlt::print(device, gru_output_evaluate);
                T abs_diff = rlt::abs_diff(device, gru_output, gru_output_evaluate) / (decltype(gru_output)::SPEC::SIZE);
                ASSERT_LT(abs_diff, 1e-15);
            }
            //            auto dloss_dgru_output_view = rlt::view(device, dloss_dgru_output, 0, rlt::tensor::ViewSpec<1>{});
            //            rlt::print(device, dloss_dgru_output_view);
            rlt::zero_gradient(device, gru);
            rlt::backward_full(device, sequential, input, dloss_dgru_output, dinput, buffers);
            rlt::load(device, grad_W_ir, gradient_group, "W_ir");
            rlt::load(device, grad_W_iz, gradient_group, "W_iz");
            rlt::load(device, grad_W_in, gradient_group, "W_in");
            rlt::load(device, grad_W_hr, gradient_group, "W_hr");
            rlt::load(device, grad_W_hz, gradient_group, "W_hz");
            rlt::load(device, grad_W_hn, gradient_group, "W_hn");
            rlt::load(device, grad_b_ir, gradient_group, "b_ir");
            rlt::load(device, grad_b_iz, gradient_group, "b_iz");
            rlt::load(device, grad_b_in, gradient_group, "b_in");
            rlt::load(device, grad_b_hr, gradient_group, "b_hr");
            rlt::load(device, grad_b_hz, gradient_group, "b_hz");
            rlt::load(device, grad_b_hn, gradient_group, "b_hn");
            auto grad_W_hr_view = rlt::view_range(device, gru.weights_hidden.gradient, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_hr = rlt::abs_diff(device, grad_W_hr_view, grad_W_hr)/decltype(grad_W_hr)::SPEC::SIZE;
            std::cout << "abs_diff_W_hr: " << abs_diff_W_hr << std::endl;

            auto grad_b_hr_view = rlt::view_range(device, gru.biases_hidden.gradient, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_hr = rlt::abs_diff(device, grad_b_hr_view, grad_b_hr)/decltype(grad_b_hr)::SPEC::SIZE;
            std::cout << "abs_diff_b_hr: " << abs_diff_b_hr << std::endl;

            auto grad_W_hz_view = rlt::view_range(device, gru.weights_hidden.gradient, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_hz = rlt::abs_diff(device, grad_W_hz_view, grad_W_hz)/decltype(grad_W_hz)::SPEC::SIZE;
            std::cout << "abs_diff_W_hz: " << abs_diff_W_hz << std::endl;

            auto grad_b_hz_view = rlt::view_range(device, gru.biases_hidden.gradient, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_hz = rlt::abs_diff(device, grad_b_hz_view, grad_b_hz)/decltype(grad_b_hz)::SPEC::SIZE;
            std::cout << "abs_diff_b_hz: " << abs_diff_b_hz << std::endl;

            auto grad_W_hn_view = rlt::view_range(device, gru.weights_hidden.gradient, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_hn = rlt::abs_diff(device, grad_W_hn_view, grad_W_hn)/decltype(grad_W_hn)::SPEC::SIZE;
            std::cout << "abs_diff_W_hn: " << abs_diff_W_hn << std::endl;

            auto grad_b_hn_view = rlt::view_range(device, gru.biases_hidden.gradient, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_hn = rlt::abs_diff(device, grad_b_hn_view, grad_b_hn)/decltype(grad_b_hn)::SPEC::SIZE;
            std::cout << "abs_diff_b_hn: " << abs_diff_b_hn << std::endl;

            auto grad_W_ir_view = rlt::view_range(device, gru.weights_input.gradient, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_ir = rlt::abs_diff(device, grad_W_ir_view, grad_W_ir)/decltype(grad_W_ir)::SPEC::SIZE;
            std::cout << "abs_diff_W_ir: " << abs_diff_W_ir << std::endl;

            auto grad_b_ir_view = rlt::view_range(device, gru.biases_input.gradient, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_ir = rlt::abs_diff(device, grad_b_ir_view, grad_b_hr)/decltype(grad_b_ir)::SPEC::SIZE;
            std::cout << "abs_diff_b_ir: " << abs_diff_b_ir << std::endl;

            auto grad_W_iz_view = rlt::view_range(device, gru.weights_input.gradient, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_iz = rlt::abs_diff(device, grad_W_iz_view, grad_W_iz)/decltype(grad_W_iz)::SPEC::SIZE;
            std::cout << "abs_diff_W_iz: " << abs_diff_W_iz << std::endl;

            auto grad_b_iz_view = rlt::view_range(device, gru.biases_input.gradient, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_iz = rlt::abs_diff(device, grad_b_iz_view, grad_b_iz)/decltype(grad_b_iz)::SPEC::SIZE;
            std::cout << "abs_diff_b_iz: " << abs_diff_b_iz << std::endl;

            auto grad_W_in_view = rlt::view_range(device, gru.weights_input.gradient, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_W_in = rlt::abs_diff(device, grad_W_in_view, grad_W_in)/decltype(grad_W_in)::SPEC::SIZE;
            std::cout << "abs_diff_W_in: " << abs_diff_W_in << std::endl;

            auto grad_b_in_view = rlt::view_range(device, gru.biases_input.gradient, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
            T abs_diff_b_in = rlt::abs_diff(device, grad_b_in_view, grad_b_in)/decltype(grad_b_in)::SPEC::SIZE;
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
    rlt::free(device, sequential);
    rlt::free(device, buffers);
    rlt::free(device, input);
    rlt::free(device, dinput);
    rlt::free(device, gru_output);
    rlt::free(device, gru_output_evaluate);
    rlt::free(device, dloss_dgru_output);
    rlt::free(device, output_target);
    rlt::free(device, weight_out);
    rlt::free(device, weight_out_grad);
    rlt::free(device, bias_out);
    rlt::free(device, bias_out_grad);
    rlt::free(device, grad_W_ir);
    rlt::free(device, grad_W_iz);
    rlt::free(device, grad_W_in);
    rlt::free(device, grad_b_ir);
    rlt::free(device, grad_b_iz);
    rlt::free(device, grad_b_in);
    rlt::free(device, grad_W_hr);
    rlt::free(device, grad_W_hz);
    rlt::free(device, grad_W_hn);
    rlt::free(device, grad_b_hr);
    rlt::free(device, grad_b_hz);
    rlt::free(device, grad_b_hn);
    rlt::free(device, dloss_dgru_output_step);
}
