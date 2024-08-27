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

TEST(RL_TOOLS_NN_LAYERS_GRU, MATRIX_MULTIPLICATION_TRANSPOSE_GENERIC){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = DEVICE::index_t;
    DEVICE device;
    using SHAPE = rlt::tensor::Shape<TI, 2, 2>;
    using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
    using SHAPE_TRANSPOSE = rlt::tensor::Shape<TI, rlt::get<1>(SHAPE{}), rlt::get<0>(SHAPE{})>;
    using STRIDE_TRANSPOSE = rlt::tensor::Stride<TI, rlt::get<1>(STRIDE{}), rlt::get<0>(STRIDE{})>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE>> A, B, C, C_target;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, STRIDE_TRANSPOSE>> A_T, B_T, C_T, C_target_T;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 2>>> bias;
    rlt::malloc(device, A);
    rlt::malloc(device, B);
    rlt::malloc(device, C);
    rlt::malloc(device, C_target);
    rlt::malloc(device, bias);
    rlt::set_all(device, bias, 0);
    rlt::set(device, bias, 1, 0);
    rlt::set(device, bias, 3, 1);
    rlt::data_reference(A_T) = rlt::data(A);
    rlt::data_reference(B_T) = rlt::data(B);
    rlt::data_reference(C_T) = rlt::data(C);
    rlt::data_reference(C_target_T) = rlt::data(C_target);

    rlt::set(device, A, -0.259093, 0, 0);
    rlt::set(device, A, -1.498961, 0, 1);
    rlt::set(device, A, +0.119264, 1, 0);
    rlt::set(device, A, +0.458181, 1, 1);

    rlt::set(device, B, +0.394975, 0, 0);
    rlt::set(device, B, +0.044197, 0, 1);
    rlt::set(device, B, -0.636256, 1, 0);
    rlt::set(device, B, +1.731264, 1, 1);

    rlt::set(device, C_target, -0.259093 * +0.394975 + -1.498961 * -0.636256 + 1, 0, 0);
    rlt::set(device, C_target, -0.259093 * +0.044197 + -1.498961 * +1.731264 + 1, 0, 1);
    rlt::set(device, C_target, +0.119264 * +0.394975 + +0.458181 * -0.636256 + 3, 1, 0);
    rlt::set(device, C_target, +0.119264 * +0.044197 + +0.458181 * +1.731264 + 3, 1, 1);
    rlt::print(device, C_target);

//    rlt::multiply(device, A, B, C);
    rlt::nn::layers::gru::helper::matrix_multiply_transpose_bias(device, A, B_T, bias, C_T);
    rlt::print(device, C);
    auto diff = rlt::abs_diff(device, C_target, C);
    std::cout << "Matrix mul diff: " << diff << std::endl;
    ASSERT_TRUE(diff < 1e-15);
    rlt::free(device, A);
    rlt::free(device, B);
    rlt::free(device, C);
    rlt::free(device, C_target);
    rlt::free(device, bias);
}

TEST(RL_TOOLS_NN_LAYERS_GRU, LOAD_GRU){
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
    rlt::Tensor<rlt::tensor::Specification<T, TI, GRU_OUTPUT_SHAPE>> gru_output, dloss_dgru_output;
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
    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
    rlt::nn::layers::gru::Layer<CAPABILITY, GRU_SPEC> gru;
    decltype(gru)::Buffer<BATCH_SIZE> buffer;
    rlt::malloc(device, gru);
    rlt::malloc(device, buffer);

    rlt::malloc(device, input);
    rlt::malloc(device, dinput);
    rlt::malloc(device, gru_output);
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


    rlt::init_weights(device, gru, rng);

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
            ASSERT_FALSE(rlt::is_nan(device, gru.weights_input.parameters));
            ASSERT_FALSE(rlt::is_nan(device, gru.weights_hidden.parameters));
            ASSERT_FALSE(rlt::is_nan(device, gru.biases_input.parameters));
            ASSERT_FALSE(rlt::is_nan(device, gru.biases_hidden.parameters));
            ASSERT_FALSE(rlt::is_nan(device, gru.initial_hidden_state.parameters));
            rlt::forward(device, gru, input, buffer, rng);
//            rlt::print(device, gru.output);
            T abs_diff = rlt::abs_diff(device, gru_output, gru.output) / (decltype(gru_output)::SPEC::SIZE);
            ASSERT_LT(abs_diff, 1e-15);
            std::cout << "abs_diff: " << abs_diff << std::endl;
//            auto dloss_dgru_output_view = rlt::view(device, dloss_dgru_output, 0, rlt::tensor::ViewSpec<1>{});
//            rlt::print(device, dloss_dgru_output_view);
            rlt::zero_gradient(device, gru);
            for(TI step=SEQUENCE_LENGTH-1; true; step--){
                auto backward_group = batch_group.getGroup("backward");
                auto gradient_group_step = backward_group.getGroup(std::to_string(step));
                rlt::load(device, grad_W_ir, gradient_group_step, "W_ir");
                rlt::load(device, grad_W_iz, gradient_group_step, "W_iz");
                rlt::load(device, grad_W_in, gradient_group_step, "W_in");
                rlt::load(device, grad_W_hr, gradient_group_step, "W_hr");
                rlt::load(device, grad_W_hz, gradient_group_step, "W_hz");
                rlt::load(device, grad_W_hn, gradient_group_step, "W_hn");
                rlt::load(device, grad_b_ir, gradient_group_step, "b_ir");
                rlt::load(device, grad_b_iz, gradient_group_step, "b_iz");
                rlt::load(device, grad_b_in, gradient_group_step, "b_in");
                rlt::load(device, grad_b_hr, gradient_group_step, "b_hr");
                rlt::load(device, grad_b_hz, gradient_group_step, "b_hz");
                rlt::load(device, grad_b_hn, gradient_group_step, "b_hn");

                rlt::_backward<true, true>(device, gru, input, dloss_dgru_output, dinput, buffer, step);


                std::cout << "Step: " << step << std::endl;
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
                if(step == 0){
                    break;
                }
            }

        }
    }
    rlt::free(device, gru);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, dinput);
    rlt::free(device, gru_output);
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


