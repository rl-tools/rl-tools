#include <gtest/gtest.h>
#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>
#include <rl_tools/containers/tensor/persist.h>

namespace rlt = rl_tools;


#include "../utils/utils.h"

using DEVICE = rlt::devices::DefaultCPU;
using T = double;
using TI = DEVICE::index_t;

constexpr T EPSILON = 1e-6;
constexpr TI SEQUENCE_LENGTH = 50;
constexpr TI BATCH_SIZE = 128;
constexpr TI INPUT_DIM = 1;
constexpr TI OUTPUT_DIM = 1;
constexpr TI HIDDEN_DIM = 16;
using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, INPUT_DIM>;
rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
using OUTPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, OUTPUT_DIM>;
rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> output;
using WI_SHAPE = rlt::tensor::Shape<TI, HIDDEN_DIM*3, INPUT_DIM>;
rlt::Tensor<rlt::tensor::Specification<T, TI, WI_SHAPE>> weight_in, weight_in_grad;
using BI_SHAPE = rlt::tensor::Shape<TI, HIDDEN_DIM*3>;
rlt::Tensor<rlt::tensor::Specification<T, TI, BI_SHAPE>> bias_in, bias_in_grad;
using WH_SHAPE = rlt::tensor::Shape<TI, HIDDEN_DIM*3, HIDDEN_DIM>;
rlt::Tensor<rlt::tensor::Specification<T, TI, WH_SHAPE>> weight_hidden, weight_hidden_grad;
using BH_SHAPE = rlt::tensor::Shape<TI, HIDDEN_DIM*3>;
rlt::Tensor<rlt::tensor::Specification<T, TI, BH_SHAPE>> bias_hidden, bias_hidden_grad;
using WOUT_SHAPE = rlt::tensor::Shape<TI, OUTPUT_DIM, HIDDEN_DIM>;
rlt::Tensor<rlt::tensor::Specification<T, TI, WOUT_SHAPE>> weight_out, weight_out_grad;
using BOUT_SHAPE = rlt::tensor::Shape<TI, OUTPUT_DIM>;
rlt::Tensor<rlt::tensor::Specification<T, TI, BOUT_SHAPE>> bias_out, bias_out_grad;

DEVICE device;


TEST(RL_TOOLS_NN_LAYERS_GRU, FULL_TRAINING) {

    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, weight_in);
    rlt::malloc(device, weight_in_grad);
    rlt::malloc(device, bias_in);
    rlt::malloc(device, bias_in_grad);
    rlt::malloc(device, weight_hidden);
    rlt::malloc(device, weight_hidden_grad);
    rlt::malloc(device, bias_hidden);
    rlt::malloc(device, bias_hidden_grad);
    rlt::malloc(device, weight_out);
    rlt::malloc(device, weight_out_grad);
    rlt::malloc(device, bias_out);
    rlt::malloc(device, bias_out_grad);

    auto W_ir = rlt::view_range(device, weight_in, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
    auto W_iz = rlt::view_range(device, weight_in, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
    auto W_in = rlt::view_range(device, weight_in, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});

    auto b_ir = rlt::view_range(device, bias_in, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
    auto b_iz = rlt::view_range(device, bias_in, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
    auto b_in = rlt::view_range(device, bias_in, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});

    auto W_hr = rlt::view_range(device, weight_hidden, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
    auto W_hz = rlt::view_range(device, weight_hidden, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
    auto W_hn = rlt::view_range(device, weight_hidden, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});

    auto b_hr = rlt::view_range(device, bias_hidden, 0*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
    auto b_hz = rlt::view_range(device, bias_hidden, 1*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});
    auto b_hn = rlt::view_range(device, bias_hidden, 2*HIDDEN_DIM, rlt::tensor::ViewSpec<0, HIDDEN_DIM>{});

    auto W_out = rlt::view_range(device, weight_out, 0*OUTPUT_DIM, rlt::tensor::ViewSpec<0, OUTPUT_DIM>{});
    auto b_out = rlt::view_range(device, bias_out, 0*OUTPUT_DIM, rlt::tensor::ViewSpec<0, OUTPUT_DIM>{});

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
            rlt::load(device, input, input_ds);
            rlt::load(device, output, output_ds);
            auto input_slice = rlt::view_range(device, input, 0, rlt::tensor::ViewSpec<0, 1>{});
            std::vector<std::vector<std::vector<T>>> input_data;
            input_ds.read(input_data);
            for(TI i=0; i < SEQUENCE_LENGTH; i++){
                for(TI j=0; j < BATCH_SIZE; j++){
                    for(TI k=0; k < INPUT_DIM; k++){
                        T diff = rlt::math::abs(device.math, static_cast<T>(input_data[i][j][k]) - static_cast<T>(rlt::get(device, input, i, j, k)));
                        ASSERT_LT(diff, EPSILON);
                    }
                }
            }
            std::vector<std::vector<std::vector<T>>> output_data;
            output_ds.read(output_data);
            for(TI i=0; i < SEQUENCE_LENGTH; i++){
                for(TI j=0; j < BATCH_SIZE; j++){
                    for(TI k=0; k < OUTPUT_DIM; k++){
                        T diff = rlt::math::abs(device.math, static_cast<T>(output_data[i][j][k]) - static_cast<T>(rlt::get(device, output, i, j, k)));
                        ASSERT_LT(diff, EPSILON);
                    }
                }
            }
            auto weight_group = batch_group.getGroup("weights");
            auto gradient_group = batch_group.getGroup("gradient");
            auto W_ir_ds = weight_group.getDataSet("W_ir");
            rlt::load(device, W_ir, W_ir_ds);
            std::vector<std::vector<T>> W_ir_data;
            W_ir_ds.read(W_ir_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                for(TI j=0; j < INPUT_DIM; j++){
                    T diff = rlt::math::abs(device.math, static_cast<T>(W_ir_data[i][j]) - static_cast<T>(rlt::get(device, W_ir, i, j)));
                    ASSERT_LT(diff, EPSILON);
                }
            }
            auto W_iz_ds = weight_group.getDataSet("W_iz");
            rlt::load(device, W_iz, W_iz_ds);
            std::vector<std::vector<T>> W_iz_data;
            W_iz_ds.read(W_iz_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                for(TI j=0; j < INPUT_DIM; j++){
                    T diff = rlt::math::abs(device.math, static_cast<T>(W_iz_data[i][j]) - static_cast<T>(rlt::get(device, W_iz, i, j)));
                    ASSERT_LT(diff, EPSILON);
                }
            }
            auto W_in_ds = weight_group.getDataSet("W_in");
            rlt::load(device, W_in, W_in_ds);
            std::vector<std::vector<T>> W_in_data;
            W_in_ds.read(W_in_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                for(TI j=0; j < INPUT_DIM; j++){
                    T diff = rlt::math::abs(device.math, static_cast<T>(W_in_data[i][j]) - static_cast<T>(rlt::get(device, W_in, i, j)));
                    ASSERT_LT(diff, EPSILON);
                }
            }
            auto W_hr_ds = weight_group.getDataSet("W_hr");
            rlt::load(device, W_hr, W_hr_ds);
            std::vector<std::vector<T>> W_hr_data;
            W_hr_ds.read(W_hr_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                for(TI j=0; j < HIDDEN_DIM; j++){
                    T diff = rlt::math::abs(device.math, static_cast<T>(W_hr_data[i][j]) - static_cast<T>(rlt::get(device, W_hr, i, j)));
                    ASSERT_LT(diff, EPSILON);
                }
            }
            auto W_hz_ds = weight_group.getDataSet("W_hz");
            rlt::load(device, W_hz, W_hz_ds);
            std::vector<std::vector<T>> W_hz_data;
            W_hz_ds.read(W_hz_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                for(TI j=0; j < HIDDEN_DIM; j++){
                    T diff = rlt::math::abs(device.math, static_cast<T>(W_hz_data[i][j]) - static_cast<T>(rlt::get(device, W_hz, i, j)));
                    ASSERT_LT(diff, EPSILON);
                }
            }
            auto W_hn_ds = weight_group.getDataSet("W_hn");
            rlt::load(device, W_hn, W_hn_ds);
            std::vector<std::vector<T>> W_hn_data;
            W_hn_ds.read(W_hn_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                for(TI j=0; j < HIDDEN_DIM; j++){
                    T diff = rlt::math::abs(device.math, static_cast<T>(W_hn_data[i][j]) - static_cast<T>(rlt::get(device, W_hn, i, j)));
                    ASSERT_LT(diff, EPSILON);
                }
            }
            auto b_ir_ds = weight_group.getDataSet("b_ir");
            rlt::load(device, b_ir, b_ir_ds);
            std::vector<T> b_ir_data;
            b_ir_ds.read(b_ir_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                T diff = rlt::math::abs(device.math, static_cast<T>(b_ir_data[i]) - static_cast<T>(rlt::get(device, b_ir, i)));
                ASSERT_LT(diff, EPSILON);
            }
            auto b_iz_ds = weight_group.getDataSet("b_iz");
            rlt::load(device, b_iz, b_iz_ds);
            std::vector<T> b_iz_data;
            b_iz_ds.read(b_iz_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                T diff = rlt::math::abs(device.math, static_cast<T>(b_iz_data[i]) - static_cast<T>(rlt::get(device, b_iz, i)));
                ASSERT_LT(diff, EPSILON);
            }
            auto b_in_ds = weight_group.getDataSet("b_in");
            rlt::load(device, b_in, b_in_ds);
            std::vector<T> b_in_data;
            b_in_ds.read(b_in_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                T diff = rlt::math::abs(device.math, static_cast<T>(b_in_data[i]) - static_cast<T>(rlt::get(device, b_in, i)));
                ASSERT_LT(diff, EPSILON);
            }
            auto b_hr_ds = weight_group.getDataSet("b_hr");
            rlt::load(device, b_hr, b_hr_ds);
            std::vector<T> b_hr_data;
            b_hr_ds.read(b_hr_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                T diff = rlt::math::abs(device.math, static_cast<T>(b_hr_data[i]) - static_cast<T>(rlt::get(device, b_hr, i)));
                ASSERT_LT(diff, EPSILON);
            }
            auto b_hz_ds = weight_group.getDataSet("b_hz");
            rlt::load(device, b_hz, b_hz_ds);
            std::vector<T> b_hz_data;
            b_hz_ds.read(b_hz_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                T diff = rlt::math::abs(device.math, static_cast<T>(b_hz_data[i]) - static_cast<T>(rlt::get(device, b_hz, i)));
                ASSERT_LT(diff, EPSILON);
            }
            auto b_hn_ds = weight_group.getDataSet("b_hn");
            rlt::load(device, b_hn, b_hn_ds);
            std::vector<T> b_hn_data;
            b_hn_ds.read(b_hn_data);
            for(TI i=0; i < HIDDEN_DIM; i++){
                T diff = rlt::math::abs(device.math, static_cast<T>(b_hn_data[i]) - static_cast<T>(rlt::get(device, b_hn, i)));
                ASSERT_LT(diff, EPSILON);
            }
            auto W_out_ds = weight_group.getDataSet("W_out");
            rlt::load(device, W_out, W_out_ds);
            std::vector<std::vector<T>> W_out_data;
            W_out_ds.read(W_out_data);
            for(TI i=0; i < OUTPUT_DIM; i++){
                for(TI j=0; j < HIDDEN_DIM; j++){
                    T diff = rlt::math::abs(device.math, static_cast<T>(W_out_data[i][j]) - static_cast<T>(rlt::get(device, W_out, i, j)));
                    ASSERT_LT(diff, EPSILON);
                }
            }
            auto b_out_ds = weight_group.getDataSet("b_out");
            rlt::load(device, b_out, b_out_ds);
            std::vector<T> b_out_data;
            b_out_ds.read(b_out_data);
            for(TI i=0; i < OUTPUT_DIM; i++){
                T diff = rlt::math::abs(device.math, static_cast<T>(b_out_data[i]) - static_cast<T>(rlt::get(device, b_out, i)));
                ASSERT_LT(diff, EPSILON);
            }
        }
    }
}
