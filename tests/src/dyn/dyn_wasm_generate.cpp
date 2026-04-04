#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>

#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/flatten/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>

#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)

namespace rlt = rl_tools;

#include <cstdio>
#include <string>

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

constexpr TI BATCH_SIZE = 2;
constexpr TI H = 4, W = 4, C = 3;

using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, H, W, C>;

using CONV_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 8, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
using CONV = rlt::nn::layers::conv2d::BindConfiguration<CONV_CONFIG>;
using CONV_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
using CONV_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<CONV_FLATTEN_CONFIG>;
using CONV_DENSE_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 10, rlt::nn::activation_functions::ActivationFunction::RELU>;
using CONV_DENSE = rlt::nn::layers::dense::BindConfiguration<CONV_DENSE_CONFIG>;
using CNN_BRANCH = rlt::nn_models::sequential::Module<CONV, CONV_FLATTEN, CONV_DENSE>;

using MLP_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
using MLP_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<MLP_FLATTEN_CONFIG>;
using MLP_DENSE1_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 16, rlt::nn::activation_functions::ActivationFunction::RELU>;
using MLP_DENSE1 = rlt::nn::layers::dense::BindConfiguration<MLP_DENSE1_CONFIG>;
using MLP_DENSE2_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 8, rlt::nn::activation_functions::ActivationFunction::RELU>;
using MLP_DENSE2 = rlt::nn::layers::dense::BindConfiguration<MLP_DENSE2_CONFIG>;
using MLP_BRANCH = rlt::nn_models::sequential::Module<MLP_FLATTEN, MLP_DENSE1, MLP_DENSE2>;

using HEAD_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 4, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
using HEAD = rlt::nn::layers::dense::BindConfiguration<HEAD_CONFIG>;

using MODEL = rlt::nn_models::parallel::Build<rlt::nn::capability::Forward<>, CNN_BRANCH, MLP_BRANCH, INPUT_SHAPE, INPUT_SHAPE, HEAD>;

int main(){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);

    MODEL model;
    typename MODEL::Buffer<> buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::OUTPUT_SHAPE>> output;
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);
    rlt::evaluate(device, model, input, input, output, buffer, rng);

    std::string path = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_wasm_checkpoint.h5";
    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::WRITE);
        auto model_group = rlt::create_group(device, file, "model");
        rlt::save(device, model, model_group);
        {
            constexpr TI N = 1 * BATCH_SIZE * H * W * C;
            hsize_t dims[] = {1, BATCH_SIZE, H, W, C};
            float flat[N];
            for(TI i = 0; i < N; i++) flat[i] = reinterpret_cast<float*>(data(input))[i];
            hid_t space = H5Screate_simple(5, dims, nullptr);
            hid_t ds = H5Dcreate2(file.id, "test_input", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat);
            H5Dclose(ds); H5Sclose(space);
        }
        {
            auto output_mat = rlt::matrix_view(device, output);
            constexpr TI OUTPUT_DIM = 4;
            constexpr TI N = 1 * BATCH_SIZE * OUTPUT_DIM;
            float flat[N];
            for(TI i = 0; i < 1 * BATCH_SIZE; i++)
                for(TI j = 0; j < OUTPUT_DIM; j++)
                    flat[i * OUTPUT_DIM + j] = rlt::get(output_mat, i, j);
            hsize_t dims[] = {(hsize_t)N};
            hid_t space = H5Screate_simple(1, dims, nullptr);
            hid_t ds = H5Dcreate2(file.id, "expected_output", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat);
            H5Dclose(ds); H5Sclose(space);
        }
    }

    printf("Wrote %s\n", path.c_str());
    printf("  Parallel model: CNN branch + MLP branch -> dense head\n");
    printf("  Input: [1, %lu, %lu, %lu, %lu]\n", (unsigned long)BATCH_SIZE, (unsigned long)H, (unsigned long)W, (unsigned long)C);
    printf("  Output: [1, %lu, 4]\n", (unsigned long)BATCH_SIZE);

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, output);
    return 0;
}
