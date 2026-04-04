#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>

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

int main(){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);

    constexpr TI INPUT_DIM = 8;
    constexpr TI HIDDEN_DIM = 16;
    constexpr TI OUTPUT_DIM = 4;
    constexpr TI BATCH_SIZE = 3;
    constexpr TI NUM_LAYERS = 3;

    using MLP_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, NUM_LAYERS, HIDDEN_DIM,
        rlt::nn::activation_functions::ActivationFunction::RELU,
        rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, INPUT_DIM>;
    using MLP_TYPE = rlt::nn_models::mlp::NeuralNetwork<MLP_CONFIG, rlt::nn::capability::Forward<>, INPUT_SHAPE>;

    MLP_TYPE model;
    typename MLP_TYPE::Buffer<> buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MLP_TYPE::OUTPUT_SHAPE>> output;
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::init_weights(device, model, rng);
    rlt::randn(device, input, rng);
    rlt::evaluate(device, model, input, output, buffer, rng);

    std::string path = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_wasm_checkpoint.h5";
    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::WRITE);
        auto model_group = rlt::create_group(device, file, "model");
        rlt::save(device, model, model_group);
        rlt::persist::backends::hdf5::detail::write_string_attribute(file.id, "input_dim", std::to_string(INPUT_DIM).c_str());
        rlt::persist::backends::hdf5::detail::write_string_attribute(file.id, "batch_size", std::to_string(BATCH_SIZE).c_str());
        auto input_mat = rlt::matrix_view(device, input);
        {
            constexpr TI N = BATCH_SIZE * INPUT_DIM;
            float flat[N];
            for(TI i = 0; i < BATCH_SIZE; i++)
                for(TI j = 0; j < INPUT_DIM; j++)
                    flat[i * INPUT_DIM + j] = rlt::get(input_mat, i, j);
            hsize_t dims[] = {N};
            hid_t space = H5Screate_simple(1, dims, nullptr);
            hid_t ds = H5Dcreate2(file.id, "test_input", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat);
            H5Dclose(ds);
            H5Sclose(space);
        }
        auto output_mat = rlt::matrix_view(device, output);
        {
            constexpr TI N = BATCH_SIZE * OUTPUT_DIM;
            float flat[N];
            for(TI i = 0; i < BATCH_SIZE; i++)
                for(TI j = 0; j < OUTPUT_DIM; j++)
                    flat[i * OUTPUT_DIM + j] = rlt::get(output_mat, i, j);
            hsize_t dims[] = {N};
            hid_t space = H5Screate_simple(1, dims, nullptr);
            hid_t ds = H5Dcreate2(file.id, "expected_output", H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, flat);
            H5Dclose(ds);
            H5Sclose(space);
        }
    }

    printf("Wrote %s\n", path.c_str());
    printf("  MLP: %lu -> %lu(RELU) -> %lu(RELU) -> %lu(IDENTITY)\n",
           (unsigned long)INPUT_DIM, (unsigned long)HIDDEN_DIM, (unsigned long)HIDDEN_DIM, (unsigned long)OUTPUT_DIM);
    printf("  Test input: [%lu, %lu]\n", (unsigned long)BATCH_SIZE, (unsigned long)INPUT_DIM);

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, output);
    return 0;
}
