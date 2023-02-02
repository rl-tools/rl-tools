// Group 1
#include <layer_in_c/operations/cpu/group_1.h>
#include <layer_in_c/operations/cuda/group_1.h>

// Group 2
#include <layer_in_c/operations/cpu/group_2.h>
#include <layer_in_c/operations/cuda/group_2.h>

// Group 3
#include <layer_in_c/operations/cpu/group_3.h>
#include <layer_in_c/operations/cuda/group_3.h>

#include <layer_in_c/nn/operations_cuda.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/nn_models/operations_cpu.h>

namespace lic = layer_in_c;
using DTYPE = float;

#include <gtest/gtest.h>

using DEVICE_CPU = lic::devices::DefaultCPU;
using DEVICE_CUDA = lic::devices::DefaultCUDA;

constexpr DEVICE_CPU::index_t BATCH_SIZE = 10;
constexpr DEVICE_CPU::index_t HIDDEN_DIM = BATCH_SIZE;

template <typename T, typename TI, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
using StructureSpecification = lic::nn_models::mlp::StructureSpecification<T, TI, HIDDEN_DIM, HIDDEN_DIM, 3, HIDDEN_DIM, ACTIVATION_FUNCTION, lic::nn::activation_functions::RELU, BATCH_SIZE>;

template <typename T, typename TI, lic::nn::activation_functions::ActivationFunction ACTIVATION_FUNCTION>
using InferenceSpecification = lic::nn_models::mlp::AdamSpecification<StructureSpecification<T, TI, ACTIVATION_FUNCTION>, lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;

TEST(LAYER_IN_C_NN_CUDA, COPY) {
    using NetworkTypeCPU = lic::nn_models::mlp::NeuralNetworkAdam<InferenceSpecification<DTYPE, DEVICE_CPU::index_t, lic::nn::activation_functions::RELU>>;
    using NetworkTypeCUDA = lic::nn_models::mlp::NeuralNetworkAdam<InferenceSpecification<DTYPE, DEVICE_CUDA::index_t, lic::nn::activation_functions::RELU>>;
    DEVICE_CPU::SPEC::LOGGING cpu_logger;
    DEVICE_CUDA::SPEC::LOGGING cuda_logger;
    DEVICE_CPU device_cpu(cpu_logger);
    DEVICE_CUDA device_cuda(cuda_logger);
    NetworkTypeCPU network_cpu;
    NetworkTypeCPU network_cpu_2;
    NetworkTypeCUDA network_cuda;
    lic::malloc(device_cpu, network_cpu);
    lic::malloc(device_cpu, network_cpu_2);
    lic::malloc(device_cuda, network_cuda);

    auto rng = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());

    lic::init_weights(device_cpu, network_cpu, rng);
    lic::init_weights(device_cpu, network_cpu_2, rng);
    auto cpu_network_diff = lic::abs_diff(device_cpu, network_cpu, network_cpu_2);
    std::cout << "CPU network diff: " << cpu_network_diff << std::endl;
    ASSERT_GT(cpu_network_diff, 0);

    lic::copy(device_cuda, device_cpu, network_cuda, network_cpu);
    lic::copy(device_cpu, device_cuda, network_cpu_2, network_cuda);
    auto cpu_network_diff_round_trip = lic::abs_diff(device_cpu, network_cpu, network_cpu_2);
    std::cout << "CPU network round-trip: " << cpu_network_diff_round_trip << std::endl;
    ASSERT_FLOAT_EQ(cpu_network_diff_round_trip, 0);

    network_cpu.hidden_layers[0].weights.data[50] += 5;
    std::cout << "CPU network weights: " << network_cpu.hidden_layers[0].weights.data[50] << std::endl;

    cpu_network_diff = lic::abs_diff(device_cpu, network_cpu, network_cpu_2);
    std::cout << "CPU network diff: " << cpu_network_diff << std::endl;
    ASSERT_FLOAT_EQ(cpu_network_diff, 5);

    lic::copy(device_cuda, device_cpu, network_cuda, network_cpu);
    lic::copy(device_cpu, device_cuda, network_cpu_2, network_cuda);
    std::cout << "CPU network weights: " << network_cpu_2.hidden_layers[0].weights.data[50] << std::endl;
    cpu_network_diff_round_trip = lic::abs_diff(device_cpu, network_cpu, network_cpu_2);
    ASSERT_FLOAT_EQ(cpu_network_diff_round_trip, 0);
    std::cout << "CPU network round-trip: " << cpu_network_diff_round_trip << std::endl;

    lic::free(device_cpu, network_cpu);
    lic::free(device_cpu, network_cpu_2);
    lic::free(device_cuda, network_cuda);
}

TEST(LAYER_IN_C_NN_CUDA, GEMM) {
    using NetworkTypeCPU = lic::nn_models::mlp::NeuralNetworkAdam<InferenceSpecification<DTYPE, DEVICE_CPU::index_t, lic::nn::activation_functions::RELU>>;
    using NetworkTypeCUDA = lic::nn_models::mlp::NeuralNetworkAdam<InferenceSpecification<DTYPE, DEVICE_CUDA::index_t, lic::nn::activation_functions::RELU>>;
    DEVICE_CPU::SPEC::LOGGING cpu_logger;
    DEVICE_CUDA::SPEC::LOGGING cuda_logger;
    DEVICE_CPU device_cpu(cpu_logger);
    DEVICE_CUDA device_cuda(cuda_logger);
    NetworkTypeCPU network_cpu;
    NetworkTypeCPU::Buffers<BATCH_SIZE> network_cpu_buffers;
    NetworkTypeCUDA network_cuda;
    NetworkTypeCPU::Buffers<BATCH_SIZE> network_cuda_buffers;
    lic::malloc(device_cpu, network_cpu);
    lic::malloc(device_cpu, network_cpu_buffers);
    lic::malloc(device_cuda, network_cuda);
    lic::malloc(device_cpu, network_cuda_buffers);

    auto rng = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());

    lic::init_weights(device_cpu, network_cpu, rng);
    lic::copy(device_cuda, device_cpu, network_cuda, network_cpu);

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cpu;
    lic::malloc(device_cpu, input_cpu);
    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_first_layer_cpu;
    lic::malloc(device_cpu, output_first_layer_cpu);
    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE_CPU::index_t, BATCH_SIZE, NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_first_layer_cuda_cpu;
    lic::malloc(device_cpu, output_first_layer_cuda_cpu);

    for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE * NetworkTypeCPU::INPUT_DIM; ++i)
    {
        input_cpu.data[i] = lic::random::uniform_real_distribution(DEVICE_CPU::SPEC::RANDOM(), (DTYPE)0, (DTYPE)1, rng);
    }

    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::INPUT_DIM>> input_cuda;
    lic::malloc(device_cuda, input_cuda);
    lic::Matrix<lic::MatrixSpecification<DTYPE, DEVICE_CUDA::index_t, BATCH_SIZE, NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM>> output_first_layer_cuda;
    lic::malloc(device_cuda, output_first_layer_cuda);

    lic::copy(device_cuda, device_cpu, input_cuda, input_cpu);

    lic::evaluate(device_cpu, network_cpu.input_layer, input_cpu, output_first_layer_cpu);
    lic::evaluate(device_cuda, network_cuda.input_layer, input_cuda, output_first_layer_cuda);
    cudaDeviceSynchronize();

    lic::copy(device_cpu, device_cuda, output_first_layer_cuda_cpu, output_first_layer_cuda);
    auto evaluation_diff = lic::abs_diff(device_cpu, output_first_layer_cuda_cpu, output_first_layer_cpu);

    for(typename NetworkTypeCPU::TI i = 0; i < BATCH_SIZE * NetworkTypeCPU::SPEC::STRUCTURE_SPEC::HIDDEN_DIM; ++i){
        std::cout << "CPU: " << output_first_layer_cpu.data[i] << " CUDA: " << output_first_layer_cuda_cpu.data[i] << std::endl;
    }

    std::cout << "Evaluation diff: " << evaluation_diff << std::endl;
}
