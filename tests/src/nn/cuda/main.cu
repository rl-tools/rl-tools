#define FUNCTION_PLACEMENT __device__ __host__

#include <layer_in_c/operations/cuda.h>
#include <layer_in_c/operations/cpu.h>

#include <layer_in_c/nn_models/operations_cuda.h>
#include <layer_in_c/nn_models/operations_cpu.h>


#include "../../utils/utils.h"

//#include <gtest/gtest.h>

#include <random>
#include <chrono>
#include <highfive/H5File.hpp>

namespace lic = layer_in_c;

typedef double DTYPE;


using DEVICE_CUDA = lic::devices::DefaultCUDA;
using DEVICE_CPU = lic::devices::DefaultCPU;

template <typename DEVICE, typename T_T>
using StructureSpecification = lic::nn_models::mlp::StructureSpecification<T_T, typename DEVICE::index_t, 5, 4, 3, 3, lic::nn::activation_functions::GELU, lic::nn::activation_functions::IDENTITY>;


using NETWORK_SPEC_CUDA = lic::nn_models::mlp::AdamSpecification<DEVICE_CUDA, StructureSpecification<DEVICE_CUDA, DTYPE>, lic::nn::optimizers::adam::DefaultParametersTF<DTYPE>>;
using NetworkType_CUDA = lic::nn_models::mlp::NeuralNetworkAdam<DEVICE_CUDA, NETWORK_SPEC_CUDA>;
using NETWORK_SPEC_CPU = lic::nn_models::mlp::AdamSpecification<DEVICE_CPU, StructureSpecification<DEVICE_CPU, DTYPE>, lic::nn::optimizers::adam::DefaultParametersTF<DTYPE>>;
using NetworkType_CPU = lic::nn_models::mlp::NeuralNetworkAdam<DEVICE_CPU, NETWORK_SPEC_CPU>;


//TEST(LAYER_IN_C_NN_MLP_CUDA, FULL_TRAINING) {
int main(){
    DEVICE_CPU::SPEC::LOGGING logger_cpu;
    DEVICE_CPU device_cpu(logger_cpu);
    NetworkType_CPU network_cpu(device_cpu);

    DEVICE_CUDA::SPEC::LOGGING logger_cuda;
    DEVICE_CUDA device_cuda(logger_cuda);
    NetworkType_CUDA network_cuda(device_cuda);

    lic::reset_optimizer_state(network_cpu);
    lic::zero_gradient(network_cpu);
    auto rng = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    lic::init_weights(network_cpu, rng);

    lic::copy(network_cuda, network_cpu);

    DTYPE input_cpu[NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM];
    DTYPE output_cpu[NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM];
    DTYPE d_loss_d_output_cpu[NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM];
    DTYPE d_input_cpu[NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM];
    for(size_t i = 0; i < NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM; ++i) {
        input_cpu[i] = lic::random::uniform_real_distribution(DEVICE_CPU::SPEC::RANDOM(), -(DTYPE)1, (DTYPE)1, rng);
    }
    for(size_t i = 0; i < NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM; ++i) {
        output_cpu[i] = lic::random::uniform_real_distribution(DEVICE_CPU::SPEC::RANDOM(), -(DTYPE)1, (DTYPE)1, rng);
    }

    lic::forward(network_cpu, input_cpu);
    lic::nn::loss_functions::d_mse_d_x<DEVICE_CPU, DTYPE, NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM, 1>(network_cpu.output_layer.output, output_cpu, d_loss_d_output_cpu);
    DTYPE loss_cpu = lic::nn::loss_functions::mse<DEVICE_CPU, DTYPE, NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM, 1>(network_cpu.output_layer.output, output_cpu);
    lic::backward(network_cpu, input_cpu, d_loss_d_output_cpu, d_input_cpu);

    NetworkType_CUDA* network_cuda_device;
    cudaMalloc(&network_cuda_device, sizeof(NetworkType_CUDA));
    cudaMemcpy(network_cuda_device, &network_cuda, sizeof(NetworkType_CUDA), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    DTYPE* input_gpu;
    cudaMalloc(&input_gpu, sizeof(DTYPE) * NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM);
    cudaMemcpy(input_gpu, input_cpu, sizeof(input_gpu) * NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    DTYPE* output_gpu;
    cudaMalloc(&output_gpu, sizeof(DTYPE) * NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM);

    lic::evaluate(*network_cuda_device, input_gpu, output_gpu);

    DTYPE output_gpu_cpu[NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM];
    cudaMemcpy(output_gpu_cpu, output_gpu, sizeof(DTYPE) * NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    DTYPE output_diff = lic::nn::layers::dense::helper::abs_diff_vector<DTYPE, NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM>(output_gpu_cpu, network_cpu.output_layer.output);
//    ASSERT_LT(output_diff, 1e-15);

    std::cout << "CPU - CUDA evaluation diff: " << output_diff << std::endl;

    DTYPE* d_loss_d_output_gpu;
    cudaMalloc(&d_loss_d_output_gpu, sizeof(DTYPE) * NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM);


    DTYPE* d_input;
    cudaMalloc(&d_input, sizeof(DTYPE) * NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM);
    return 0;
}
