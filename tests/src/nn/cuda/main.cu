#define FUNCTION_PLACEMENT __device__ __host__

#include <layer_in_c/operations/cuda.h>
#include <layer_in_c/operations/cpu.h>

#include <layer_in_c/nn/operations_cuda.h>
#include <layer_in_c/nn_models/operations_cuda.h>
#include <layer_in_c/nn_models/operations_cpu.h>


#include "../../utils/utils.h"

//#include <gtest/gtest.h>

#include <random>
#include <chrono>
#include <highfive/H5File.hpp>

#ifdef LAYER_IN_C_TESTS_NN_CUDA_ENABLE_CUTLASS
#include "cutlass/gemm/device/gemm.h"
#endif

namespace lic = layer_in_c;

using DTYPE = float;


using DEVICE_CUDA = lic::devices::DefaultCUDA;
using DEVICE_CUDA_GENERIC = lic::devices::CUDA_GENERIC<DEVICE_CUDA::SPEC>;
using DEVICE_CPU = lic::devices::DefaultCPU;

constexpr DEVICE_CPU::index_t BATCH_SIZE = 1000;

template <typename DEVICE, typename T_T>
using StructureSpecification = lic::nn_models::mlp::StructureSpecification<T_T, typename DEVICE::index_t, 10, 5, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY>;


using NETWORK_SPEC_CUDA = lic::nn_models::mlp::AdamSpecification<StructureSpecification<DEVICE_CUDA_GENERIC, DTYPE>, lic::nn::optimizers::adam::DefaultParametersTF<DTYPE>>;
using NetworkType_CUDA = lic::nn_models::mlp::NeuralNetworkAdam<NETWORK_SPEC_CUDA>;
using NETWORK_SPEC_CPU = lic::nn_models::mlp::AdamSpecification<StructureSpecification<DEVICE_CPU, DTYPE>, lic::nn::optimizers::adam::DefaultParametersTF<DTYPE>>;
using NetworkType_CPU = lic::nn_models::mlp::NeuralNetworkAdam<NETWORK_SPEC_CPU>;

DEVICE_CPU::SPEC::LOGGING logger_cpu;
DEVICE_CPU device_cpu(logger_cpu);
NetworkType_CPU network_cpu;

DEVICE_CUDA::SPEC::LOGGING logger_cuda;
DEVICE_CUDA device_cuda(logger_cuda);
NetworkType_CUDA network_cuda;

//TEST(LAYER_IN_C_NN_MLP_CUDA, FULL_TRAINING) {
int main(){

    lic::reset_optimizer_state(device_cpu, network_cpu);
    lic::zero_gradient(device_cpu, network_cpu);
    auto rng = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    lic::init_weights(device_cpu, network_cpu, rng);

    lic::copy(network_cuda, network_cpu);

    DTYPE input_cpu[BATCH_SIZE][NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM];
    DTYPE output_cpu[BATCH_SIZE][NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM];
    DTYPE output_first_layer_cpu[BATCH_SIZE][NETWORK_SPEC_CPU::STRUCTURE_SPEC::HIDDEN_DIM];
    for(DEVICE_CPU::CPU::index_t batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
        for(size_t i = 0; i < NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM; ++i) {
            input_cpu[batch_i][i] = lic::random::uniform_real_distribution(DEVICE_CPU::SPEC::RANDOM(), -(DTYPE) 1, (DTYPE) 1, rng);
        }
    }

    constexpr unsigned NUM_ITERATIONS = 1;

    {
        auto start = std::chrono::high_resolution_clock::now();
        for(DEVICE_CPU::index_t i = 0; i < NUM_ITERATIONS; ++i) {
            for(DEVICE_CPU::CPU::index_t batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                lic::forward(device_cpu, network_cpu, input_cpu[batch_i], output_cpu[batch_i]);
                memcpy(output_first_layer_cpu[batch_i], network_cpu.input_layer.output, sizeof(DTYPE) * NETWORK_SPEC_CPU::STRUCTURE_SPEC::HIDDEN_DIM);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "Elapsed time CPU forward: " << elapsed_seconds.count() * 1000 * 1000 << " us" << std::endl;
    }

    // GPU part
    DEVICE_CUDA* device_cuda_gpu;
    cudaMalloc(&device_cuda_gpu, sizeof(DEVICE_CUDA));
    cudaMemcpy(device_cuda_gpu, &device_cuda, sizeof(DEVICE_CUDA), cudaMemcpyHostToDevice);

    NetworkType_CUDA* network_cuda_device;
    cudaMalloc(&network_cuda_device, sizeof(NetworkType_CUDA));
    cudaMemcpy(network_cuda_device, &network_cuda, sizeof(NetworkType_CUDA), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    DTYPE* input_gpu;
    cudaMalloc(&input_gpu, sizeof(DTYPE) * BATCH_SIZE * NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM);
    cudaMemcpy(input_gpu, input_cpu, sizeof(DTYPE) * BATCH_SIZE * NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Test first layer

    {
        DTYPE* output_first_layer_gpu;
        cudaMalloc(&output_first_layer_gpu, sizeof(DTYPE) * BATCH_SIZE * NETWORK_SPEC_CPU::STRUCTURE_SPEC::HIDDEN_DIM);
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(DEVICE_CPU::index_t i = 0; i < NUM_ITERATIONS; ++i) {
            for(DEVICE_CPU::CPU::index_t batch_i = 0; batch_i < BATCH_SIZE; batch_i++) {
                lic::evaluate(*device_cuda_gpu, network_cuda_device->input_layer, &input_gpu[batch_i * NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM], &output_first_layer_gpu[batch_i * NETWORK_SPEC_CPU::STRUCTURE_SPEC::HIDDEN_DIM]);
            }
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "Elapsed time GPU layer: " << elapsed_seconds.count() * 1000 * 1000 << " us" << std::endl;

        DTYPE output_first_layer_gpu_cpu[BATCH_SIZE][NETWORK_SPEC_CPU::STRUCTURE_SPEC::HIDDEN_DIM];
        cudaMemcpy(output_first_layer_gpu_cpu, output_first_layer_gpu, sizeof(DTYPE) * BATCH_SIZE * NETWORK_SPEC_CPU::STRUCTURE_SPEC::HIDDEN_DIM, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        DTYPE output_first_layer_diff_per_weight = lic::nn::layers::dense::helper::abs_diff_matrix<DTYPE, BATCH_SIZE, NETWORK_SPEC_CPU::STRUCTURE_SPEC::HIDDEN_DIM>(output_first_layer_gpu_cpu, output_first_layer_cpu) / NetworkType_CUDA::NUM_WEIGHTS ;

        std::cout << "CPU - CUDA evaluation diff input layer: " << output_first_layer_diff_per_weight << std::endl;
        assert(output_first_layer_diff_per_weight < 1e-7);
    }





#ifdef LAYER_IN_C_TESTS_NN_CUDA_ENABLE_CUTLASS
    // Speed tests CUTLASS
    {
        constexpr unsigned M = NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM;
        constexpr unsigned K = NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM;
        constexpr unsigned N = 1;
        constexpr DTYPE alpha = 1, beta = 1;
        constexpr unsigned lda = M, ldb = K, ldc = M;
        using Majority = cutlass::layout::RowMajor;
        using CutlassGemm = cutlass::gemm::device::Gemm<DTYPE,        // Data-type of A matrix
                Majority,  // Layout of A matrix
                float,        // Data-type of B matrix
                Majority,  // Layout of B matrix
                float,        // Data-type of C matrix
                Majority>; // Layout of C matrix
        CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                                    {(DTYPE *) network_cuda_device->input_layer.weights, K},    // Tensor-ref for source matrix A
                                    {input_gpu, N},    // Tensor-ref for source matrix B
                                    {(DTYPE *) network_cuda_device->input_layer.biases, N},    // Tensor-ref for source matrix C
                                    {output_first_layer_gpu, N},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                    {alpha, beta}); // Scalars used in the Epilogue

        CutlassGemm gemm_operator;
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(DEVICE_CPU::index_t i = 0; i < NUM_ITERATIONS; ++i) {
            cutlass::Status status = gemm_operator(args);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "Elapsed time CUTLASS layer: " << elapsed_seconds.count() * 1000 * 1000 << " us" << std::endl;
    }

    cudaMemcpy(output_first_layer_gpu_cpu, output_first_layer_gpu, sizeof(DTYPE) * NETWORK_SPEC_CPU::INPUT_LAYER::SPEC::OUTPUT_DIM, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    DTYPE output_first_layer_cutlass_diff = lic::nn::layers::dense::helper::abs_diff_vector<DTYPE, NETWORK_SPEC_CPU::INPUT_LAYER::SPEC::OUTPUT_DIM>(output_first_layer_gpu_cpu, network_cpu.input_layer.output);

    std::cout << "CPU - CUDA evaluation diff input layer cutlass: " << output_first_layer_cutlass_diff << std::endl;
#endif

    // Test full network
    DTYPE* output_full_network_gpu;
    cudaMalloc(&output_full_network_gpu, sizeof(DTYPE) * BATCH_SIZE * NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM);

    {
        DTYPE* layer_output_tick;
        DTYPE* layer_output_tock;
        cudaMalloc((void**)&layer_output_tick, sizeof(DTYPE) * NETWORK_SPEC_CUDA::STRUCTURE_SPEC::HIDDEN_DIM);
        cudaMalloc((void**)&layer_output_tock, sizeof(DTYPE) * NETWORK_SPEC_CUDA::STRUCTURE_SPEC::HIDDEN_DIM);
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(DEVICE_CPU::index_t i = 0; i < NUM_ITERATIONS; ++i) {
            for(DEVICE_CPU::CPU::index_t batch_i = 0; batch_i < BATCH_SIZE; batch_i++) {
                lic::evaluate_memless(*device_cuda_gpu, *network_cuda_device, &input_gpu[batch_i * NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM], &output_full_network_gpu[batch_i * NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM], layer_output_tick, layer_output_tock);
            }
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
//        cudaFree((void**)&layer_output_tick);
//        cudaFree((void**)&layer_output_tock);
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "Elapsed time GPU forward: " << elapsed_seconds.count() * 1000 * 1000 << " us" << std::endl;
    }

    DTYPE output_full_network_gpu_cpu[BATCH_SIZE][NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM];
    cudaMemcpy(output_full_network_gpu_cpu, output_full_network_gpu, sizeof(DTYPE) * BATCH_SIZE * NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    DTYPE output_full_network_diff = lic::nn::layers::dense::helper::abs_diff_matrix<DTYPE, BATCH_SIZE, NETWORK_SPEC_CPU::STRUCTURE_SPEC::OUTPUT_DIM>(output_full_network_gpu_cpu, output_cpu);

    std::cout << "CPU - CUDA evaluation diff full output: " << output_full_network_diff << std::endl;

    // Test batch layer evaluation
    {
        DTYPE* output;
        cudaMalloc(&output, sizeof(DTYPE) * BATCH_SIZE * NETWORK_SPEC_CPU::STRUCTURE_SPEC::HIDDEN_DIM);

        DTYPE* input;
        cudaMalloc(&input, sizeof(DTYPE) * BATCH_SIZE * NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM);
        cudaMemcpy(input, input_cpu, sizeof(DTYPE) * BATCH_SIZE * NETWORK_SPEC_CPU::STRUCTURE_SPEC::INPUT_DIM, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(DEVICE_CPU::index_t i = 0; i < NUM_ITERATIONS; ++i) {
            lic::evaluate_batch<DEVICE_CUDA::SPEC, NetworkType_CUDA::SPEC::INPUT_LAYER::SPEC, BATCH_SIZE>(*device_cuda_gpu, network_cuda_device->input_layer, input, output);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end-start;
        std::cout << "Elapsed time GPU batch forward: " << elapsed_seconds.count() * 1000 * 1000 << " us" << std::endl;


        DTYPE output_layer_batch_gpu_cpu[BATCH_SIZE][NETWORK_SPEC_CPU::STRUCTURE_SPEC::HIDDEN_DIM];
        cudaMemcpy(output_layer_batch_gpu_cpu, output, sizeof(DTYPE) * BATCH_SIZE * NETWORK_SPEC_CPU::STRUCTURE_SPEC::HIDDEN_DIM, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        DTYPE output_layer_batch_network_diff_per_weight = lic::nn::layers::dense::helper::abs_diff_matrix<DTYPE, BATCH_SIZE, NETWORK_SPEC_CPU::STRUCTURE_SPEC::HIDDEN_DIM>(output_layer_batch_gpu_cpu, output_first_layer_cpu) / NetworkType_CUDA::NUM_WEIGHTS;
        std::cout << "CPU - CUDA evaluation batch diff: " << output_layer_batch_network_diff_per_weight << std::endl;
        auto layer_cpu = network_cpu.input_layer;
        decltype(layer_cpu)* input_layer_gpu;
        cudaMalloc(&input_layer_gpu, sizeof(decltype(layer_cpu)));
        cudaMemcpy(input_layer_gpu, &network_cuda.input_layer, sizeof(decltype(layer_cpu)), cudaMemcpyDeviceToHost);
        assert(output_layer_batch_network_diff_per_weight < 1e-15);
    }

    return 0;
}
