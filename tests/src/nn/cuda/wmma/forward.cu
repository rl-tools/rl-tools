
#define RL_TOOLS_FUNCTION_PLACEMENT __device__ __host__
#define RL_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION
#include <rl_tools/operations/cpu.h>
#include <rl_tools/operations/cuda.h>
#include <rl_tools/nn/layers/operations_cuda.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <chrono>
namespace rlt = rl_tools;

using namespace nvcuda;

__global__ void wmma_kernel(__nv_bfloat16* global_a, __nv_bfloat16* global_b, float* global_c, int M, int N, int K) {
    __shared__ __nv_bfloat162 shared_a_mem[16][8];
    __shared__ __nv_bfloat162 shared_b_mem[16][8];

    int loads_per_row = 8;
    int lane_id = threadIdx.x % loads_per_row;
    int row_id = threadIdx.x / loads_per_row;
    int increment = 32 / loads_per_row;
    for (int i = 0; i < 16; i += increment) {
        int index = (i + row_id) * K + lane_id * 2;
        shared_a_mem[i + row_id][lane_id] = *reinterpret_cast<__nv_bfloat162*>(&global_a[index]);
        shared_b_mem[i + row_id][lane_id] = *reinterpret_cast<__nv_bfloat162*>(&global_b[index]);
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag; // B is column-major
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    __nv_bfloat16* shared_a = reinterpret_cast<__nv_bfloat16*>(shared_a_mem);
    __nv_bfloat16* shared_b = reinterpret_cast<__nv_bfloat16*>(shared_b_mem);
    wmma::load_matrix_sync(a_frag, shared_a, 16);
    wmma::load_matrix_sync(b_frag, shared_b, 16);

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(&global_c[0], c_frag, N, wmma::mem_row_major);
}
using DEVICE_CUDA = rlt::devices::DefaultCUDA;
using DEVICE_CPU = rlt::devices::DefaultCPU;
using T = float;
using TI = DEVICE_CUDA::index_t;
constexpr bool DYNAMIC_ALLOCATION = true;
using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
using LOOP_CORE_PARAMETERS = rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>;
using RNG_CUDA = DEVICE_CUDA::SPEC::RANDOM::ENGINE<>;
using RNG_CPU = DEVICE_CPU::SPEC::RANDOM::ENGINE<>;

constexpr TI SEQUENCE_LENGTH = 1;
constexpr TI BATCH_SIZE = 16;
constexpr TI INPUT_DIM = 16;
constexpr TI OUTPUT_DIM = 16;
constexpr TI HIDDEN_DIM = 16;
constexpr TI NUM_LAYERS = 3;

using CAPABILITY = rlt::nn::capability::Forward<true>;
using CAPABILITY_STATIC = rlt::nn::capability::Forward<false>;
namespace network_builder{
    using namespace rlt;
    using INPUT_SHAPE = tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, INPUT_DIM>;
    using MLP_CONFIG = nn_models::mlp::Configuration<T, TI, OUTPUT_DIM, NUM_LAYERS, HIDDEN_DIM, nn::activation_functions::RELU, nn::activation_functions::IDENTITY>;
    using MLP = nn_models::mlp::BindConfiguration<MLP_CONFIG>;
    template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
    using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
    using MODULE_CHAIN = Module<MLP>;
    using MODEL = nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
    using MODEL_STATIC = nn_models::sequential::Build<CAPABILITY_STATIC, MODULE_CHAIN, INPUT_SHAPE>;
};

using NETWORK = network_builder::MODEL;
using NETWORK_STATIC = network_builder::MODEL_STATIC;


struct CUDA_FUSED{
    using index_t = DEVICE_CUDA::index_t;
    using SPEC = DEVICE_CUDA::SPEC;
};
struct NORNG{};

template <typename INPUT, typename OUTPUT, typename BUFFER>
__global__ void evaluate_fused(NETWORK nn, INPUT input, OUTPUT output, BUFFER buffer){
    __shared__ NETWORK_STATIC nn_shared;
    if(threadIdx.x == 0){
        CUDA_FUSED device;
        NORNG rng;
        rlt::copy(device, device, nn, nn_shared);
        for (TI i = 0; i < 1000; i++){
            rlt::evaluate(device, nn_shared, input, output, buffer, rng);
        }
    }
}


int main() {
    DEVICE_CUDA device_cuda;
    RNG_CUDA rng_cuda;
    rlt::init(device_cuda);
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    DEVICE_CPU device_cpu;
    RNG_CPU rng_cpu;
    rlt::init(device_cpu);
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 0);

    NETWORK nn_cuda, nn_cpu;
    NETWORK::Buffer<> buffer_cuda, buffer_cpu;
    rlt::malloc(device_cuda, nn_cuda);
    rlt::malloc(device_cpu, nn_cpu);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::malloc(device_cpu, buffer_cpu);

    rlt::Tensor<rlt::tensor::Specification<T, TI, NETWORK::INPUT_SHAPE>> input_cuda, input_cpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, NETWORK::OUTPUT_SHAPE>> output_cuda, output_cpu, output_cuda_cpu;
    rlt::malloc(device_cuda, input_cuda);
    rlt::malloc(device_cpu, input_cpu);
    rlt::malloc(device_cuda, output_cuda);
    rlt::malloc(device_cpu, output_cpu);
    rlt::malloc(device_cpu, output_cuda_cpu);

    rlt::randn(device_cuda, input_cuda, rng_cuda);
    rlt::copy(device_cuda, device_cpu, input_cuda, input_cpu);
    rlt::init_weights(device_cuda, nn_cuda, rng_cuda);
    rlt::copy(device_cuda, device_cpu, nn_cuda, nn_cpu);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    evaluate_fused<<<1, 32>>>(nn_cuda, input_cuda, output_cuda, buffer_cuda);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds/1000);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
    }
    auto start_cpu = std::chrono::high_resolution_clock::now();
    rlt::evaluate(device_cpu, nn_cpu, input_cpu, output_cpu, buffer_cpu, rng_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_cpu - start_cpu;
    printf("Time: %f ms\n", elapsed_seconds.count() * 1000);


    rlt::copy(device_cuda, device_cpu, output_cuda, output_cuda_cpu);
    T diff = rlt::abs_diff(device_cpu, output_cpu, output_cuda_cpu);
    rlt::print(device_cpu, output_cuda_cpu);
    printf("Difference Output: %f\n", diff);

}

