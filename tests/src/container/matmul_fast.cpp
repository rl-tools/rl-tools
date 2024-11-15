
#define RL_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION

#include <thread>
#include <rl_tools/operations/cpu.h>
#include <rl_tools/operations/cpu_mkl.h>
namespace rlt = rl_tools;
using DEVICE = rlt::devices::DefaultCPU;
using DEVICE_MKL = rlt::devices::DefaultCPU_MKL;
using T = float;
using TI = typename DEVICE::index_t;

#include <immintrin.h> // For AVX intrinsics

#include <chrono>

namespace rl_tools
{
    template<typename DEVICE_SPEC, typename INPUT_SPEC_A, typename INPUT_SPEC_B, typename OUTPUT_SPEC>
    void multiply_naive(rlt::devices::CPU<DEVICE_SPEC>& device, const rlt::Matrix<INPUT_SPEC_A>& A, const rlt::Matrix<INPUT_SPEC_B>& B, rlt::Matrix<OUTPUT_SPEC>& C){
        static_assert(INPUT_SPEC_A::ROWS == OUTPUT_SPEC::ROWS);
        static_assert(INPUT_SPEC_A::COLS == INPUT_SPEC_B::ROWS);
        static_assert(INPUT_SPEC_B::COLS == OUTPUT_SPEC::COLS);

        using T = typename OUTPUT_SPEC::T;
        using TI = typename DEVICE::index_t;

        constexpr TI M = INPUT_SPEC_A::ROWS;
        constexpr TI N = INPUT_SPEC_B::COLS;
        constexpr TI K = INPUT_SPEC_A::COLS;

        T * __restrict__ A_data = A._data;
        T * __restrict__ B_data = B._data;
        T * __restrict__ C_data = C._data;

        for (TI i = 0; i < M * N; ++i) {
            C_data[i] = 0.0f;
        }

        for (TI i = 0; i < M; ++i) {
            for (TI k_idx = 0; k_idx < K; ++k_idx) {
                float A_val = A_data[i * K + k_idx];
                for (TI j = 0; j < N; ++j) {
                    C_data[i * N + j] += A_val * B_data[k_idx * N + j];
                }
            }
        }
    }
    template<typename DEVICE_SPEC, typename INPUT_SPEC_A, typename INPUT_SPEC_B, typename OUTPUT_SPEC>
    void multiply_tiled(rlt::devices::CPU<DEVICE_SPEC>& device, const rlt::Matrix<INPUT_SPEC_A>& A, const rlt::Matrix<INPUT_SPEC_B>& B, rlt::Matrix<OUTPUT_SPEC>& C) {
        static_assert(INPUT_SPEC_A::ROWS == OUTPUT_SPEC::ROWS);
        static_assert(INPUT_SPEC_A::COLS == INPUT_SPEC_B::ROWS);
        static_assert(INPUT_SPEC_B::COLS == OUTPUT_SPEC::COLS);

        using T = typename OUTPUT_SPEC::T;
        using TI = typename DEVICE::index_t;

        constexpr TI M = INPUT_SPEC_A::ROWS;
        constexpr TI N = INPUT_SPEC_B::COLS;
        constexpr TI K = INPUT_SPEC_A::COLS;
        constexpr TI blockSize = 32;

        T * __restrict__ A_data = A._data;
        T * __restrict__ B_data = B._data;
        T * __restrict__ C_data = C._data;

        // Initialize C to zero
        for (TI i = 0; i < M * N; ++i) {
            C_data[i] = 0.0f;
        }

        for (TI ii = 0; ii < M; ii += blockSize){
            for (TI kk = 0; kk < K; kk += blockSize){
                for (TI jj = 0; jj < N; jj += blockSize){
                    TI i_end = (ii + blockSize > M) ? M : ii + blockSize;
                    TI k_end = (kk + blockSize > K) ? K : kk + blockSize;
                    TI j_end = (jj + blockSize > N) ? N : jj + blockSize;
                    for (TI i = ii; i < i_end; ++i) {
                        for (TI k_idx = kk; k_idx < k_end; ++k_idx) {
                            float A_val = A_data[i * K + k_idx];
                            for (TI j = jj; j < j_end; ++j) {
                                C_data[i * N + j] += A_val * B_data[k_idx * N + j];
                            }
                        }
                    }
                }
            }
        }
    }

    template<typename DEVICE_SPEC, typename INPUT_SPEC_A, typename INPUT_SPEC_B, typename OUTPUT_SPEC>
    void multiply_tiled_simd(rlt::devices::CPU<DEVICE_SPEC>& device, const rlt::Matrix<INPUT_SPEC_A>& A, const rlt::Matrix<INPUT_SPEC_B>& B, rlt::Matrix<OUTPUT_SPEC>& C) {
        // Ensure matrix dimensions are compatible
        static_assert(INPUT_SPEC_A::ROWS == OUTPUT_SPEC::ROWS, "A.rows must equal C.rows");
        static_assert(INPUT_SPEC_A::COLS == INPUT_SPEC_B::ROWS, "A.cols must equal B.rows");
        static_assert(INPUT_SPEC_B::COLS == OUTPUT_SPEC::COLS, "B.cols must equal C.cols");

        using T = typename OUTPUT_SPEC::T;
        using TI = typename DEVICE_SPEC::index_t; // Corrected index type

        constexpr TI M = INPUT_SPEC_A::ROWS;
        constexpr TI N = INPUT_SPEC_B::COLS;
        constexpr TI K = INPUT_SPEC_A::COLS;
        constexpr TI blockSize = 32; // Adjust block size as needed

        T* __restrict__ A_data = A._data;
        T* __restrict__ B_data = B._data;
        T* __restrict__ C_data = C._data;

        // Initialize C to zero
        for (TI i = 0; i < M * N; ++i) {
            C_data[i] = static_cast<T>(0);
        }

        // Tiled matrix multiplication with SIMD intrinsics
        for (TI ii = 0; ii < M; ii += blockSize) {
            for (TI kk = 0; kk < K; kk += blockSize) {
                for (TI jj = 0; jj < N; jj += blockSize) {
                    TI i_end = (ii + blockSize > M) ? M : ii + blockSize;
                    TI k_end = (kk + blockSize > K) ? K : kk + blockSize;
                    TI j_end = (jj + blockSize > N) ? N : jj + blockSize;

                    for (TI i = ii; i < i_end; ++i) {
                        for (TI k_idx = kk; k_idx < k_end; ++k_idx) {
                            T A_val = A_data[i * K + k_idx];

                            // SIMD vectorization over the j loop
                            TI j = jj;
                            // Process 8 elements at a time using AVX
                            for (; j <= j_end - 8; j += 8) {
                                // Load B and C data into vectors
                                __m256 B_vec = _mm256_loadu_ps(&B_data[k_idx * N + j]);
                                __m256 C_vec = _mm256_loadu_ps(&C_data[i * N + j]);
                                __m256 A_vec = _mm256_set1_ps(A_val); // Broadcast A_val

                                // Perform fused multiply-add: C_vec += A_vec * B_vec
                                C_vec = _mm256_fmadd_ps(A_vec, B_vec, C_vec);

                                // Store the result back to C_data
                                _mm256_storeu_ps(&C_data[i * N + j], C_vec);
                            }

                            // Handle remaining elements
                            for (; j < j_end; ++j) {
                                C_data[i * N + j] += A_val * B_data[k_idx * N + j];
                            }
                        }
                    }
                }
            }
        }
    }

    template<typename DEVICE_SPEC, typename INPUT_SPEC_A, typename INPUT_SPEC_B, typename OUTPUT_SPEC>
    void multiply(rlt::devices::CPU<DEVICE_SPEC>& device, const rlt::Matrix<INPUT_SPEC_A>& A, const rlt::Matrix<INPUT_SPEC_B>& B, rlt::Matrix<OUTPUT_SPEC>& C){
        using TI = typename DEVICE::index_t;
        constexpr TI M = INPUT_SPEC_A::ROWS;
        constexpr TI N = INPUT_SPEC_B::COLS;
        constexpr TI K = INPUT_SPEC_A::COLS;
        if(M <= 64 || N <= 64 || K <= 64){
            multiply_tiled(device, A, B, C);
        } else {
            multiply_naive(device, A, B, C);
        }
        // multiply_naive(device, A, B, C);
    }
}

template <TI ITERATIONS, typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_C>
void benchmark(DEVICE& device, rlt::Matrix<SPEC_A>& A, rlt::Matrix<SPEC_B>& B, rlt::Matrix<SPEC_C>& C, std::string device_name, bool print = true){
    constexpr TI M = SPEC_C::ROWS;
    constexpr TI N = SPEC_C::COLS;
    constexpr TI K = SPEC_A::COLS;
    constexpr TI FLOPS = 2 * M * N * K * ITERATIONS;
    auto now = std::chrono::high_resolution_clock::now();
    for(TI i = 0; i < ITERATIONS; i++){
        rlt::multiply(device, A, B, C);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - now;
    T checksum = 0;
    for(TI i = 0; i < M; ++i){
        for(TI j = 0; j < N; ++j){
            checksum += rlt::get(C, i, j);
        }
    }
    if(print){
        std::cout << "        Device: " << device_name << std::endl;
        std::cout << "            Checksum: " << checksum << std::endl;
        std::cout << "            Elapsed time: " << elapsed.count() << "s" << std::endl;
        std::cout << "            GFLOPS: " << FLOPS / elapsed.count() / 1e9 << std::endl;
    }
}

template <TI M, TI N, TI K, TI ITERATIONS, bool DYNAMIC_ALLOCATION>
void matmul(){
    DEVICE device;
    DEVICE_MKL device_mkl;
    auto rng = rlt::random::default_engine(device.random);

    rlt::Matrix<rlt::matrix::Specification<T, TI, M, K, DYNAMIC_ALLOCATION>> A;
    rlt::Matrix<rlt::matrix::Specification<T, TI, K, N, DYNAMIC_ALLOCATION>> B;
    rlt::Matrix<rlt::matrix::Specification<T, TI, M, N, DYNAMIC_ALLOCATION>> C, C_target;
    if constexpr(DYNAMIC_ALLOCATION){
        rlt::malloc(device, A);
        rlt::malloc(device, B);
        rlt::malloc(device, C);
        rlt::utils::assert_exit(device, reinterpret_cast<TI>(A._data) % 64 == 0, "A._data % 64 == 0");
        rlt::utils::assert_exit(device, reinterpret_cast<TI>(B._data) % 64 == 0, "B._data % 64 == 0");
        rlt::utils::assert_exit(device, reinterpret_cast<TI>(C._data) % 64 == 0, "C._data % 64 == 0");
    }
    rlt::randn(device, A, rng);
    rlt::randn(device, B, rng);
    std::cout << "    M: " << M << " N: " << N << " K: " << K << std::endl;
    benchmark<20>(device_mkl, A, B, C, "mkl", false);
    benchmark<ITERATIONS>(device_mkl, A, B, C, "mkl");
    benchmark<20>(device, A, B, C, "generic", false);
    benchmark<ITERATIONS/10>(device, A, B, C, "generic");

}

int main(){
    // stack
    // std::cout << "Stack" << std::endl;
    // matmul<32, 32, 32, 1000000, false>();
    // matmul<64, 64, 64, 100000, false>();
    // matmul<128, 128, 128, 10000, false>();
    // heap
    std::cout << "Heap" << std::endl;
    matmul<32, 32, 32, 1000000, true>();
    matmul<64, 64, 64, 100000, true>();
    matmul<128, 128, 128, 10000, true>();
    // matmul<256, 256, 256, 1000, true>();
    // matmul<512, 512, 512, 100, true>();
    // matmul<1024, 1024, 1024, 10, true>();
    return 0;
}
