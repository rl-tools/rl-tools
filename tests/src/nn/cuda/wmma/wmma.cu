#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

using namespace nvcuda;

__global__ void init_kernel(__nv_bfloat16* global_a, __nv_bfloat16* global_b, float* global_c, int M, int N, int K) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                global_a[i * K + j] = __nv_bfloat16(i * K + j);
                global_b[i * N + j] = __nv_bfloat16(j * N + i); // Transpose B for WMMA
                global_c[i * N + j] = 0.0f; // Initialize C to zero
            }
        }
    }
}

__global__ void wmma_kernel(__nv_bfloat16* global_a, __nv_bfloat16* global_b, float* global_c, int M, int N, int K) {
    // Declare shared memory for tiles
    __shared__ __nv_bfloat16 shared_a[16][16];
    __shared__ __nv_bfloat16 shared_b[16][16];

    // Thread ID within the warp
    int lane_id = threadIdx.x % 32;

    // Load tiles into shared memory
    for (int i = 0; i < 16; i++) {
        shared_a[i][lane_id] = global_a[i * K + lane_id];
        shared_b[i][lane_id] = global_b[i * N + lane_id];
    }
    __syncthreads();

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag; // B is column-major
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Load matrices into fragments
    wmma::load_matrix_sync(a_frag, &shared_a[0][0], 16);
    wmma::load_matrix_sync(b_frag, &shared_b[0][0], 16);

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Perform matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the result back to global memory
    wmma::store_matrix_sync(&global_c[0], c_frag, N, wmma::mem_row_major);
}

int main() {
    int M = 16;
    int N = 16;
    int K = 16;

    // Allocate device memory
    __nv_bfloat16 *a, *b;
    float *c;
    cudaMalloc(&a, M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&b, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&c, M * N * sizeof(float));

    // Allocate host memory for verification
    __nv_bfloat16* b_cpu = static_cast<__nv_bfloat16*>(malloc(K * N * sizeof(__nv_bfloat16)));
    float* c_cpu = static_cast<float*>(malloc(M * N * sizeof(float)));

    // Initialize matrices A and B
    init_kernel<<<1, 1>>>(a, b, c, M, N, K);
    cudaDeviceSynchronize();

    // Launch the WMMA kernel
    wmma_kernel<<<1, 32>>>(a, b, c, M, N, K);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(c_cpu, c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_cpu, b, K * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    // Print matrix B
    printf("Matrix B:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", __bfloat162float(b_cpu[i * N + j]));
        }
        printf("\n");
    }

    // Print matrix C
    printf("\nMatrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", c_cpu[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    free(b_cpu);
    free(c_cpu);

    return 0;
}

