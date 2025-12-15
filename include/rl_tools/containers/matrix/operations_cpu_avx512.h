#include <immintrin.h>
#include <type_traits>

template <typename T, unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC>
void gemm_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C){
    // Compute C = A * B, where:
    // A is row-major (A[i, k] => A[i*LDA + k]), B is column-major (B[k, j] => B[k + j*LDB])
    // C is row-major (C[i, j] => C[i*LDC + j])
    // We exploit compile-time M,N,K, strides and use a packed-B + vectorized (over N) micro-kernel.

    if constexpr (std::is_same_v<T, float>) {
    #if defined(__AVX512F__)
        constexpr unsigned VEC = 16; // __m512: 16 floats
        alignas(64) float Bp[VEC * K]; // packed B tile (K x VEC), aligned for fast loads

        // Register-blocking: process multiple rows (MR) per micro-kernel so each loaded B vector
        // is reused across several A rows (improves L1 reuse and ILP on Zen4).
        constexpr unsigned MR = 8; // tune: 8 rows in register block
        for (unsigned jj = 0; jj < N; jj += VEC) {
            const unsigned jb = (N - jj >= VEC) ? VEC : (N - jj);

            // Pack a VEC-wide column-panel of B into row-major contiguous blocks per k
            for (unsigned k = 0; k < K; ++k) {
                unsigned t = 0;
                for (; t < jb; ++t) Bp[k * VEC + t] = B[k + (jj + t) * LDB];
                for (; t < VEC; ++t) Bp[k * VEC + t] = 0.0f; // zero-pad the tail for masked store
            }

            const __mmask16 tail_mask = (jb == VEC) ? (__mmask16)0xFFFF : (__mmask16)((1u << jb) - 1u);

            // Tile over rows of A/C in MR blocks so we can keep MR accumulators in registers
            for (unsigned ii = 0; ii < M; ii += MR) {
                const unsigned R = (M - ii < MR) ? (M - ii) : MR;

                // Pointers to the R rows of A
                const float* __restrict__ a_rows[MR];
                #pragma GCC unroll 8
                for (unsigned r = 0; r < R; ++r) a_rows[r] = A + (ii + r) * LDA;

                // Initialize accumulators: one zmm per row in the block
                __m512 c[MR];
                #pragma GCC unroll 8
                for (unsigned r = 0; r < R; ++r) c[r] = _mm512_setzero_ps();

                // Reduce across K with unrolling to expose ILP
                unsigned k = 0;
                const unsigned K4 = (K / 4) * 4;
                for (; k < K4; k += 4) {
                    const __m512 vb0 = _mm512_loadu_ps(&Bp[(k + 0) * VEC]);
                    const __m512 vb1 = _mm512_loadu_ps(&Bp[(k + 1) * VEC]);
                    const __m512 vb2 = _mm512_loadu_ps(&Bp[(k + 2) * VEC]);
                    const __m512 vb3 = _mm512_loadu_ps(&Bp[(k + 3) * VEC]);
                    #pragma GCC unroll 8
                    for (unsigned r = 0; r < R; ++r) {
                        const __m512 a0 = _mm512_set1_ps(a_rows[r][k + 0]);
                        const __m512 a1 = _mm512_set1_ps(a_rows[r][k + 1]);
                        const __m512 a2 = _mm512_set1_ps(a_rows[r][k + 2]);
                        const __m512 a3 = _mm512_set1_ps(a_rows[r][k + 3]);
                        c[r] = _mm512_fmadd_ps(a0, vb0, c[r]);
                        c[r] = _mm512_fmadd_ps(a1, vb1, c[r]);
                        c[r] = _mm512_fmadd_ps(a2, vb2, c[r]);
                        c[r] = _mm512_fmadd_ps(a3, vb3, c[r]);
                    }
                }
                for (; k < K; ++k) {
                    const __m512 vb = _mm512_loadu_ps(&Bp[k * VEC]);
                    #pragma GCC unroll 8
                    for (unsigned r = 0; r < R; ++r) {
                        const __m512 a = _mm512_set1_ps(a_rows[r][k]);
                        c[r] = _mm512_fmadd_ps(a, vb, c[r]);
                    }
                }

                // Store accumulators to C (masked if tail)
                #pragma GCC unroll 8
                for (unsigned r = 0; r < R; ++r) {
                    float* __restrict__ Ci = C + (ii + r) * LDC + jj;
                    if (jb == VEC) {
                        _mm512_storeu_ps(Ci, c[r]);
                    } else {
                        _mm512_mask_storeu_ps(Ci, tail_mask, c[r]);
                    }
                }
            }
        }
        return;
    #elif defined(__AVX2__)
        constexpr unsigned VEC = 8; // __m256: 8 floats
        alignas(32) float Bp[VEC * K];

        unsigned jj = 0;
        // Full vector tiles
        for (; jj + VEC <= N; jj += VEC) {
            for (unsigned k = 0; k < K; ++k) {
                // Pack VEC columns j=jj..jj+VEC-1 of row k
                for (unsigned t = 0; t < VEC; ++t) {
                    Bp[k * VEC + t] = B[k + (jj + t) * LDB];
                }
            }

            for (unsigned i = 0; i < M; ++i) {
                __m256 acc = _mm256_setzero_ps();
                const float* __restrict__ Ai = A + i * LDA;

                unsigned k = 0;
                for (; k + 3 < K; k += 4) {
                    acc = _mm256_fmadd_ps(_mm256_load_ps(&Bp[(k + 0) * VEC]), _mm256_set1_ps(Ai[k + 0]), acc);
                    acc = _mm256_fmadd_ps(_mm256_load_ps(&Bp[(k + 1) * VEC]), _mm256_set1_ps(Ai[k + 1]), acc);
                    acc = _mm256_fmadd_ps(_mm256_load_ps(&Bp[(k + 2) * VEC]), _mm256_set1_ps(Ai[k + 2]), acc);
                    acc = _mm256_fmadd_ps(_mm256_load_ps(&Bp[(k + 3) * VEC]), _mm256_set1_ps(Ai[k + 3]), acc);
                }
                for (; k < K; ++k) {
                    acc = _mm256_fmadd_ps(_mm256_load_ps(&Bp[k * VEC]), _mm256_set1_ps(Ai[k]), acc);
                }

                _mm256_storeu_ps(C + i * LDC + jj, acc);
            }
        }

        // Scalar remainder columns
        for (; jj < N; ++jj) {
            for (unsigned i = 0; i < M; ++i) {
                float sum = 0.0f;
                const float* __restrict__ Ai = A + i * LDA;
                for (unsigned k = 0; k < K; ++k) {
                    sum += Ai[k] * B[k + jj * LDB];
                }
                C[i * LDC + jj] = sum;
            }
        }
        return;
    #endif
    } else if constexpr (std::is_same_v<T, double>) {
    #if defined(__AVX512F__)
        constexpr unsigned VEC = 8; // __m512d: 8 doubles
        alignas(64) double Bp[VEC * K];

        for (unsigned jj = 0; jj < N; jj += VEC) {
            const unsigned jb = (N - jj >= VEC) ? VEC : (N - jj);

            // Pack B
            for (unsigned k = 0; k < K; ++k) {
                unsigned t = 0;
                for (; t < jb; ++t) Bp[k * VEC + t] = B[k + (jj + t) * LDB];
                for (; t < VEC; ++t) Bp[k * VEC + t] = 0.0;
            }

            for (unsigned i = 0; i < M; ++i) {
                __m512d acc = _mm512_setzero_pd();
                const double* __restrict__ Ai = A + i * LDA;

                unsigned k = 0;
                for (; k + 3 < K; k += 4) {
                    acc = _mm512_fmadd_pd(_mm512_load_pd(&Bp[(k + 0) * VEC]), _mm512_set1_pd(Ai[k + 0]), acc);
                    acc = _mm512_fmadd_pd(_mm512_load_pd(&Bp[(k + 1) * VEC]), _mm512_set1_pd(Ai[k + 1]), acc);
                    acc = _mm512_fmadd_pd(_mm512_load_pd(&Bp[(k + 2) * VEC]), _mm512_set1_pd(Ai[k + 2]), acc);
                    acc = _mm512_fmadd_pd(_mm512_load_pd(&Bp[(k + 3) * VEC]), _mm512_set1_pd(Ai[k + 3]), acc);
                }
                for (; k < K; ++k) {
                    acc = _mm512_fmadd_pd(_mm512_load_pd(&Bp[k * VEC]), _mm512_set1_pd(Ai[k]), acc);
                }

                double* __restrict__ Ci = C + i * LDC + jj;
                if (jb == VEC) {
                    _mm512_storeu_pd(Ci, acc);
                } else {
                    const __mmask8 mask = (jb == 8) ? 0xFF : ((__mmask8(1) << jb) - 1);
                    _mm512_mask_storeu_pd(Ci, mask, acc);
                }
            }
        }
        return;
    #elif defined(__AVX2__)
        constexpr unsigned VEC = 4; // __m256d: 4 doubles
        alignas(32) double Bp[VEC * K];

        unsigned jj = 0;
        for (; jj + VEC <= N; jj += VEC) {
            for (unsigned k = 0; k < K; ++k) {
                for (unsigned t = 0; t < VEC; ++t) {
                    Bp[k * VEC + t] = B[k + (jj + t) * LDB];
                }
            }

            for (unsigned i = 0; i < M; ++i) {
                __m256d acc = _mm256_setzero_pd();
                const double* __restrict__ Ai = A + i * LDA;

                unsigned k = 0;
                for (; k + 3 < K; k += 4) {
                    acc = _mm256_fmadd_pd(_mm256_load_pd(&Bp[(k + 0) * VEC]), _mm256_set1_pd(Ai[k + 0]), acc);
                    acc = _mm256_fmadd_pd(_mm256_load_pd(&Bp[(k + 1) * VEC]), _mm256_set1_pd(Ai[k + 1]), acc);
                    acc = _mm256_fmadd_pd(_mm256_load_pd(&Bp[(k + 2) * VEC]), _mm256_set1_pd(Ai[k + 2]), acc);
                    acc = _mm256_fmadd_pd(_mm256_load_pd(&Bp[(k + 3) * VEC]), _mm256_set1_pd(Ai[k + 3]), acc);
                }
                for (; k < K; ++k) {
                    acc = _mm256_fmadd_pd(_mm256_load_pd(&Bp[k * VEC]), _mm256_set1_pd(Ai[k]), acc);
                }

                _mm256_storeu_pd(C + i * LDC + jj, acc);
            }
        }

        for (; jj < N; ++jj) {
            for (unsigned i = 0; i < M; ++i) {
                double sum = 0.0;
                const double* __restrict__ Ai = A + i * LDA;
                for (unsigned k = 0; k < K; ++k) {
                    sum += Ai[k] * B[k + jj * LDB];
                }
                C[i * LDC + jj] = sum;
            }
        }
        return;
    #endif
    }

    // Generic scalar fallback (also used when no SIMD available)
    for (unsigned i = 0; i < M; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            T sum = T(0);
            for (unsigned k = 0; k < K; ++k) {
                sum += A[i * LDA + k] * B[k + j * LDB];
            }
            C[i * LDC + j] = sum;
        }
    }
}