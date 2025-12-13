#if defined(__aarch64__) || defined(__ARM_NEON)
  #include <arm_neon.h>
#endif

template <typename T, unsigned M, unsigned N, unsigned K, unsigned LDA, unsigned LDB, unsigned LDC>
void gemm_kernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C){
    // Optimized single-threaded GEMM for small M,N,K (<= ~300), B is column-major (transposed relative to A/C)
    // Key ideas:
    // - Remove separate zeroing pass; accumulate into registers and store once.
    // - Block across columns (N) to reuse A[i,k] across multiple C(i, j..j+u-1).
    // - Pointer arithmetic to eliminate repeated index multiplies.
    // - Unroll K by 4 for ILP and FMA throughput; dispatch on N to 8/4/2/1-column micro-kernels.
    // - All sizes and strides are compile-time constants; use if constexpr to prune code.
    auto rowptr = [&](unsigned i) -> const T* { return A + i * LDA; };
    auto colptr = [&](unsigned j) -> const T* { return B + j * LDB; };
    auto cptr   = [&](unsigned i, unsigned j) -> T* { return C + i * LDC + j; };

    // 8-column micro-kernel (NEON-accelerated where available; scalar fallback otherwise)
    // Improvements:
    // - Fast path for very small compile-time K: fully unrolled scalar accumulation.
    // - Assume-aligned pointers to help the compiler generate better code.
    // - Unroll by 8 with prefetching to improve ILP and hide latency.
    auto compute_n8 = [&](const T* __restrict__ Ap, const T* __restrict__ Bp, T* __restrict__ Cp){
        // Small-K fast path (compile-time): eliminates loop overhead and maximizes FMA density
        if constexpr (K <= 4) {
            const T* __restrict__ a = Ap;
            const T* __restrict__ p0 = Bp + 0 * LDB;
            const T* __restrict__ p1 = Bp + 1 * LDB;
            const T* __restrict__ p2 = Bp + 2 * LDB;
            const T* __restrict__ p3 = Bp + 3 * LDB;
            const T* __restrict__ p4 = Bp + 4 * LDB;
            const T* __restrict__ p5 = Bp + 5 * LDB;
            const T* __restrict__ p6 = Bp + 6 * LDB;
            const T* __restrict__ p7 = Bp + 7 * LDB;

            T s0 = T(0), s1 = T(0), s2 = T(0), s3 = T(0);
            T s4 = T(0), s5 = T(0), s6 = T(0), s7 = T(0);

            #pragma unroll
            for (unsigned kk = 0; kk < K; ++kk) {
                T av = a[kk];
                s0 += av * p0[kk];
                s1 += av * p1[kk];
                s2 += av * p2[kk];
                s3 += av * p3[kk];
                s4 += av * p4[kk];
                s5 += av * p5[kk];
                s6 += av * p6[kk];
                s7 += av * p7[kk];
            }

            Cp[0] = s0; Cp[1] = s1; Cp[2] = s2; Cp[3] = s3;
            Cp[4] = s4; Cp[5] = s5; Cp[6] = s6; Cp[7] = s7;
            return;
        }

#if defined(__aarch64__)
        if constexpr (std::is_same_v<T, float>) {
            const float* __restrict__ a  = (const float*)__builtin_assume_aligned(Ap, 16);
            const float* __restrict__ p0 = (const float*)__builtin_assume_aligned(Bp + 0 * LDB, 16);
            const float* __restrict__ p1 = (const float*)__builtin_assume_aligned(Bp + 1 * LDB, 16);
            const float* __restrict__ p2 = (const float*)__builtin_assume_aligned(Bp + 2 * LDB, 16);
            const float* __restrict__ p3 = (const float*)__builtin_assume_aligned(Bp + 3 * LDB, 16);
            const float* __restrict__ p4 = (const float*)__builtin_assume_aligned(Bp + 4 * LDB, 16);
            const float* __restrict__ p5 = (const float*)__builtin_assume_aligned(Bp + 5 * LDB, 16);
            const float* __restrict__ p6 = (const float*)__builtin_assume_aligned(Bp + 6 * LDB, 16);
            const float* __restrict__ p7 = (const float*)__builtin_assume_aligned(Bp + 7 * LDB, 16);

            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f), acc5 = vdupq_n_f32(0.0f);
            float32x4_t acc6 = vdupq_n_f32(0.0f), acc7 = vdupq_n_f32(0.0f);

            unsigned k = 0;
            __builtin_prefetch(a);
            __builtin_prefetch(p0); __builtin_prefetch(p1); __builtin_prefetch(p2); __builtin_prefetch(p3);
            __builtin_prefetch(p4); __builtin_prefetch(p5); __builtin_prefetch(p6); __builtin_prefetch(p7);

            for (; k + 7 < K; k += 8) {
                float32x4_t av0 = vld1q_f32(a + k);
                float32x4_t av1 = vld1q_f32(a + k + 4);

                float32x4_t b0_0 = vld1q_f32(p0 + k);
                float32x4_t b0_1 = vld1q_f32(p0 + k + 4);
                acc0 = vfmaq_f32(acc0, av0, b0_0);
                acc0 = vfmaq_f32(acc0, av1, b0_1);

                float32x4_t b1_0 = vld1q_f32(p1 + k);
                float32x4_t b1_1 = vld1q_f32(p1 + k + 4);
                acc1 = vfmaq_f32(acc1, av0, b1_0);
                acc1 = vfmaq_f32(acc1, av1, b1_1);

                float32x4_t b2_0 = vld1q_f32(p2 + k);
                float32x4_t b2_1 = vld1q_f32(p2 + k + 4);
                acc2 = vfmaq_f32(acc2, av0, b2_0);
                acc2 = vfmaq_f32(acc2, av1, b2_1);

                float32x4_t b3_0 = vld1q_f32(p3 + k);
                float32x4_t b3_1 = vld1q_f32(p3 + k + 4);
                acc3 = vfmaq_f32(acc3, av0, b3_0);
                acc3 = vfmaq_f32(acc3, av1, b3_1);

                float32x4_t b4_0 = vld1q_f32(p4 + k);
                float32x4_t b4_1 = vld1q_f32(p4 + k + 4);
                acc4 = vfmaq_f32(acc4, av0, b4_0);
                acc4 = vfmaq_f32(acc4, av1, b4_1);

                float32x4_t b5_0 = vld1q_f32(p5 + k);
                float32x4_t b5_1 = vld1q_f32(p5 + k + 4);
                acc5 = vfmaq_f32(acc5, av0, b5_0);
                acc5 = vfmaq_f32(acc5, av1, b5_1);

                float32x4_t b6_0 = vld1q_f32(p6 + k);
                float32x4_t b6_1 = vld1q_f32(p6 + k + 4);
                acc6 = vfmaq_f32(acc6, av0, b6_0);
                acc6 = vfmaq_f32(acc6, av1, b6_1);

                float32x4_t b7_0 = vld1q_f32(p7 + k);
                float32x4_t b7_1 = vld1q_f32(p7 + k + 4);
                acc7 = vfmaq_f32(acc7, av0, b7_0);
                acc7 = vfmaq_f32(acc7, av1, b7_1);
            }

            for (; k + 3 < K; k += 4) {
                float32x4_t av = vld1q_f32(a + k);
                acc0 = vfmaq_f32(acc0, av, vld1q_f32(p0 + k));
                acc1 = vfmaq_f32(acc1, av, vld1q_f32(p1 + k));
                acc2 = vfmaq_f32(acc2, av, vld1q_f32(p2 + k));
                acc3 = vfmaq_f32(acc3, av, vld1q_f32(p3 + k));
                acc4 = vfmaq_f32(acc4, av, vld1q_f32(p4 + k));
                acc5 = vfmaq_f32(acc5, av, vld1q_f32(p5 + k));
                acc6 = vfmaq_f32(acc6, av, vld1q_f32(p6 + k));
                acc7 = vfmaq_f32(acc7, av, vld1q_f32(p7 + k));
            }

            float s0 = vaddvq_f32(acc0);
            float s1 = vaddvq_f32(acc1);
            float s2 = vaddvq_f32(acc2);
            float s3 = vaddvq_f32(acc3);
            float s4 = vaddvq_f32(acc4);
            float s5 = vaddvq_f32(acc5);
            float s6 = vaddvq_f32(acc6);
            float s7 = vaddvq_f32(acc7);

            for (; k < K; ++k) {
                float avs = a[k];
                s0 += avs * p0[k];
                s1 += avs * p1[k];
                s2 += avs * p2[k];
                s3 += avs * p3[k];
                s4 += avs * p4[k];
                s5 += avs * p5[k];
                s6 += avs * p6[k];
                s7 += avs * p7[k];
            }

            Cp[0] = s0; Cp[1] = s1; Cp[2] = s2; Cp[3] = s3;
            Cp[4] = s4; Cp[5] = s5; Cp[6] = s6; Cp[7] = s7;
            return;
        } else if constexpr (std::is_same_v<T, double>) {
            const double* __restrict__ a  = (const double*)__builtin_assume_aligned(Ap, 16);
            const double* __restrict__ p0 = (const double*)__builtin_assume_aligned(Bp + 0 * LDB, 16);
            const double* __restrict__ p1 = (const double*)__builtin_assume_aligned(Bp + 1 * LDB, 16);
            const double* __restrict__ p2 = (const double*)__builtin_assume_aligned(Bp + 2 * LDB, 16);
            const double* __restrict__ p3 = (const double*)__builtin_assume_aligned(Bp + 3 * LDB, 16);
            const double* __restrict__ p4 = (const double*)__builtin_assume_aligned(Bp + 4 * LDB, 16);
            const double* __restrict__ p5 = (const double*)__builtin_assume_aligned(Bp + 5 * LDB, 16);
            const double* __restrict__ p6 = (const double*)__builtin_assume_aligned(Bp + 6 * LDB, 16);
            const double* __restrict__ p7 = (const double*)__builtin_assume_aligned(Bp + 7 * LDB, 16);

            float64x2_t acc0 = vdupq_n_f64(0.0), acc1 = vdupq_n_f64(0.0);
            float64x2_t acc2 = vdupq_n_f64(0.0), acc3 = vdupq_n_f64(0.0);
            float64x2_t acc4 = vdupq_n_f64(0.0), acc5 = vdupq_n_f64(0.0);
            float64x2_t acc6 = vdupq_n_f64(0.0), acc7 = vdupq_n_f64(0.0);

            unsigned k = 0;
            __builtin_prefetch(a);
            for (; k + 1 < K; k += 2) {
                float64x2_t av = vld1q_f64(a + k);
                acc0 = vfmaq_f64(acc0, av, vld1q_f64(p0 + k));
                acc1 = vfmaq_f64(acc1, av, vld1q_f64(p1 + k));
                acc2 = vfmaq_f64(acc2, av, vld1q_f64(p2 + k));
                acc3 = vfmaq_f64(acc3, av, vld1q_f64(p3 + k));
                acc4 = vfmaq_f64(acc4, av, vld1q_f64(p4 + k));
                acc5 = vfmaq_f64(acc5, av, vld1q_f64(p5 + k));
                acc6 = vfmaq_f64(acc6, av, vld1q_f64(p6 + k));
                acc7 = vfmaq_f64(acc7, av, vld1q_f64(p7 + k));
            }
            double s0 = vaddvq_f64(acc0);
            double s1 = vaddvq_f64(acc1);
            double s2 = vaddvq_f64(acc2);
            double s3 = vaddvq_f64(acc3);
            double s4 = vaddvq_f64(acc4);
            double s5 = vaddvq_f64(acc5);
            double s6 = vaddvq_f64(acc6);
            double s7 = vaddvq_f64(acc7);
            for (; k < K; ++k) {
                double avs = a[k];
                s0 += avs * p0[k];
                s1 += avs * p1[k];
                s2 += avs * p2[k];
                s3 += avs * p3[k];
                s4 += avs * p4[k];
                s5 += avs * p5[k];
                s6 += avs * p6[k];
                s7 += avs * p7[k];
            }
            Cp[0] = s0; Cp[1] = s1; Cp[2] = s2; Cp[3] = s3;
            Cp[4] = s4; Cp[5] = s5; Cp[6] = s6; Cp[7] = s7;
            return;
        }
#endif
        // Scalar fallback
        const T* __restrict__ b0 = Bp + 0 * LDB;
        const T* __restrict__ b1 = Bp + 1 * LDB;
        const T* __restrict__ b2 = Bp + 2 * LDB;
        const T* __restrict__ b3 = Bp + 3 * LDB;
        const T* __restrict__ b4 = Bp + 4 * LDB;
        const T* __restrict__ b5 = Bp + 5 * LDB;
        const T* __restrict__ b6 = Bp + 6 * LDB;
        const T* __restrict__ b7 = Bp + 7 * LDB;

        T s0 = T(0), s1 = T(0), s2 = T(0), s3 = T(0);
        T s4 = T(0), s5 = T(0), s6 = T(0), s7 = T(0);

        const T* __restrict__ a = Ap;
        const T* __restrict__ p0 = b0;
        const T* __restrict__ p1 = b1;
        const T* __restrict__ p2 = b2;
        const T* __restrict__ p3 = b3;
        const T* __restrict__ p4 = b4;
        const T* __restrict__ p5 = b5;
        const T* __restrict__ p6 = b6;
        const T* __restrict__ p7 = b7;

        unsigned k = 0;
        for (; k + 3 < K; k += 4) {
            T a0 = a[0];
            s0 += a0 * p0[0]; s1 += a0 * p1[0]; s2 += a0 * p2[0]; s3 += a0 * p3[0];
            s4 += a0 * p4[0]; s5 += a0 * p5[0]; s6 += a0 * p6[0]; s7 += a0 * p7[0];

            T a1 = a[1];
            s0 += a1 * p0[1]; s1 += a1 * p1[1]; s2 += a1 * p2[1]; s3 += a1 * p3[1];
            s4 += a1 * p4[1]; s5 += a1 * p5[1]; s6 += a1 * p6[1]; s7 += a1 * p7[1];

            T a2 = a[2];
            s0 += a2 * p0[2]; s1 += a2 * p1[2]; s2 += a2 * p2[2]; s3 += a2 * p3[2];
            s4 += a2 * p4[2]; s5 += a2 * p5[2]; s6 += a2 * p6[2]; s7 += a2 * p7[2];

            T a3 = a[3];
            s0 += a3 * p0[3]; s1 += a3 * p1[3]; s2 += a3 * p2[3]; s3 += a3 * p3[3];
            s4 += a3 * p4[3]; s5 += a3 * p5[3]; s6 += a3 * p6[3]; s7 += a3 * p7[3];

            a += 4; p0 += 4; p1 += 4; p2 += 4; p3 += 4; p4 += 4; p5 += 4; p6 += 4; p7 += 4;
        }
        for (; k < K; ++k) {
            T av = *a++;
            s0 += av * (*p0++); s1 += av * (*p1++); s2 += av * (*p2++);
            s3 += av * (*p3++); s4 += av * (*p4++); s5 += av * (*p5++);
            s6 += av * (*p6++); s7 += av * (*p7++);
        }
        Cp[0] = s0; Cp[1] = s1; Cp[2] = s2; Cp[3] = s3;
        Cp[4] = s4; Cp[5] = s5; Cp[6] = s6; Cp[7] = s7;
    };

    // 4-column micro-kernel (NEON + fallback)
    // Improvements: unroll by 8 when possible, prefetch, single reduction.
    auto compute_n4 = [&](const T* __restrict__ Ap, const T* __restrict__ Bp, T* __restrict__ Cp){
#if defined(__aarch64__)
        if constexpr (std::is_same_v<T, float>) {
            const float* __restrict__ a = (const float*)__builtin_assume_aligned(Ap, 16);
            const float* __restrict__ p0 = (const float*)__builtin_assume_aligned(Bp + 0 * LDB, 16);
            const float* __restrict__ p1 = (const float*)__builtin_assume_aligned(Bp + 1 * LDB, 16);
            const float* __restrict__ p2 = (const float*)__builtin_assume_aligned(Bp + 2 * LDB, 16);
            const float* __restrict__ p3 = (const float*)__builtin_assume_aligned(Bp + 3 * LDB, 16);

            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);

            unsigned k = 0;
            __builtin_prefetch(a);
            __builtin_prefetch(p0); __builtin_prefetch(p1);

            for (; k + 7 < K; k += 8) {
                float32x4_t av0 = vld1q_f32(a + k);
                float32x4_t av1 = vld1q_f32(a + k + 4);

                float32x4_t b0_0 = vld1q_f32(p0 + k), b0_1 = vld1q_f32(p0 + k + 4);
                acc0 = vfmaq_f32(acc0, av0, b0_0); acc0 = vfmaq_f32(acc0, av1, b0_1);

                float32x4_t b1_0 = vld1q_f32(p1 + k), b1_1 = vld1q_f32(p1 + k + 4);
                acc1 = vfmaq_f32(acc1, av0, b1_0); acc1 = vfmaq_f32(acc1, av1, b1_1);

                float32x4_t b2_0 = vld1q_f32(p2 + k), b2_1 = vld1q_f32(p2 + k + 4);
                acc2 = vfmaq_f32(acc2, av0, b2_0); acc2 = vfmaq_f32(acc2, av1, b2_1);

                float32x4_t b3_0 = vld1q_f32(p3 + k), b3_1 = vld1q_f32(p3 + k + 4);
                acc3 = vfmaq_f32(acc3, av0, b3_0); acc3 = vfmaq_f32(acc3, av1, b3_1);
            }
            for (; k + 3 < K; k += 4) {
                float32x4_t av = vld1q_f32(a + k);
                acc0 = vfmaq_f32(acc0, av, vld1q_f32(p0 + k));
                acc1 = vfmaq_f32(acc1, av, vld1q_f32(p1 + k));
                acc2 = vfmaq_f32(acc2, av, vld1q_f32(p2 + k));
                acc3 = vfmaq_f32(acc3, av, vld1q_f32(p3 + k));
            }
            float s0 = vaddvq_f32(acc0);
            float s1 = vaddvq_f32(acc1);
            float s2 = vaddvq_f32(acc2);
            float s3 = vaddvq_f32(acc3);
            for (; k < K; ++k) {
                float avs = a[k];
                s0 += avs * p0[k];
                s1 += avs * p1[k];
                s2 += avs * p2[k];
                s3 += avs * p3[k];
            }
            Cp[0] = s0; Cp[1] = s1; Cp[2] = s2; Cp[3] = s3;
            return;
        } else if constexpr (std::is_same_v<T, double>) {
            const double* __restrict__ a = reinterpret_cast<const double*>(Ap);
            const double* __restrict__ p0 = reinterpret_cast<const double*>(Bp + 0 * LDB);
            const double* __restrict__ p1 = reinterpret_cast<const double*>(Bp + 1 * LDB);
            const double* __restrict__ p2 = reinterpret_cast<const double*>(Bp + 2 * LDB);
            const double* __restrict__ p3 = reinterpret_cast<const double*>(Bp + 3 * LDB);

            float64x2_t acc0 = vdupq_n_f64(0.0), acc1 = vdupq_n_f64(0.0);
            float64x2_t acc2 = vdupq_n_f64(0.0), acc3 = vdupq_n_f64(0.0);

            unsigned k = 0;
            for (; k + 1 < K; k += 2) {
                float64x2_t av = vld1q_f64(a + k);
                acc0 = vfmaq_f64(acc0, av, vld1q_f64(p0 + k));
                acc1 = vfmaq_f64(acc1, av, vld1q_f64(p1 + k));
                acc2 = vfmaq_f64(acc2, av, vld1q_f64(p2 + k));
                acc3 = vfmaq_f64(acc3, av, vld1q_f64(p3 + k));
            }
            double s0 = vaddvq_f64(acc0);
            double s1 = vaddvq_f64(acc1);
            double s2 = vaddvq_f64(acc2);
            double s3 = vaddvq_f64(acc3);
            for (; k < K; ++k) {
                double avs = a[k];
                s0 += avs * p0[k];
                s1 += avs * p1[k];
                s2 += avs * p2[k];
                s3 += avs * p3[k];
            }
            Cp[0] = s0; Cp[1] = s1; Cp[2] = s2; Cp[3] = s3;
            return;
        }
#endif
        // Scalar fallback
        const T* __restrict__ b0 = Bp + 0 * LDB;
        const T* __restrict__ b1 = Bp + 1 * LDB;
        const T* __restrict__ b2 = Bp + 2 * LDB;
        const T* __restrict__ b3 = Bp + 3 * LDB;

        T s0 = T(0), s1 = T(0), s2 = T(0), s3 = T(0);

        const T* __restrict__ a = Ap;
        const T* __restrict__ p0 = b0;
        const T* __restrict__ p1 = b1;
        const T* __restrict__ p2 = b2;
        const T* __restrict__ p3 = b3;

        unsigned k = 0;
        for (; k + 3 < K; k += 4) {
            T a0 = a[0];
            s0 += a0 * p0[0]; s1 += a0 * p1[0]; s2 += a0 * p2[0]; s3 += a0 * p3[0];
            T a1 = a[1];
            s0 += a1 * p0[1]; s1 += a1 * p1[1]; s2 += a1 * p2[1]; s3 += a1 * p3[1];
            T a2 = a[2];
            s0 += a2 * p0[2]; s1 += a2 * p1[2]; s2 += a2 * p2[2]; s3 += a2 * p3[2];
            T a3 = a[3];
            s0 += a3 * p0[3]; s1 += a3 * p1[3]; s2 += a3 * p2[3]; s3 += a3 * p3[3];

            a += 4; p0 += 4; p1 += 4; p2 += 4; p3 += 4;
        }
        for (; k < K; ++k) {
            T av = *a++;
            s0 += av * (*p0++); s1 += av * (*p1++); s2 += av * (*p2++); s3 += av * (*p3++);
        }
        Cp[0] = s0; Cp[1] = s1; Cp[2] = s2; Cp[3] = s3;
    };

    // 2-column micro-kernel (NEON + fallback)
    // Improvements: stronger unrolling and prefetch for better throughput on small to medium K.
    auto compute_n2 = [&](const T* __restrict__ Ap, const T* __restrict__ Bp, T* __restrict__ Cp){
#if defined(__aarch64__)
        if constexpr (std::is_same_v<T, float>) {
            const float* __restrict__ a = (const float*)__builtin_assume_aligned(Ap, 16);
            const float* __restrict__ p0 = (const float*)__builtin_assume_aligned(Bp + 0 * LDB, 16);
            const float* __restrict__ p1 = (const float*)__builtin_assume_aligned(Bp + 1 * LDB, 16);

            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            unsigned k = 0;
            __builtin_prefetch(a);
            __builtin_prefetch(p0);

            for (; k + 7 < K; k += 8) {
                float32x4_t av0 = vld1q_f32(a + k);
                float32x4_t av1 = vld1q_f32(a + k + 4);

                float32x4_t b0_0 = vld1q_f32(p0 + k), b0_1 = vld1q_f32(p0 + k + 4);
                acc0 = vfmaq_f32(acc0, av0, b0_0); acc0 = vfmaq_f32(acc0, av1, b0_1);

                float32x4_t b1_0 = vld1q_f32(p1 + k), b1_1 = vld1q_f32(p1 + k + 4);
                acc1 = vfmaq_f32(acc1, av0, b1_0); acc1 = vfmaq_f32(acc1, av1, b1_1);
            }
            for (; k + 3 < K; k += 4) {
                float32x4_t av = vld1q_f32(a + k);
                acc0 = vfmaq_f32(acc0, av, vld1q_f32(p0 + k));
                acc1 = vfmaq_f32(acc1, av, vld1q_f32(p1 + k));
            }
            float s0 = vaddvq_f32(acc0);
            float s1 = vaddvq_f32(acc1);
            for (; k < K; ++k) {
                float avs = a[k];
                s0 += avs * p0[k];
                s1 += avs * p1[k];
            }
            Cp[0] = s0; Cp[1] = s1;
            return;
        } else if constexpr (std::is_same_v<T, double>) {
            const double* __restrict__ a = reinterpret_cast<const double*>(Ap);
            const double* __restrict__ p0 = reinterpret_cast<const double*>(Bp + 0 * LDB);
            const double* __restrict__ p1 = reinterpret_cast<const double*>(Bp + 1 * LDB);

            float64x2_t acc0 = vdupq_n_f64(0.0), acc1 = vdupq_n_f64(0.0);
            unsigned k = 0;
            for (; k + 1 < K; k += 2) {
                float64x2_t av = vld1q_f64(a + k);
                acc0 = vfmaq_f64(acc0, av, vld1q_f64(p0 + k));
                acc1 = vfmaq_f64(acc1, av, vld1q_f64(p1 + k));
            }
            double s0 = vaddvq_f64(acc0);
            double s1 = vaddvq_f64(acc1);
            for (; k < K; ++k) {
                double avs = a[k];
                s0 += avs * p0[k];
                s1 += avs * p1[k];
            }
            Cp[0] = s0; Cp[1] = s1;
            return;
        }
#endif
        // Scalar fallback
        const T* __restrict__ b0 = Bp + 0 * LDB;
        const T* __restrict__ b1 = Bp + 1 * LDB;

        T s0 = T(0), s1 = T(0);

        const T* __restrict__ a = Ap;
        const T* __restrict__ p0 = b0;
        const T* __restrict__ p1 = b1;

        unsigned k = 0;
        for (; k + 3 < K; k += 4) {
            T a0 = a[0]; s0 += a0 * p0[0]; s1 += a0 * p1[0];
            T a1 = a[1]; s0 += a1 * p0[1]; s1 += a1 * p1[1];
            T a2 = a[2]; s0 += a2 * p0[2]; s1 += a2 * p1[2];
            T a3 = a[3]; s0 += a3 * p0[3]; s1 += a3 * p1[3];
            a += 4; p0 += 4; p1 += 4;
        }
        for (; k < K; ++k) {
            T av = *a++;
            s0 += av * (*p0++); s1 += av * (*p1++);
        }
        Cp[0] = s0; Cp[1] = s1;
    };

    // 1-column micro-kernel (NEON + fallback)
    // Improvements: unroll by 8 to increase ILP for tiny K and reduce loop overhead.
    auto compute_n1 = [&](const T* __restrict__ Ap, const T* __restrict__ Bp, T* __restrict__ Cp){
#if defined(__aarch64__)
        if constexpr (std::is_same_v<T, float>) {
            const float* __restrict__ a = (const float*)__builtin_assume_aligned(Ap, 16);
            const float* __restrict__ p0 = (const float*)__builtin_assume_aligned(Bp + 0 * LDB, 16);
            float32x4_t acc = vdupq_n_f32(0.0f);
            unsigned k = 0;
            __builtin_prefetch(a); __builtin_prefetch(p0);
            for (; k + 7 < K; k += 8) {
                float32x4_t av0 = vld1q_f32(a + k);
                float32x4_t av1 = vld1q_f32(a + k + 4);
                float32x4_t b0 = vld1q_f32(p0 + k);
                float32x4_t b1 = vld1q_f32(p0 + k + 4);
                acc = vfmaq_f32(acc, av0, b0);
                acc = vfmaq_f32(acc, av1, b1);
            }
            for (; k + 3 < K; k += 4) {
                float32x4_t av = vld1q_f32(a + k);
                acc = vfmaq_f32(acc, av, vld1q_f32(p0 + k));
            }
            float s = vaddvq_f32(acc);
            for (; k < K; ++k) s += a[k] * p0[k];
            Cp[0] = s;
            return;
        } else if constexpr (std::is_same_v<T, double>) {
            const double* __restrict__ a = reinterpret_cast<const double*>(Ap);
            const double* __restrict__ p0 = reinterpret_cast<const double*>(Bp + 0 * LDB);
            float64x2_t acc = vdupq_n_f64(0.0);
            unsigned k = 0;
            for (; k + 1 < K; k += 2) {
                float64x2_t av = vld1q_f64(a + k);
                acc = vfmaq_f64(acc, av, vld1q_f64(p0 + k));
            }
            double s = vaddvq_f64(acc);
            for (; k < K; ++k) s += a[k] * p0[k];
            Cp[0] = s;
            return;
        }
#endif
        // Scalar fallback
        const T* __restrict__ p0 = Bp + 0 * LDB;
        T s0 = T(0);
        const T* __restrict__ a = Ap;

        unsigned k = 0;
        for (; k + 3 < K; k += 4) {
            T a0 = a[0]; s0 += a0 * p0[0];
            T a1 = a[1]; s0 += a1 * p0[1];
            T a2 = a[2]; s0 += a2 * p0[2];
            T a3 = a[3]; s0 += a3 * p0[3];
            a += 4; p0 += 4;
        }
        for (; k < K; ++k) {
            s0 += (*a++) * (*p0++);
        }
        Cp[0] = s0;
    };

    // Row-blocked outer loop for float to reuse B loads across two rows, boosts arithmetic intensity.
    if constexpr (std::is_same_v<T, float> && M >= 2) {
        unsigned i = 0;
        for (; i + 1 < M; i += 2) {
            const float* __restrict__ a0 = (const float*)__builtin_assume_aligned(rowptr(i), 16);
            const float* __restrict__ a1 = (const float*)__builtin_assume_aligned(rowptr(i + 1), 16);
            unsigned j = 0;

            if constexpr (N >= 8) {
                for (; j + 8 <= N; j += 8) {
                    const float* __restrict__ Bp = reinterpret_cast<const float*>(colptr(j));
                    float* __restrict__ C0 = reinterpret_cast<float*>(cptr(i, j));
                    float* __restrict__ C1 = reinterpret_cast<float*>(cptr(i + 1, j));

#if defined(__aarch64__)
                    const float* __restrict__ p0 = Bp + 0 * LDB;
                    const float* __restrict__ p1 = Bp + 1 * LDB;
                    const float* __restrict__ p2 = Bp + 2 * LDB;
                    const float* __restrict__ p3 = Bp + 3 * LDB;
                    const float* __restrict__ p4 = Bp + 4 * LDB;
                    const float* __restrict__ p5 = Bp + 5 * LDB;
                    const float* __restrict__ p6 = Bp + 6 * LDB;
                    const float* __restrict__ p7 = Bp + 7 * LDB;

                    float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f), acc02 = vdupq_n_f32(0.0f), acc03 = vdupq_n_f32(0.0f);
                    float32x4_t acc04 = vdupq_n_f32(0.0f), acc05 = vdupq_n_f32(0.0f), acc06 = vdupq_n_f32(0.0f), acc07 = vdupq_n_f32(0.0f);
                    float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f);
                    float32x4_t acc14 = vdupq_n_f32(0.0f), acc15 = vdupq_n_f32(0.0f), acc16 = vdupq_n_f32(0.0f), acc17 = vdupq_n_f32(0.0f);

                    unsigned k = 0;
                    for (; k + 3 < K; k += 4) {
                        float32x4_t a0v = vld1q_f32(a0 + k);
                        float32x4_t a1v = vld1q_f32(a1 + k);

                        float32x4_t b0v = vld1q_f32(p0 + k);
                        float32x4_t b1v = vld1q_f32(p1 + k);
                        float32x4_t b2v = vld1q_f32(p2 + k);
                        float32x4_t b3v = vld1q_f32(p3 + k);
                        float32x4_t b4v = vld1q_f32(p4 + k);
                        float32x4_t b5v = vld1q_f32(p5 + k);
                        float32x4_t b6v = vld1q_f32(p6 + k);
                        float32x4_t b7v = vld1q_f32(p7 + k);

                        acc00 = vfmaq_f32(acc00, a0v, b0v); acc10 = vfmaq_f32(acc10, a1v, b0v);
                        acc01 = vfmaq_f32(acc01, a0v, b1v); acc11 = vfmaq_f32(acc11, a1v, b1v);
                        acc02 = vfmaq_f32(acc02, a0v, b2v); acc12 = vfmaq_f32(acc12, a1v, b2v);
                        acc03 = vfmaq_f32(acc03, a0v, b3v); acc13 = vfmaq_f32(acc13, a1v, b3v);
                        acc04 = vfmaq_f32(acc04, a0v, b4v); acc14 = vfmaq_f32(acc14, a1v, b4v);
                        acc05 = vfmaq_f32(acc05, a0v, b5v); acc15 = vfmaq_f32(acc15, a1v, b5v);
                        acc06 = vfmaq_f32(acc06, a0v, b6v); acc16 = vfmaq_f32(acc16, a1v, b6v);
                        acc07 = vfmaq_f32(acc07, a0v, b7v); acc17 = vfmaq_f32(acc17, a1v, b7v);
                    }

                    float s00 = vaddvq_f32(acc00), s01 = vaddvq_f32(acc01), s02 = vaddvq_f32(acc02), s03 = vaddvq_f32(acc03);
                    float s04 = vaddvq_f32(acc04), s05 = vaddvq_f32(acc05), s06 = vaddvq_f32(acc06), s07 = vaddvq_f32(acc07);
                    float s10 = vaddvq_f32(acc10), s11 = vaddvq_f32(acc11), s12 = vaddvq_f32(acc12), s13 = vaddvq_f32(acc13);
                    float s14 = vaddvq_f32(acc14), s15 = vaddvq_f32(acc15), s16 = vaddvq_f32(acc16), s17 = vaddvq_f32(acc17);

                    for (; k < K; ++k) {
                        float av0 = a0[k];
                        float av1 = a1[k];
                        s00 += av0 * p0[k]; s01 += av0 * p1[k]; s02 += av0 * p2[k]; s03 += av0 * p3[k];
                        s04 += av0 * p4[k]; s05 += av0 * p5[k]; s06 += av0 * p6[k]; s07 += av0 * p7[k];
                        s10 += av1 * p0[k]; s11 += av1 * p1[k]; s12 += av1 * p2[k]; s13 += av1 * p3[k];
                        s14 += av1 * p4[k]; s15 += av1 * p5[k]; s16 += av1 * p6[k]; s17 += av1 * p7[k];
                    }

                    C0[0] = s00; C0[1] = s01; C0[2] = s02; C0[3] = s03;
                    C0[4] = s04; C0[5] = s05; C0[6] = s06; C0[7] = s07;
                    C1[0] = s10; C1[1] = s11; C1[2] = s12; C1[3] = s13;
                    C1[4] = s14; C1[5] = s15; C1[6] = s16; C1[7] = s17;
#else
                    compute_n8(reinterpret_cast<const T*>(a0), reinterpret_cast<const T*>(Bp), reinterpret_cast<T*>(C0));
                    compute_n8(reinterpret_cast<const T*>(a1), reinterpret_cast<const T*>(Bp), reinterpret_cast<T*>(C1));
#endif
                }
            }

            if constexpr (N >= 4) {
                for (; j + 4 <= N; j += 4) {
                    const float* __restrict__ Bp = reinterpret_cast<const float*>(colptr(j));
                    float* __restrict__ C0 = reinterpret_cast<float*>(cptr(i, j));
                    float* __restrict__ C1 = reinterpret_cast<float*>(cptr(i + 1, j));

#if defined(__aarch64__)
                    const float* __restrict__ p0 = Bp + 0 * LDB;
                    const float* __restrict__ p1 = Bp + 1 * LDB;
                    const float* __restrict__ p2 = Bp + 2 * LDB;
                    const float* __restrict__ p3 = Bp + 3 * LDB;

                    float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f), acc02 = vdupq_n_f32(0.0f), acc03 = vdupq_n_f32(0.0f);
                    float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f), acc13 = vdupq_n_f32(0.0f);

                    unsigned k = 0;
                    for (; k + 3 < K; k += 4) {
                        float32x4_t a0v = vld1q_f32(a0 + k);
                        float32x4_t a1v = vld1q_f32(a1 + k);

                        float32x4_t b0v = vld1q_f32(p0 + k);
                        float32x4_t b1v = vld1q_f32(p1 + k);
                        float32x4_t b2v = vld1q_f32(p2 + k);
                        float32x4_t b3v = vld1q_f32(p3 + k);

                        acc00 = vfmaq_f32(acc00, a0v, b0v); acc10 = vfmaq_f32(acc10, a1v, b0v);
                        acc01 = vfmaq_f32(acc01, a0v, b1v); acc11 = vfmaq_f32(acc11, a1v, b1v);
                        acc02 = vfmaq_f32(acc02, a0v, b2v); acc12 = vfmaq_f32(acc12, a1v, b2v);
                        acc03 = vfmaq_f32(acc03, a0v, b3v); acc13 = vfmaq_f32(acc13, a1v, b3v);
                    }

                    float s00 = vaddvq_f32(acc00), s01 = vaddvq_f32(acc01), s02 = vaddvq_f32(acc02), s03 = vaddvq_f32(acc03);
                    float s10 = vaddvq_f32(acc10), s11 = vaddvq_f32(acc11), s12 = vaddvq_f32(acc12), s13 = vaddvq_f32(acc13);

                    for (; k < K; ++k) {
                        float av0 = a0[k];
                        float av1 = a1[k];
                        s00 += av0 * p0[k]; s01 += av0 * p1[k]; s02 += av0 * p2[k]; s03 += av0 * p3[k];
                        s10 += av1 * p0[k]; s11 += av1 * p1[k]; s12 += av1 * p2[k]; s13 += av1 * p3[k];
                    }

                    C0[0] = s00; C0[1] = s01; C0[2] = s02; C0[3] = s03;
                    C1[0] = s10; C1[1] = s11; C1[2] = s12; C1[3] = s13;
#else
                    compute_n4(reinterpret_cast<const T*>(a0), reinterpret_cast<const T*>(Bp), reinterpret_cast<T*>(C0));
                    compute_n4(reinterpret_cast<const T*>(a1), reinterpret_cast<const T*>(Bp), reinterpret_cast<T*>(C1));
#endif
                }
            }

            if constexpr (N >= 2) {
                if (j + 2 <= N) {
                    const float* __restrict__ Bp = reinterpret_cast<const float*>(colptr(j));
                    float* __restrict__ C0 = reinterpret_cast<float*>(cptr(i, j));
                    float* __restrict__ C1 = reinterpret_cast<float*>(cptr(i + 1, j));

#if defined(__aarch64__)
                    const float* __restrict__ p0 = Bp + 0 * LDB;
                    const float* __restrict__ p1 = Bp + 1 * LDB;

                    float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f);
                    float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f);

                    unsigned k = 0;
                    for (; k + 3 < K; k += 4) {
                        float32x4_t a0v = vld1q_f32(a0 + k);
                        float32x4_t a1v = vld1q_f32(a1 + k);
                        float32x4_t b0v = vld1q_f32(p0 + k);
                        float32x4_t b1v = vld1q_f32(p1 + k);

                        acc00 = vfmaq_f32(acc00, a0v, b0v); acc10 = vfmaq_f32(acc10, a1v, b0v);
                        acc01 = vfmaq_f32(acc01, a0v, b1v); acc11 = vfmaq_f32(acc11, a1v, b1v);
                    }

                    float s00 = vaddvq_f32(acc00), s01 = vaddvq_f32(acc01);
                    float s10 = vaddvq_f32(acc10), s11 = vaddvq_f32(acc11);

                    for (; k < K; ++k) {
                        float av0 = a0[k];
                        float av1 = a1[k];
                        s00 += av0 * p0[k]; s01 += av0 * p1[k];
                        s10 += av1 * p0[k]; s11 += av1 * p1[k];
                    }

                    C0[0] = s00; C0[1] = s01;
                    C1[0] = s10; C1[1] = s11;
#else
                    compute_n2(reinterpret_cast<const T*>(a0), reinterpret_cast<const T*>(Bp), reinterpret_cast<T*>(C0));
                    compute_n2(reinterpret_cast<const T*>(a1), reinterpret_cast<const T*>(Bp), reinterpret_cast<T*>(C1));
#endif
                    j += 2;
                }
            }

            if (j < N) {
                const T* __restrict__ Bp = colptr(j);
                T* __restrict__ Cp0 = cptr(i, j);
                T* __restrict__ Cp1 = cptr(i + 1, j);
                compute_n1(reinterpret_cast<const T*>(a0), Bp, Cp0);
                compute_n1(reinterpret_cast<const T*>(a1), Bp, Cp1);
                ++j;
            }
        }

        // process last odd row if M is odd
        if (i < M) {
            const T* __restrict__ Ap = rowptr(i);
            unsigned j = 0;
            if constexpr (N >= 8) {
                for (; j + 8 <= N; j += 8) {
                    const T* __restrict__ Bp = colptr(j);
                    T* __restrict__ Cp = cptr(i, j);
                    compute_n8(Ap, Bp, Cp);
                }
            }
            if constexpr (N >= 4) {
                for (; j + 4 <= N; j += 4) {
                    const T* __restrict__ Bp = colptr(j);
                    T* __restrict__ Cp = cptr(i, j);
                    compute_n4(Ap, Bp, Cp);
                }
            }
            if constexpr (N >= 2) {
                if (j + 2 <= N) {
                    const T* __restrict__ Bp = colptr(j);
                    T* __restrict__ Cp = cptr(i, j);
                    compute_n2(Ap, Bp, Cp);
                    j += 2;
                }
            }
            if (j < N) {
                const T* __restrict__ Bp = colptr(j);
                T* __restrict__ Cp = cptr(i, j);
                compute_n1(Ap, Bp, Cp);
                ++j;
            }
        }
    } else {
        // Generic path (non-float or very small M): original per-row logic
        for (unsigned i = 0; i < M; ++i) {
            const T* __restrict__ Ap = rowptr(i);
            unsigned j = 0;

            if constexpr (N >= 8) {
                for (; j + 8 <= N; j += 8) {
                    const T* __restrict__ Bp = colptr(j);
                    T* __restrict__ Cp = cptr(i, j);
                    compute_n8(Ap, Bp, Cp);
                }
            }
            if constexpr (N >= 4) {
                for (; j + 4 <= N; j += 4) {
                    const T* __restrict__ Bp = colptr(j);
                    T* __restrict__ Cp = cptr(i, j);
                    compute_n4(Ap, Bp, Cp);
                }
            }
            if constexpr (N >= 2) {
                if (j + 2 <= N) {
                    const T* __restrict__ Bp = colptr(j);
                    T* __restrict__ Cp = cptr(i, j);
                    compute_n2(Ap, Bp, Cp);
                    j += 2;
                }
            }
            if (j < N) {
                const T* __restrict__ Bp = colptr(j);
                T* __restrict__ Cp = cptr(i, j);
                compute_n1(Ap, Bp, Cp);
                ++j;
            }
        }
    }
}











