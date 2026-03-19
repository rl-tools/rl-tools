#pragma once

#include <cuda_runtime.h>

// Shared CUDA kernels for yaw prediction model forward/backward orchestration.
// Included by .cu files that need the model-level evaluate/forward/backward.

namespace rl_tools::rendering::raytracing::yaw_prediction::cuda_kernels {

    // Concatenate two tensors [N, D_A] and [N, D_B] into [N, D_A + D_B] along last dim
    template<typename T>
    __global__ void concatenate_kernel(
        const T* __restrict__ a, int d_a,
        const T* __restrict__ b, int d_b,
        T* __restrict__ output,
        int n
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int d_out = d_a + d_b;
        const int total = n * d_out;
        if (idx >= total) return;
        const int row = idx / d_out;
        const int col = idx % d_out;
        if (col < d_a) {
            output[idx] = a[row * d_a + col];
        } else {
            output[idx] = b[row * d_b + (col - d_a)];
        }
    }

    // Split [N, D_A + D_B] into [N, D_A] and [N, D_B] along last dim
    template<typename T>
    __global__ void split_kernel(
        const T* __restrict__ input, int d_a, int d_b,
        T* __restrict__ a,
        T* __restrict__ b,
        int n
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int d_in = d_a + d_b;
        const int total = n * d_in;
        if (idx >= total) return;
        const int row = idx / d_in;
        const int col = idx % d_in;
        if (col < d_a) {
            a[row * d_a + col] = input[idx];
        } else {
            b[row * d_b + (col - d_a)] = input[idx];
        }
    }

    // Element-wise add: dst[i] += src[i]
    template<typename T>
    __global__ void add_tensors_kernel(
        const T* __restrict__ src,
        T* __restrict__ dst,
        int n
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        dst[idx] = (T)((float)dst[idx] + (float)src[idx]);
    }

    // Convert RGBA uint32 framebuffer pixels to tensor [BATCH, H, W, 3] normalized to [0,1]
    template<typename T>
    __global__ void rgba_to_activation_kernel(
        const uint32_t* __restrict__ framebuffer,
        T* __restrict__ output,
        int cam_width, int cam_height, int camera_offset, int dst_offset
    ) {
        const int sample_idx = blockIdx.x;
        const int camera_idx = camera_offset + sample_idx;
        const int cam_pixels = cam_width * cam_height;

        const uint32_t* src = framebuffer + camera_idx * cam_pixels;
        T* dst = output + (dst_offset + sample_idx) * cam_height * cam_width * 3;

        for (int pixel = threadIdx.x; pixel < cam_pixels; pixel += blockDim.x) {
            const uint32_t rgba = src[pixel];
            const float r = static_cast<float>((rgba >>  0) & 0xFF) / 255.0f;
            const float g = static_cast<float>((rgba >>  8) & 0xFF) / 255.0f;
            const float b = static_cast<float>((rgba >> 16) & 0xFF) / 255.0f;

            const int out_base = pixel * 3;
            dst[out_base + 0] = (T)r;
            dst[out_base + 1] = (T)g;
            dst[out_base + 2] = (T)b;
        }
    }

} // namespace rl_tools::rendering::raytracing::yaw_prediction::cuda_kernels

namespace rl_tools {

    // ======================== evaluate_cuda (2-input, CUDA-optimized) ========================

    template<typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE>
    void evaluate_cuda(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model, const INPUT_A& input_a, const INPUT_B& input_b, OUTPUT& output, rendering::raytracing::yaw_prediction::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode) {
        using TI = typename SPEC::TI;

        // 1. Early encoders
        evaluate(device, model.early_encoder_a, input_a, buffer.features_a, buffer.buffer_early_a, rng, mode);
        evaluate(device, model.early_encoder_b, input_b, buffer.features_b, buffer.buffer_early_b, rng, mode);

        // 2. Mid conv A
        evaluate(device, model.mid_conv_a, buffer.features_a, buffer.intermediate_mid_a, buffer.buffer_mid_a, rng, mode);

        // 3. Mid conv B (cross-conv or independent conv)
        if constexpr (SPEC::MODEL_CONFIG::USE_CROSS_CONV) {
            using KW_4D = typename SPEC::KERNEL_WEIGHTS_4D_SHAPE;
            auto kw_4d = view_memory<KW_4D>(device, buffer.intermediate_mid_a);
            evaluate(device, model.mid_conv_b, buffer.features_b, kw_4d, buffer.intermediate_mid_b, buffer.buffer_mid_b, rng, mode);
        } else {
            evaluate(device, model.mid_conv_b, buffer.features_b, buffer.intermediate_mid_b, buffer.buffer_mid_b, rng, mode);
        }

        // 4. Late encoders
        evaluate(device, model.late_encoder_a, buffer.intermediate_mid_a, buffer.intermediate_a, buffer.buffer_late_a, rng, mode);
        evaluate(device, model.late_encoder_b, buffer.intermediate_mid_b, buffer.intermediate_b, buffer.buffer_late_b, rng, mode);

        // 5. Concatenate (CUDA kernel)
        {
            constexpr TI LATE_DIM = SPEC::LATE_LAST_DIM;
            constexpr TI CONCAT_DIM = LATE_DIM * 2;
            constexpr TI LATE_TOTAL = product(typename SPEC::LATE_OUTPUT_SHAPE{});
            constexpr TI CONCAT_ROWS = LATE_TOTAL / LATE_DIM;
            constexpr int total = CONCAT_ROWS * CONCAT_DIM;
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            rendering::raytracing::yaw_prediction::cuda_kernels::concatenate_kernel<<<blocks, threads, 0, device.stream>>>(
                buffer.intermediate_a._data, (int)LATE_DIM,
                buffer.intermediate_b._data, (int)LATE_DIM,
                buffer.concatenated._data, (int)CONCAT_ROWS
            );
        }

        // 6. Head
        evaluate(device, model.head, buffer.concatenated, output, buffer.buffer_head, rng, mode);
    }

    // ======================== forward_cuda (2-input, CUDA-optimized training) ========================

    template<typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename BUFFER_SPEC, typename RNG, typename MODE>
    void forward_cuda(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model, INPUT_A& input_a, INPUT_B& input_b, rendering::raytracing::yaw_prediction::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode) {
        using TI = typename SPEC::TI;

        // 1. Early encoders
        forward(device, model.early_encoder_a, input_a, buffer.buffer_early_a, rng, mode);
        forward(device, model.early_encoder_b, input_b, buffer.buffer_early_b, rng, mode);

        // 2. Save features (contiguous copies for backward)
        {
            auto fa = rl_tools::output(device, model.early_encoder_a);
            auto fb = rl_tools::output(device, model.early_encoder_b);
            copy(device, device, fa, buffer.features_a);
            copy(device, device, fb, buffer.features_b);
        }

        // 3. Mid conv A
        forward(device, model.mid_conv_a, buffer.features_a, buffer.buffer_mid_a, rng, mode);

        // 4. Mid conv B (cross-conv or independent conv)
        if constexpr (SPEC::MODEL_CONFIG::USE_CROSS_CONV) {
            using KW_4D = typename SPEC::KERNEL_WEIGHTS_4D_SHAPE;
            auto kw_4d = view_memory<KW_4D>(device, rl_tools::output(device, model.mid_conv_a));
            forward(device, model.mid_conv_b, buffer.features_b, kw_4d, buffer.buffer_mid_b, rng, mode);
        } else {
            forward(device, model.mid_conv_b, buffer.features_b, buffer.buffer_mid_b, rng, mode);
        }

        // 5. Late encoders
        {
            auto mid_a_out = rl_tools::output(device, model.mid_conv_a);
            auto mid_b_out = rl_tools::output(device, model.mid_conv_b);
            forward(device, model.late_encoder_a, mid_a_out, buffer.buffer_late_a, rng, mode);
            forward(device, model.late_encoder_b, mid_b_out, buffer.buffer_late_b, rng, mode);
        }

        // 6. Copy late encoder outputs and concatenate (CUDA kernel)
        {
            auto la = rl_tools::output(device, model.late_encoder_a);
            auto lb = rl_tools::output(device, model.late_encoder_b);
            copy(device, device, la, buffer.intermediate_a);
            copy(device, device, lb, buffer.intermediate_b);
        }
        {
            constexpr TI LATE_DIM = SPEC::LATE_LAST_DIM;
            constexpr TI CONCAT_DIM = LATE_DIM * 2;
            constexpr TI LATE_TOTAL = product(typename SPEC::LATE_OUTPUT_SHAPE{});
            constexpr TI CONCAT_ROWS = LATE_TOTAL / LATE_DIM;
            constexpr int total = CONCAT_ROWS * CONCAT_DIM;
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            rendering::raytracing::yaw_prediction::cuda_kernels::concatenate_kernel<<<blocks, threads, 0, device.stream>>>(
                buffer.intermediate_a._data, (int)LATE_DIM,
                buffer.intermediate_b._data, (int)LATE_DIM,
                buffer.concatenated._data, (int)CONCAT_ROWS
            );
        }

        // 7. Head
        forward(device, model.head, buffer.concatenated, buffer.buffer_head, rng, mode);
        {
            auto head_output = rl_tools::output(device, model.head);
            copy(device, device, head_output, model.output);
        }
    }

    // ======================== backward_full_cuda (2-input, CUDA-optimized) ========================

    template<typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename D_OUTPUT, typename D_INPUT_A, typename D_INPUT_B, typename BUFFER_SPEC>
    void backward_full_cuda(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleGradient<SPEC>& model, INPUT_A& input_a, INPUT_B& input_b, D_OUTPUT& d_output, D_INPUT_A& d_input_a, D_INPUT_B& d_input_b, rendering::raytracing::yaw_prediction::Buffer<BUFFER_SPEC>& buffer) {
        using TI = typename SPEC::TI;
        constexpr TI LATE_DIM = SPEC::LATE_LAST_DIM;
        constexpr TI CONCAT_DIM = LATE_DIM * 2;
        constexpr TI LATE_TOTAL = product(typename SPEC::LATE_OUTPUT_SHAPE{});
        constexpr TI CONCAT_ROWS = LATE_TOTAL / LATE_DIM;

        // 1. Backward head
        backward_full(device, model.head, buffer.concatenated, d_output, buffer.d_concatenated, buffer.buffer_head);

        // 2. Split d_concatenated (CUDA kernel)
        {
            constexpr int total = CONCAT_ROWS * CONCAT_DIM;
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            rendering::raytracing::yaw_prediction::cuda_kernels::split_kernel<<<blocks, threads, 0, device.stream>>>(
                buffer.d_concatenated._data, (int)LATE_DIM, (int)LATE_DIM,
                buffer.d_output_a._data,
                buffer.d_output_b._data,
                (int)CONCAT_ROWS
            );
        }

        // 3. Backward late encoders
        {
            auto mid_a_out = rl_tools::output(device, model.mid_conv_a);
            auto mid_b_out = rl_tools::output(device, model.mid_conv_b);
            backward_full(device, model.late_encoder_a, mid_a_out, buffer.d_output_a, buffer.d_mid_a, buffer.buffer_late_a);
            backward_full(device, model.late_encoder_b, mid_b_out, buffer.d_output_b, buffer.d_mid_b, buffer.buffer_late_b);
        }

        // 4. Backward mid conv B and accumulate cross-conv gradient
        if constexpr (SPEC::MODEL_CONFIG::USE_CROSS_CONV) {
            using KW_4D = typename SPEC::KERNEL_WEIGHTS_4D_SHAPE;
            auto kw_4d = view_memory<KW_4D>(device, rl_tools::output(device, model.mid_conv_a));
            backward_full(device, model.mid_conv_b,
                buffer.features_b, kw_4d, buffer.d_mid_b,
                buffer.d_features_b, buffer.d_kw_for_b, buffer.buffer_mid_b);

            // Accumulate kernel weight gradient into d_mid_a
            constexpr TI MID_OUTPUT_TOTAL = product(typename SPEC::MID_CONV_A_OUTPUT_SHAPE{});
            {
                constexpr int threads = 256;
                constexpr int blocks = (MID_OUTPUT_TOTAL + threads - 1) / threads;
                rendering::raytracing::yaw_prediction::cuda_kernels::add_tensors_kernel<<<blocks, threads, 0, device.stream>>>(
                    buffer.d_kw_for_b._data, buffer.d_mid_a._data, (int)MID_OUTPUT_TOTAL);
            }
        } else {
            backward_full(device, model.mid_conv_b,
                buffer.features_b, buffer.d_mid_b,
                buffer.d_features_b, buffer.buffer_mid_b);
        }

        // 5. Backward mid conv A
        backward_full(device, model.mid_conv_a,
            buffer.features_a, buffer.d_mid_a,
            buffer.d_features_a, buffer.buffer_mid_a);

        // 6. Backward early encoders
        backward_full(device, model.early_encoder_a, input_a, buffer.d_features_a, d_input_a, buffer.buffer_early_a);
        backward_full(device, model.early_encoder_b, input_b, buffer.d_features_b, d_input_b, buffer.buffer_early_b);
    }

    // ======================== evaluate_cuda (parallel::Build, CUDA-optimized) ========================

    template<typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename OUTPUT, typename BUFFER_SPEC, typename RNG, typename MODE>
    void evaluate_cuda(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& model, const INPUT_A& input_a, const INPUT_B& input_b, OUTPUT& output, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode) {
        using TI = typename SPEC::TI;

        evaluate(device, model.pipeline_a, input_a, buffer.intermediate_a, buffer.buffer_a, rng, mode);
        evaluate(device, model.pipeline_b, input_b, buffer.intermediate_b, buffer.buffer_b, rng, mode);

        {
            constexpr TI LAST_DIM_A = SPEC::LAST_DIM_A;
            constexpr TI LAST_DIM_B = SPEC::LAST_DIM_B;
            constexpr TI TOTAL_A = product(typename SPEC::OUTPUT_SHAPE_A{});
            constexpr TI LEADING = TOTAL_A / LAST_DIM_A;
            constexpr int total = LEADING * (LAST_DIM_A + LAST_DIM_B);
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            rendering::raytracing::yaw_prediction::cuda_kernels::concatenate_kernel<<<blocks, threads, 0, device.stream>>>(
                buffer.intermediate_a._data, (int)LAST_DIM_A,
                buffer.intermediate_b._data, (int)LAST_DIM_B,
                buffer.concatenated._data, (int)LEADING
            );
        }

        if constexpr(SPEC::HAS_HEAD) {
            evaluate(device, model.head, buffer.concatenated, output, buffer.head_buffer, rng, mode);
        } else {
            copy(device, device, buffer.concatenated, output);
        }
    }

    // ======================== forward_cuda (parallel::Build, CUDA-optimized training) ========================

    template<typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename BUFFER_SPEC, typename RNG, typename MODE>
    void forward_cuda(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& model, INPUT_A& input_a, INPUT_B& input_b, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer, RNG& rng, const Mode<MODE>& mode) {
        using TI = typename SPEC::TI;

        forward(device, model.pipeline_a, input_a, buffer.buffer_a, rng, mode);
        forward(device, model.pipeline_b, input_b, buffer.buffer_b, rng, mode);

        {
            auto output_a = rl_tools::output(device, model.pipeline_a);
            auto output_b = rl_tools::output(device, model.pipeline_b);
            copy(device, device, output_a, buffer.intermediate_a);
            copy(device, device, output_b, buffer.intermediate_b);
        }

        {
            constexpr TI LAST_DIM_A = SPEC::LAST_DIM_A;
            constexpr TI LAST_DIM_B = SPEC::LAST_DIM_B;
            constexpr TI TOTAL_A = product(typename SPEC::OUTPUT_SHAPE_A{});
            constexpr TI LEADING = TOTAL_A / LAST_DIM_A;
            constexpr int total = LEADING * (LAST_DIM_A + LAST_DIM_B);
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            rendering::raytracing::yaw_prediction::cuda_kernels::concatenate_kernel<<<blocks, threads, 0, device.stream>>>(
                buffer.intermediate_a._data, (int)LAST_DIM_A,
                buffer.intermediate_b._data, (int)LAST_DIM_B,
                buffer.concatenated._data, (int)LEADING
            );
        }

        if constexpr(SPEC::HAS_HEAD) {
            forward(device, model.head, buffer.concatenated, buffer.head_buffer, rng, mode);
            auto head_output = rl_tools::output(device, model.head);
            copy(device, device, head_output, model.output);
        } else {
            copy(device, device, buffer.concatenated, model.output);
        }
    }

    // ======================== backward_full_cuda (parallel::Build, CUDA-optimized) ========================

    template<typename DEVICE, typename SPEC, typename INPUT_A, typename INPUT_B, typename D_OUTPUT, typename D_INPUT_A, typename D_INPUT_B, typename BUFFER_SPEC>
    void backward_full_cuda(DEVICE& device, nn_models::parallel::ModuleGradient<SPEC>& model, INPUT_A& input_a, INPUT_B& input_b, D_OUTPUT& d_output, D_INPUT_A& d_input_a, D_INPUT_B& d_input_b, nn_models::parallel::ModuleBuffer<BUFFER_SPEC>& buffer) {
        using TI = typename SPEC::TI;
        constexpr TI LAST_DIM_A = SPEC::LAST_DIM_A;
        constexpr TI LAST_DIM_B = SPEC::LAST_DIM_B;
        constexpr TI TOTAL_A = product(typename SPEC::OUTPUT_SHAPE_A{});
        constexpr TI LEADING = TOTAL_A / LAST_DIM_A;

        if constexpr(SPEC::HAS_HEAD) {
            backward_full(device, model.head, buffer.concatenated, d_output, buffer.d_concatenated, buffer.head_buffer);
        }

        {
            constexpr int total = LEADING * (LAST_DIM_A + LAST_DIM_B);
            constexpr int threads = 256;
            constexpr int blocks = (total + threads - 1) / threads;
            if constexpr(SPEC::HAS_HEAD) {
                rendering::raytracing::yaw_prediction::cuda_kernels::split_kernel<<<blocks, threads, 0, device.stream>>>(
                    buffer.d_concatenated._data, (int)LAST_DIM_A, (int)LAST_DIM_B,
                    buffer.d_output_a._data,
                    buffer.d_output_b._data,
                    (int)LEADING
                );
            } else {
                rendering::raytracing::yaw_prediction::cuda_kernels::split_kernel<<<blocks, threads, 0, device.stream>>>(
                    d_output._data, (int)LAST_DIM_A, (int)LAST_DIM_B,
                    buffer.d_output_a._data,
                    buffer.d_output_b._data,
                    (int)LEADING
                );
            }
        }

        backward_full(device, model.pipeline_a, input_a, buffer.d_output_a, d_input_a, buffer.buffer_a);
        backward_full(device, model.pipeline_b, input_b, buffer.d_output_b, d_input_b, buffer.buffer_b);
    }

} // namespace rl_tools
