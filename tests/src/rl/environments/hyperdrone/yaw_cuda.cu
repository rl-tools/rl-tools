#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/l2f/multirotor.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/yaw/operations_cuda.h>

#include <gtest/gtest.h>
#include <metra/metra.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>

namespace rl_tools::test_hyperdrone_yaw_cuda{
    using DEVICE = devices::DefaultCPU;
    using DEVICE_GPU = devices::DEVICE_FACTORY_CUDA<>;
    using TI = DEVICE_GPU::index_t;
    using T = float;
    using STATE = rl::environments::l2f::StateBase<rl::environments::l2f::StateSpecification<T, TI>>;

    template<typename T_TARGET>
    void check_labels(T tolerance){
        int device_count = 0;
        if(cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0){
            GTEST_SKIP() << "CUDA device unavailable";
        }
        DEVICE device;
        DEVICE_GPU device_gpu;
        init(device);
        init(device_gpu);
        constexpr TI N = 37;
        using STATES = Tensor<tensor::Specification<STATE, TI, tensor::Shape<TI, N>>>;
        using TARGETS = Tensor<tensor::Specification<T_TARGET, TI, tensor::Shape<TI, N + 2, 4>>>;
        STATES states, states_gpu;
        TARGETS targets, targets_gpu;
        malloc(device, states);
        malloc(device_gpu, states_gpu);
        malloc(device, targets);
        malloc(device_gpu, targets_gpu);
        for(TI i = 0; i < N; i++){
            const T yaw = -2.3f + 0.11f * i;
            const T cy = std::cos(yaw / 2), sy = std::sin(yaw / 2);
            const T cp = std::cos(0.2f), sp = std::sin(0.2f);
            const T cr = std::cos(-0.15f), sr = std::sin(-0.15f);
            STATE state{};
            state.orientation[0] = cr * cp * cy + sr * sp * sy;
            state.orientation[1] = sr * cp * cy - cr * sp * sy;
            state.orientation[2] = cr * sp * cy + sr * cp * sy;
            state.orientation[3] = cr * cp * sy - sr * sp * cy;
            set(device, states, state, i);
        }
        set_all(device, targets, static_cast<T_TARGET>(17));
        copy(device, device_gpu, states, states_gpu);
        copy(device, device_gpu, targets, targets_gpu);
        auto rows = view_range(device_gpu, targets_gpu, (TI)1, tensor::ViewSpec<0, N>{});
        auto labels = view_range(device_gpu, rows, (TI)1, tensor::ViewSpec<1, 2>{});
        yaw_targets(device_gpu, states_gpu, labels);
        copy(device_gpu, device, targets_gpu, targets);
        T maximum_error = 0;
        for(TI i = 0; i < N + 2; i++){
            for(TI channel = 0; channel < 4; channel++){
                const T actual = static_cast<T>(get(device, targets, i, channel));
                if(i == 0 || i == N + 1 || channel == 0 || channel == 3){
                    EXPECT_EQ(actual, 17);
                } else {
                    const T yaw = -2.3f + 0.11f * (i - 1);
                    const T expected = channel == 1 ? std::cos(yaw) : std::sin(yaw);
                    EXPECT_NEAR(actual, expected, tolerance);
                    maximum_error = std::max(maximum_error, std::abs(actual - expected));
                }
            }
        }
        metra::log(sizeof(T_TARGET) == sizeof(float) ? "hyperdrone/yaw/cuda/float_label_max_error" : "hyperdrone/yaw/cuda/bf16_label_max_error", static_cast<double>(maximum_error));
        free(device_gpu, targets_gpu);
        free(device, targets);
        free(device_gpu, states_gpu);
        free(device, states);
        free(device_gpu);
    }
}

TEST(RL_TOOLS_HYPERDRONE_YAW_CUDA, FLOAT_STRIDED_PARTIAL_BLOCK){
    rl_tools::test_hyperdrone_yaw_cuda::check_labels<float>(3e-6f);
}

TEST(RL_TOOLS_HYPERDRONE_YAW_CUDA, BF16_STRIDED_PARTIAL_BLOCK){
    rl_tools::test_hyperdrone_yaw_cuda::check_labels<__nv_bfloat16>(0.002f);
}
