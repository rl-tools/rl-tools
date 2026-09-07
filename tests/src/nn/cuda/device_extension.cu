#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/renderer.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_cuda.h>
#include <rl_tools/nn/optimizers/sgd/operations_cuda.h>
#include <metra/metra.h>

#include <gtest/gtest.h>
#include <mutex>
#include <type_traits>

namespace rlt = rl_tools;

namespace test_device_extension {
    struct Logger: rlt::devices::logging::CPU {
        std::mutex mutex;
    };
    using HOST_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, Logger>;
    using HOST = rlt::devices::CPU<HOST_SPEC>;
    using GPU_SPEC = rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, HOST>;
    using GPU = rlt::devices::CUDA<GPU_SPEC>;
    using TAG = rlt::devices::cuda::TAG<GPU, true>;
    static_assert(!std::is_copy_constructible<HOST>::value);
    static_assert(!std::is_copy_constructible<GPU>::value);
    static_assert(std::is_trivially_copyable<TAG>::value && TAG::TAG && TAG::KERNEL);
    static_assert(!rlt::rendering::raytracing::device::HasRendering<TAG>::value);

    template <bool ADAM, bool MASTER>
    void optimizer_updates() {
        using namespace rlt;
        using TI = GPU::index_t;
        using TYPE_POLICY = std::conditional_t<MASTER,
            rlt::numeric_types::Policy<float, rlt::numeric_types::UseCase<rlt::numeric_types::categories::MasterParameter, double>>,
            rlt::numeric_types::Policy<float>>;
        using SHAPE = rlt::tensor::Shape<TI, 2, 3>;
        using PARAMETER = std::conditional_t<ADAM,
            rlt::nn::parameters::Adam::Instance<rlt::nn::parameters::Adam::Specification<TYPE_POLICY, TI, SHAPE, rlt::nn::parameters::groups::Normal, rlt::nn::parameters::categories::Weights, true>>,
            rlt::nn::parameters::SGD::Instance<rlt::nn::parameters::SGD::Specification<TYPE_POLICY, TI, SHAPE, rlt::nn::parameters::groups::Normal, rlt::nn::parameters::categories::Weights, true>>>;
        using OPTIMIZER = std::conditional_t<ADAM,
            rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI>>,
            rlt::nn::optimizers::SGD<rlt::nn::optimizers::sgd::Specification<TYPE_POLICY, TI>>>;

        GPU gpu;
        auto& host = gpu.rendering;
        init(host);
        init(gpu);
        ASSERT_EQ(&get_rendering_device(gpu), &host);
        ASSERT_EQ(&get_rendering_device(host), &host);
        ASSERT_TRUE(host.initialized);

        PARAMETER cpu_parameter, gpu_parameter, result;
        OPTIMIZER cpu_optimizer, gpu_optimizer;
        malloc(host, cpu_parameter);
        malloc(gpu, gpu_parameter);
        malloc(host, result);
        malloc(host, cpu_optimizer);
        malloc(gpu, gpu_optimizer);
        init(host, cpu_optimizer);
        init(gpu, gpu_optimizer);
        set_all(host, cpu_parameter.parameters, 1.0f);
        copy(host, gpu, cpu_parameter.parameters, gpu_parameter.parameters);

        for(TI step_i = 0; step_i < 6; step_i++) {
            if(step_i == 0 || step_i == 3) {
                reset_optimizer_state(host, cpu_optimizer, cpu_parameter);
                reset_optimizer_state(gpu, gpu_optimizer, gpu_parameter);
            }
            set_all(host, cpu_parameter.gradient, step_i % 2 == 0 ? 0.25f : -0.5f);
            copy(host, gpu, cpu_parameter.gradient, gpu_parameter.gradient);
            step(host, cpu_optimizer, cpu_parameter);
            step(gpu, gpu_optimizer, gpu_parameter);
            ASSERT_EQ(cudaStreamSynchronize(gpu.stream), cudaSuccess);
            copy(gpu, host, gpu_parameter, result);
            const auto error = abs_diff(host, cpu_parameter, result);
            EXPECT_LT(error, 1e-5f) << "step " << step_i;
            metra::log(ADAM ? (MASTER ? "cuda/device_extension/adam_master/abs_diff" : "cuda/device_extension/adam/abs_diff") : (MASTER ? "cuda/device_extension/sgd_master/abs_diff" : "cuda/device_extension/sgd/abs_diff"), static_cast<double>(error));
        }

        EXPECT_NE(get(host, result.parameters, 0, 0), 1.0f);

        free(host, cpu_optimizer);
        free(gpu, gpu_optimizer);
        free(host, cpu_parameter);
        free(gpu, gpu_parameter);
        free(host, result);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
        ASSERT_EQ(cudnnDestroy(gpu.cudnn_handle), CUDNN_STATUS_SUCCESS);
#endif
        ASSERT_EQ(cublasDestroy(gpu.handle), CUBLAS_STATUS_SUCCESS);
        ASSERT_EQ(cudaStreamDestroy(gpu.stream), cudaSuccess);
    }
}

TEST(RL_TOOLS_CUDA_DEVICE_EXTENSION, ADAM) {
    test_device_extension::optimizer_updates<true, false>();
}
TEST(RL_TOOLS_CUDA_DEVICE_EXTENSION, ADAM_MASTER_PARAMETERS) {
    test_device_extension::optimizer_updates<true, true>();
}
TEST(RL_TOOLS_CUDA_DEVICE_EXTENSION, SGD) {
    test_device_extension::optimizer_updates<false, false>();
}
TEST(RL_TOOLS_CUDA_DEVICE_EXTENSION, SGD_MASTER_PARAMETERS) {
    test_device_extension::optimizer_updates<false, true>();
}
