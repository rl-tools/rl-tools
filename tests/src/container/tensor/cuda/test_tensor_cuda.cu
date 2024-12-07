#include <gtest/gtest.h>

#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

namespace rlt = rl_tools;


using DEVICE = rlt::devices::DEVICE_FACTORY_CUDA<>;
using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;


template <typename T, typename TI, typename SHAPE, typename DEVICE, typename DEVICE_CPU>
void test(DEVICE& device, DEVICE_CPU& device_cpu, T epsilon) {
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, true>> tensor, tensor_cpu, target;
    rlt::malloc(device, tensor);
    rlt::malloc(device_cpu, tensor_cpu);
    rlt::malloc(device_cpu, target);
    {
        rlt::set_all(device, tensor, 1337);
        rlt::set_all(device_cpu, target, 1337);
        rlt::copy(device, device_cpu, tensor, tensor_cpu);
        T diff = rlt::abs_diff(device_cpu, tensor_cpu, target);
        std::cout << "set_all diff: " << diff << std::endl;
        ASSERT_EQ(diff, 0);
    }
    {
        rlt::set_all(device, tensor, 1.3);
        rlt::set_all(device_cpu, target, 1.3);
        rlt::exp(device, tensor);
        rlt::exp(device_cpu, target);
        rlt::copy(device, device_cpu, tensor, tensor_cpu);
        T diff = rlt::abs_diff(device_cpu, tensor_cpu, target);
        if constexpr(SHAPE::LENGTH == 1 && SHAPE::template GET<0> < 100) {
            rlt::print(device_cpu, target);
        }
        std::cout << "exp diff: " << diff << std::endl;
        ASSERT_LT(diff, epsilon);
    }
    {
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1>, true>> result, result_cpu, result_target;
        rlt::malloc(device, result);
        rlt::malloc(device_cpu, result_cpu);
        rlt::malloc(device_cpu, result_target);
        rlt::set_all(device, tensor, 1.3);
        rlt::set_all(device_cpu, tensor_cpu, 1.3);
        rlt::cast_reduce_sum<T>(device, tensor, result);
        rlt::cast_reduce_sum<T>(device_cpu, tensor_cpu, result_target);
        rlt::copy(device, device_cpu, result, result_cpu);
        T diff = rlt::abs_diff(device_cpu, result_cpu, result_target);
        std::cout << "Result target: " << std::endl;
        rlt::print(device_cpu, result_target);
        std::cout << "unary associative reduce: sum diff: " << diff << std::endl;
        ASSERT_LT(diff, epsilon);
        rlt::free(device, result);
        rlt::free(device_cpu, result_cpu);
        rlt::free(device_cpu, result_target);
    }
}

TEST(RL_TOOLS_CONTAINER_TENSOR_CUDA, MAIN){
    DEVICE device;
    DEVICE_CPU device_cpu;
    rlt::init(device);
    using T = float;
    using TI = typename DEVICE::index_t;
    T epsilon = 1e-6;
    test<T, TI, rlt::tensor::Shape<TI, 1>>(device, device_cpu, epsilon);
    test<T, TI, rlt::tensor::Shape<TI, 2>>(device, device_cpu, epsilon);
    test<T, TI, rlt::tensor::Shape<TI, 10>>(device, device_cpu, epsilon);
    test<T, TI, rlt::tensor::Shape<TI, 10, 1>>(device, device_cpu, epsilon);
    test<T, TI, rlt::tensor::Shape<TI, 1, 10>>(device, device_cpu, epsilon);
    test<T, TI, rlt::tensor::Shape<TI, 10, 10>>(device, device_cpu, epsilon);
    test<T, TI, rlt::tensor::Shape<TI, 10, 10, 1>>(device, device_cpu, epsilon);
    test<T, TI, rlt::tensor::Shape<TI, 10, 1, 10>>(device, device_cpu, epsilon);
    test<T, TI, rlt::tensor::Shape<TI, 1, 10, 10>>(device, device_cpu, epsilon);
    test<T, TI, rlt::tensor::Shape<TI, 10, 10, 10>>(device, device_cpu, epsilon);
}

