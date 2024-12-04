#include <gtest/gtest.h>

#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

namespace rlt = rl_tools;


using DEVICE = rlt::devices::DEVICE_FACTORY_CUDA<>;
using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;

TEST(RL_TOOLS_CONTAINER_TENSOR_CUDA, MAIN){
    DEVICE device;
    DEVICE_CPU device_cpu;
    rlt::init(device);
    using T = float;
    using TI = typename DEVICE::index_t;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 10>, true>> tensor, tensor_cpu;
    rlt::malloc(device, tensor);
    rlt::malloc(device_cpu, tensor_cpu);
    rlt::set_all(device, tensor, 1337);
    rlt::copy(device, device_cpu, tensor, tensor_cpu);
    rlt::print(device_cpu, tensor_cpu);
}
