#include <rl_tools/operations/cpu.h>

#include "policy.h"

#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>


namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
constexpr bool DYNAMIC_ALLOCATION = true;
using T = rl_tools::checkpoint::actor::TYPE::TYPE_POLICY::DEFAULT;
using TI = typename DEVICE::index_t;
constexpr TI SEED = 0;

int main(){
    DEVICE device;
    RNG rng;
    rl_tools::checkpoint::actor::TYPE::Buffer<DYNAMIC_ALLOCATION> policy_buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rl_tools::checkpoint::actor::TYPE::OUTPUT_SHAPE>> output;
    rlt::malloc(device);
    rlt::malloc(device, rng);
    rlt::malloc(device, policy_buffer);
    rlt::malloc(device, output);
    rlt::init(device);
    rlt::init(device, rng, SEED);
    rlt::evaluate(device, rl_tools::checkpoint::actor::module, rl_tools::checkpoint::example::input::container, output, policy_buffer, rng);
    T abs_diff = rlt::abs_diff(device, rl_tools::checkpoint::example::output::container, output) / decltype(output)::SPEC::SIZE;
    std::cout << "Difference base <-> orig: " << abs_diff << std::endl;
    return 0;
}