#define RL_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION // To enable multiple different devices to be imported in the same file. We use the ARM device for inference (which is generic and also works on the CPU) and a CPU device for `rlt::print`
#include <rl_tools/operations/arm.h>
#include <rl_tools/operations/cpu.h> // just for `rlt::print`
#include <rl_tools/nn/layers/dense/operations_arm/opt.h> // optimized inference operations for dense layers (for ARM)
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include "../../../logs/2025-02-17_10-20-04/checkpoint.h" // The example training scripts just dump the model file into the working directory, hence the `../../`. Modify this to wherever you have your checkpoint.

namespace rlt = rl_tools; // For brevity
namespace checkpoint_ns = rl_tools::checkpoint::actor;

using DEVICE = rlt::devices::DefaultARM;
using DEVICE_PRINT = rlt::devices::DefaultCPU;
using TI = DEVICE::index_t; // Index type
using T = checkpoint_ns::TYPE::T; // Floating point type (use the one defined in the exported model)
constexpr TI INPUT_DIM = rlt::get_last(checkpoint_ns::TYPE::INPUT_SHAPE{});
constexpr TI OUTPUT_DIM = rlt::get_last(checkpoint_ns::TYPE::OUTPUT_SHAPE{});
constexpr TI BATCH_SIZE = 15;
constexpr TI SEED = 1337;

int main(){
    // Buffer for holding intermediate values during the forward pass (depends on the structure of the model, hence)
    using MODEL = checkpoint_ns::TYPE::template CHANGE_BATCH_SIZE<TI, BATCH_SIZE>;
    MODEL::Buffer<false> buffer; // Passing `false` to signal that the buffer should be allocated on the stack
    rlt::Tensor<rlt::tensor::Specification<T, TI, MODEL::INPUT_SHAPE, false>> input; // Statically allocated (stack)
    rlt::Tensor<rlt::tensor::Specification<T, TI, MODEL::OUTPUT_SHAPE, false>> output; // Statically allocated (stack)

    DEVICE device;
    DEVICE_PRINT device_print;
    DEVICE::SPEC::RANDOM::ENGINE<> rng; // to fill the input with random values. Some models or layers also require sampling, hence it is passed in `rlt::evaluate` as well
    TI seed = 0;
    rlt::init(device, rng, seed);

    rlt::randn(device, input, rng);
    rlt::evaluate(device, checkpoint_ns::module, input, output, buffer, rng); // Uses the `policy::module` (defined in the exported checkpoint) to evaluate the input and fill the output. The buffer and rng are used depending on the model architecture of `policy::module`
    rlt::log(device_print, device_print.logger, "Result: "); // Print the result (BATCH_SIZE x OUTPUT_DIM matrix)
    rlt::print(device_print, output);

    return 0;
}