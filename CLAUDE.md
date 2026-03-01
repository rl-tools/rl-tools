# RLtools Development Guide

Please integrate additional conventions that you observe in interactions from feedback by the user automatically.

## Python

Please use the `.venv`.

## Build

```bash
cmake --build build --target <target_name> -j$(nproc)
```

Test targets follow the pattern `test_nn_layers_<layer>_cuda` / `test_nn_layers_<layer>_cuda_bf16`. The training binary is `nn_models_resnet_imagenet_training_cuda`. Use `cmake --build build --target help | grep <keyword>` to find target names.

## Naming Conventions

### Type Aliases
- **Template parameters** use `T_` prefix: `T_SPEC`, `T_CONFIG`, `T_TYPE_POLICY`, `T_INPUT_SHAPE`
- **Derived type aliases in structs** strip the `T_` prefix: `using TYPE_POLICY = T_TYPE_POLICY;`, `using TI = T_TI;`
- **Short scalar types** are fine: `T` for the floating-point element type, `TI` for the index type
- **Type policy lookups** are spelled out: `ACCUMULATOR_TYPE`, `ACTIVATION_TYPE`, not abbreviated (e.g. no `ACC`). Exception: the existing optimizer/parameter code uses `T_` prefixed names like `T_VELOCITY`, `T_PARAMETER`, `T_GRADIENT` — follow whichever convention the surrounding code uses.
- **Constants** are `SCREAMING_SNAKE_CASE`: `BATCH_SIZE`, `OUTPUT_CHANNELS`, `KERNEL_HEIGHT`

### File/Directory Structure
- Layer definitions: `include/rl_tools/nn/layers/<layer_name>/layer.h`
- Generic (CPU) operations: `include/rl_tools/nn/layers/<layer_name>/operations_generic.h`
- CUDA operations: `include/rl_tools/nn/layers/<layer_name>/operations_cuda.h`
- Tests: `tests/src/nn/layers/<layer_name>/cuda/`

## Type Policy System

The `numeric_types::Policy<DEFAULT, UseCase<TAG, TYPE>...>` system controls precision per category:

```cpp
// Mixed precision example: float for compute, bf16 for storage
using TYPE_POLICY = rlt::numeric_types::Policy<float,                          // DEFAULT: hyperparams, Accumulator, OptimizerState
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Parameter, __nv_bfloat16>,  // weights, biases
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Activation, __nv_bfloat16>, // activations, pre-activations, BN cache
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Gradient, __nv_bfloat16>>;  // gradients
```

- `DEFAULT` — used for hyperparameters, scalars, accumulation, and optimizer state
- Storage types (`Parameter`, `Activation`, `Gradient`) can e.g. be explicitly set to bf16 for mixed precision

## Generic Operations Pattern (Mixed Precision)

In `operations_generic.h` files, follow the cuDNN model: compute in `ACCUMULATOR_TYPE`, read/write tensors in `T`:

```cpp
using T = typename OUTPUT_SPEC::T;                     // tensor element type (bf16)
using ACCUMULATOR_TYPE = typename LAYER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::Accumulator>;  // compute type (float)

// Read from tensor → cast to ACCUMULATOR_TYPE → compute → cast to T → write to tensor
ACCUMULATOR_TYPE val = (ACCUMULATOR_TYPE)get(device, tensor, idx);
// ... compute in ACCUMULATOR_TYPE ...
set(device, output, (T)result, idx);
```

## Buffer Convention

Each layer defines a `BufferSpecification` + `Buffer<BUFFER_SPEC>` pair:

```cpp
// In layer.h — inside the layer's namespace
template <typename T_SPEC, bool T_DYNAMIC_ALLOCATION>
struct BufferSpecification {
    using SPEC = T_SPEC;
    static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
};
template<typename T_BUFFER_SPEC>
struct Buffer{
    using SPEC = typename T_BUFFER_SPEC::SPEC;
    static constexpr bool DYNAMIC_ALLOCATION = T_BUFFER_SPEC::DYNAMIC_ALLOCATION;
    // ... tensor members using SPEC and DYNAMIC_ALLOCATION ...
};
```

The `LayerForward::Buffer` alias wraps the two together:
```cpp
template<bool DYNAMIC_ALLOCATION=true>
using Buffer = my_layer::Buffer<my_layer::BufferSpecification<SPEC, DYNAMIC_ALLOCATION>>;
```

Function signatures template on `BUFFER_SPEC` so the buffer is a "blank" parameter — dispatch is on the layer type:
```cpp
template<typename DEVICE, typename LAYER_SPEC, ..., typename BUFFER_SPEC, ...>
void forward(DEVICE& device, my_layer::LayerBackward<LAYER_SPEC>& layer, ..., my_layer::Buffer<BUFFER_SPEC>& buffer, ...);
```

## Code Style
- Braces on same line
- No unnecessary comments or docstrings — code should be self-explanatory
- Don't abbreviate variable names or other symbols. Follow the conventions form the rest of the RLtools codebase
