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

## Capability System

Layers and models use a capability tag (`Forward`, `Backward`, `Gradient`) to control what data they store:
- `ModuleForward<SPEC>` — weights only (for inference)
- `ModuleBackward<SPEC>` — inherits Forward (for backward_input without gradient accumulation)
- `ModuleGradient<SPEC>` — inherits Backward, adds output tensor (for training with gradient accumulation)

A `BuildModuleType<CAPABILITY, SPEC>` struct selects the appropriate module type based on the capability tag. The `Build` entry point inherits from the resolved type and provides `CHANGE_CAPABILITY`.

## Composable Model Pattern (BindConfiguration / Layer alias)

Any type that can be used as a building block in `sequential::Module` or `parallel::Build` must expose:
```cpp
template <typename CAPABILITY, typename INPUT_SHAPE>
using Layer = ConcreteBuiltType<CAPABILITY, INPUT_SHAPE>;
```

This is how sequential resolves each layer: `HEAD::template Layer<CAPABILITY, INPUT_SHAPE>`. Examples:
- `nn::layers::dense::BindConfiguration<CONFIG>` — wraps a dense layer config, `Layer` resolves to the built dense layer
- `nn_models::mlp::BindConfiguration<CONFIG>` — wraps an MLP config, `Layer` resolves to the built MLP
- `nn_models::sequential::Module<LAYERS...>` — `Layer` resolves to `sequential::Build<CAPABILITY, Module<LAYERS...>, INPUT_SHAPE>`

This means `parallel::Build` accepts any such type (dense, MLP, sequential, other composites) on either side.

## nn_models Structure

Models live in `include/rl_tools/nn_models/<model_name>/`:
- `model.h` — type definitions (Specification, ModuleForward/Backward/Gradient, Buffer, State, Build)
- `operations_generic.h` — all operations (malloc/free, init_weights, evaluate, forward, backward variants, zero_gradient, update, _reset_optimizer_state, copy, abs_diff, is_nan, output)

## Delegating Operations for Composite Models

Composite models (sequential, parallel, resnet_block) delegate all operations to their sub-modules. The pattern:
```cpp
template <typename DEVICE, typename SPEC, typename RNG>
void init_weights(DEVICE& device, nn_models::my_model::ModuleForward<SPEC>& module, RNG& rng){
    init_weights(device, module.sub_a, rng);
    init_weights(device, module.sub_b, rng);
}
```

All standard operations must be implemented: `malloc`, `free`, `init_weights`, `evaluate`, `forward`, `backward_full`, `backward`, `backward_input`, `zero_gradient`, `update`, `_reset_optimizer_state`, `reset_forward_state`, `copy`, `abs_diff`, `is_nan`, `output`.

## Include Guard Convention

Headers use a two-part guard:
```cpp
#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_<PATH>_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_<PATH>_H
// ...
#endif
```
Where `<PATH>` follows the directory structure in `SCREAMING_SNAKE_CASE` (e.g. `RL_TOOLS_NN_MODELS_PARALLEL_MODEL_H`).

## Adam Optimizer Usage

To use the Adam optimizer in tests/training:
1. `rlt::malloc(device, optimizer)` — allocates internal tensors (age, bias corrections, parameters)
2. `rlt::init(device, optimizer)` — sets hyperparameters (alpha, beta_1, beta_2, etc.) from defaults
3. `rlt::reset_optimizer_state(device, optimizer, model)` — zeros momentum tensors, sets age to 1
4. Training loop: `rlt::step(device, optimizer, model)` — calls `_step` (updates bias corrections, increments age) then `update`

Include order matters: `adam/instance/operations_generic.h` must be included before layer operations that call `update`/`_reset_optimizer_state` on Adam parameter instances. The top-level `adam/operations_generic.h` (with `reset_optimizer_state`, `step`) should be included after model operations so ADL finds the model's `_reset_optimizer_state`.

## Tensor Operations

- `get(device, tensor, indices...)` / `set(device, tensor, value, indices...)` — element access
- `get_flat(device, tensor, flat_index)` — flat index access (requires contiguous row-major layout)
- `data(tensor)` — raw pointer to tensor data
- `view_memory<SHAPE>(device, tensor)` — reshape view (shape rank must match stride rank)
- `copy(device, device, src, dst)` — copy between tensors (handles non-contiguous strides)
- When concatenating/splitting tensors from model outputs (which may be non-contiguous views), copy to contiguous intermediates first

## Code Style
- Braces on same line
- No unnecessary comments or docstrings — code should be self-explanatory
- Don't abbreviate variable names or other symbols. Follow the conventions from the rest of the RLtools codebase
- All functions are marked with `RL_TOOLS_FUNCTION_PLACEMENT` for CUDA compatibility
