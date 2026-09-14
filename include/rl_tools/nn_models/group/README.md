# Model groups and optimizer selection

`nn_models::Group<Tag, Module>` attaches a user-defined type tag to a module's parameters. It adds no runtime storage or layer index. Groups are independent of the existing parameter roles (`Normal`, `Input`, `Output`) and categories (`Weights`, `Biases`).

```cpp
struct Encoder{};
struct Head{};
using CHAIN = nn_models::sequential::Module<nn_models::Group<Encoder, ENCODER>, nn_models::Group<Head, HEAD>>;
using MODEL = nn_models::sequential::Build<CAPABILITY, CHAIN, INPUT_SHAPE>;
using FREEZE_ENCODER = Mode<nn::optimizers::mode::Freeze<mode::Default<>, nn::group::Only<Encoder>>>;

zero_gradient(device, model);
forward(device, model, input, buffer, rng);
backward(device, model, input, d_output, buffer);
step(device, optimizer, model, FREEZE_ENCODER{});
```

`Freeze` selects the parameters to leave unchanged; all other parameters train, including ungrouped parameters and newly added groups. The mode is consumed only by `step`. All layers compute their usual gradients, including input derivatives through frozen modules. The optimizer skips the entire parameter update for excluded parameters: values, master values, momentum/moments, and weight decay remain unchanged. Layer operations and their signatures need no knowledge of groups. Adam, SGD, and Lamb use their existing update implementations through an optimizer adapter; default calls keep the existing dispatch, including fused CUDA updates.

To unfreeze, pass another selection or call `step(device, optimizer, model)` to update all parameters. The model type, allocation, and parameter storage stay the same. Actor and critic calls can use different selections without introducing domains.

## Selection

Any user-defined type can be a tag. Ungrouped parameters belong to `nn::group::Default`. A named group replaces that implicit default; an explicit `Group<nn::group::Default, Module>` leaves inherited membership unchanged. Nested named groups retain membership in each ancestor; `Only<Parent>` therefore selects the entire subtree. Repeated uses of the same tag are selected together.

Selectors expose `template<typename TARGET> static constexpr bool MATCH` and operate on descriptors supplied by each effect. Built-in selectors are `All`, `Only<Tags...>`, `ParameterGroup<Role>`, `Category<Category>`, `Any<Selectors...>`, `Every<Selectors...>`, and `Not<Selector>`. For optimizer updates, `Every<Only<Head>, Category<nn::parameters::categories::Weights>>` selects head weights while preserving its biases. `Only` reads `TARGET::MODEL_GROUPS`, `ParameterGroup` reads `TARGET::GROUP_TAG`, and `Category` reads `TARGET::CATEGORY_TAG`; a descriptor only needs the fields used by its selectors.

## Descriptors belong to effects

The group system owns membership and reusable matchers. The shared optimizer header `nn/optimizers/update.h` owns `UpdateTarget<PARAMETER_SPEC>`, `mode::Update<BASE, SELECTOR>`, `mode::Freeze<BASE, SELECTOR>`, the mode lookup, and the optimizer adapter. `UpdateTarget` supplies model membership, parameter role, and parameter category; its `PARAMETER_SPEC` alias lets a custom selector inspect additional parameter properties such as shape or precision.

`nn::optimizers::mode::Freeze` carries the selector for frozen parameters. It is equivalent to `Update<BASE, Not<SELECTOR>>`; both selectors match the same `UpdateTarget`. `Freeze<>` defaults to `All` and freezes every parameter, while a selector matching nothing freezes nothing. `nn::optimizers::mode::Update` remains available for callers that prefer an allowlist.

Both descriptors control the same optimizer update policy: the outermost `Update` or `Freeze` in the mode chain wins. Combine multiple frozen sets with `Any` inside one `Freeze` descriptor. An absent descriptor, a plain default mode, or a mode for an unrelated effect retains the default of updating all parameters.

Future effects can define their own target descriptors and mode tags in their owning components, then use these same selectors. For example, a descriptor with only `MODEL_GROUPS` can use `Only`, `Any`, `Every`, and `Not` without carrying parameter or optimizer metadata. Each effect resolves only its own mode tag, so several effects can coexist in one mode chain with different selections. Group definitions, layers, and persistence need no changes for that extension.

`mlp_unconditional_stddev` also accepts `CONFIG::LOG_STD_GROUP` to label its exploration parameter independently of the mean network. This is optional model wiring; its operations remain unchanged.

## Optimizer state and accumulation

Consumers own phase boundaries. Call `zero_gradient` before starting a fresh accumulation window, including after changing the selection. Gradients for excluded parameters are still computed and accumulated; `step` neither clears nor consumes them. Do not reuse those accumulated gradients accidentally after unfreezing. Global gradient transformations such as norm clipping also see excluded gradients unless the consumer deliberately restricts them.

Adam and Lamb retain their existing optimizer-wide age: every `step` advances it, even if the selection matches no parameters. Excluded parameter moments stay paused while that age advances. Unfreezing resumes with those retained moments and the current bias correction; this is not equivalent to an independently paused Adam optimizer. Consumers that require independent clocks can use separate optimizer instances for disjoint groups and step only the active instance. Moments live in the model's parameter instances, so overlapping selections still share moments. Resetting an optimizer against the whole model resets all of its parameter state; initialization/reset and checkpoint restoration must respect that ownership.

Selection freezes parameter updates only. Forward-pass state, such as batch-normalization running statistics or recurrent state, follows the existing forward mode and lifecycle.

## Copying and persistence

Grouped and ungrouped versions with compatible layers use the existing `copy` operations. Group tags change compile-time metadata, not tensor paths or shapes in tar/HDF5 checkpoints. Loading either representation into the other model uses the destination model's group definitions.

C++ code export preserves group membership. Custom tags need a `nn::group::CodeName<Tag>` specialization from `group/persist_code.h`, providing `VALUE` (the qualified C++ type name) and `DEFINITION` (generated declarations or an include directive). Empty tag types can be forward-declared in exported code:

```cpp
namespace rl_tools::nn::group{
    template<> struct CodeName<::Encoder>{
        static constexpr const char* VALUE = "::Encoder";
        static constexpr const char* DEFINITION = "struct Encoder;";
    };
}
```

## Consumers and design constraints

| Consumers/use cases considered | Decision they constrain |
| --- | --- |
| Mixed visual actor: fixed prefix, controller, mean head, and readout bias; learned visual layers and exploration scale | Freeze fixed parameters except the nested exploration parameter, plus readout biases; use membership and category selectors without enumerating all trainable groups. |
| Separate actor/critic phases and user models | Arbitrary user tags and call-local modes; no domains. |
| Nested sequential modules, parallel branches, MLP heads | Carry metadata through capabilities; preserve flattened layer indices and existing operation traversal. |
| Dense, convolutional, recurrent, normalization, and future layers | Keep forward/backward/update layer APIs unchanged; all gradients remain available. |
| Adam/SGD on CPU and CUDA, Lamb on CPU, master parameters, momentum, and weight decay | Share optimizer-local `UpdateTarget`, `mode::Update`, and `mode::Freeze`; skip the complete parameter-level update and retain ordinary dispatch for default calls. |
| Future effects reusing groups and independent selections in one mode chain | Keep matchers effect-neutral; each effect owns its descriptor fields, mode tag, lookup, and default behavior. |
| Freeze/unfreeze schedules, default calls, and gradient accumulation | Keep allocation stable and default calls unrestricted; consumers manage gradients, optimizer clocks, resets, and moment ownership. |
| Models gaining additional groups and composed optimizer modes | Freeze only explicit matches; retain the existing outermost-policy precedence for both `Update` and `Freeze`. |
| Inference, compatible model copies, tar/HDF5, and generated C++ | Preserve tensor layout and checkpoint paths; export type tags explicitly when generating C++. |

Backward compute pruning is a separate optimization. Optimizer selection intentionally pays the cost of computing all gradients.
