# Changes to latest release

- The default Adam parameters are set from compile-time constants now
- All forward and evaluate operators take an `RNG` as input now
  - This is to allow for stochastic models (stochastic policies, VAEs etc.) that might e.g. be using the reparameterization trick to maintain differentiability
- The `DEVICE` (and `device.logger` by extension) are initialized outside the core loop (they are instantiated outside anyways)
- Layer capabilities are handled more consistently
  - Layer capabilities are
    - Operations supported: Forward, Backward, Gradient
    - Parameter type: Plain, Gradient, Adam, ...
  - Instead of e.g. `rl_tools::nn::layers::dense::LayerForward<SPEC>` we use `rl_tools::nn::layers::dense::Layer<CAPABILITY, SPEC>` now where e.g. `using CAPABILITY = rl_tools::nn::layer_capability::Forward;` (which implies the `Plain` parameter type)
    - For Adam we can use: `using CAPABILITY = rl_tools::nn::layer_capability::Gradient<rl_tools::nn::parameters::Adam>;`


- The way RNGs are declared is more consistent with the rest of the API now. The `default_engine` function has been removed. Use  `using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;` followed by `RNG rng; 
- Optimizers need to be malloc'd now to make them seemlessly work on CPU and GPU. This is because e.g. the Adam `age` is part of the state that should be copied to the device.
- The SequentialBatch is not defined as a type inside the OffPolicyRunner anymore. The SequentialBatch describes how the batch is sampled so the `gather_batch_step` operation adjusts accordingly. Hence, the SequentialBatch should be configured outside the OffPolicyRunner to separate the concerns. For an example please check out [this](https://github.com/rl-tools/rl-tools/blob/3dea1bc877a8593dcd8349f6fdc4e362f025a0ca/include/rl_tools/rl/algorithms/sac/loop/core/state.h#L26). Please also configure the parameters `rl::components::off_policy_runner::SequentialBatchParametersDefault` to suit your problem. In the Markovian case with `SEQUENCE_LENGTH=1` the default parameters should work fine.