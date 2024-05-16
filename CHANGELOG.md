# Changes to latest release

- Environments should inherit from the `rl_tools::rl::environments::Environment` tag now. This allows default no-op implementations of e.g. `init` `malloc` etc. to avoid boilerplate
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