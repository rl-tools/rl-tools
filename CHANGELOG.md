# Changes to latest release

- Environments should inherit from the `rl_tools::rl::environments::Environment` tag now. This allows default no-op implementations of e.g. `init` `malloc` etc. to avoid boilerplate
- The default Adam parameters are set from compile-time constants now
- All forward and evaluate operators take an `RNG` as input now
  - This is to allow for stochastic models (stochastic policies, VAEs etc.) that might e.g. be using the reparameterization trick to maintain differentiability