# ToDos


## Broad
- Remove all `#ifdef _MSC_VER`: find, analyze and restructure so that the implementation is generic for all compilers
- Change CMake configuration to "auto-detect" instead of manually enabling features
- Move away from Tensorboard (protobuf is just too much of a liability)
- Python bindings for RLtools: Since e.g. the observation and action dimensions need to be known at compile-time use the PyTorch C++ extensions to compile it on the fly
- Consider autodetecting available dependencies (MKL, ACCELERATE, Tensorboard etc.)
- Dissallow all C-style casts and move to C++-style casts
- Rethink/redesign the MatrixStatic vs MatrixDynamic system: can/should they be unified using a flag in the Specification? (goes in accordance with the move to n-dim tensors)
- Pre/Post-processing models (e.g. for observation normalization)
  - Removing e.g. the explicit (observation mean/std) parameters in the evaluation functions
- Enable warnings as errors
- Move run description from cpu device to a shared, top-level loop state
- Consider mutation testing
- Add generic checkpointing loop step
- Design and add full checkpointing step for the training state (to stop/resume training), including test cases verifying that the results are exactly identical
- Introduce flexible observation spaces and reward functions (tag dispatch)
  - To generalize privileged and normal observations
- Re-consider the `persist_code` format (currently assumes tha the endianness is the same between the trainin and inference platform)
- Create a sampling layer
  - For SAC, we would like to sample after the final identity layer, then squash with a tanh
  - Currently, the model only goes up to the identity and the sampling and tanh are done separately
    - This requires multiple places to correctly post-process the actions
      - Off-policy runner
      - SAC actor&critic training
      - Inference
    - In particular one needs to pay attention when exporting the model because the tanh-squashing is not exported
  - It would be best to implement a sampling layer and an element-wise layer (for the tanh squashing)
- Make the `EPISODE_STEP_LIMIT` an assumed property of the Environment type
- Move all C-style casts to C++ style casts (`static_cast` etc.)
- Add a mode tag to the `evaluate` and `forward` signature to signal the desired behavior (similar destincation as `.train()` and `.eval()` in PyTorch, but allowing for finer-grained and model specific control)
- Move the `observation_mean` and `observation_std` into the `nn_model`
- Iterator design that replaces the `get` and `set` functions for matrices and tensors. The current functions recalculate the index every time while an iterator-based design could just advance the pointer
## Atoms
- nn-mlp: msvc does not allow zero-sized arrays (hidden_layers are 0 if n layers = 2)
  - find a fix for all compilers (tests with n_layers = 2 are disabled for msvc for now)
- Debug why MKL fails when using Windows (MSVC) and CUDA in the `rl_environments_pendulum_sac_cuda`
- Port NaN init to the CUDA device
- Investigate MuJoCo build flags (used for the official release builds)
- Move `container.h` into `containers`
- Trying SonarCube
- Separate `gather_batch` from the off-policy runner (should be an operation on the replay buffer)
- Add Github action that test the compilation of the PX4 module [embedded_platforms/px4](embedded_platforms/px4)
- Check using separate learning rates for actor and critic (inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl/blob/8cbca61360ef98660f149e3d76762350ce613323/cleanrl/sac_continuous_action.py#L52))
- Check the [PPO implementation details regarding observation/reward normalization etc](https://github.com/vwxyzjn/cleanrl/blob/8cbca61360ef98660f149e3d76762350ce613323/cleanrl/ppo_continuous_action.py#L94)
  - Check [high epsilon Adam for PPO](https://github.com/vwxyzjn/cleanrl/blob/8cbca61360ef98660f149e3d76762350ce613323/cleanrl/ppo_continuous_action.py#L183)
- ~~Roll custom alloc_aligned (std::alloc_aligend does not work on Windows and macOS) cf. [this post](https://github.com/marian-nmt/marian-dev/issues/227#issuecomment-385912753)~~
- Allow generic BLAS (`-lblas`)
  - Renaming OpenBLAS to BLAS should be mostly sufficient
- Remove the experiment option scaffolding from `rl_environments_pendulum_ppo_training`
- Enable -Werror on Windows
- Implement more generic binary operations on `nn_model`s
  - Instead of hard-coding the `update_target_network` for mlp and sequential in both TD3 and SAC, implement a generic operation that applies a function between layers
- Change `nn_models::xxx::NeuralNetwork` to `nn_models::xxx::Model`
- Change ``nn::layer_capability`` to ``nn::capability``
- Make `nn_models::mlp_unconditional_stddev` a layer instead
- Make everything ready for `__fp16`
  - E.g. some RNG functions probably should cast to float for generation and then cast back
  - Make sure that `sgemm`/`dgemm` is not called
- Make the `Tests (minimal)` suite work again on the Windows Github actions (it works on my (windows) PC @tm)
- Investigate the "zero-sized array" error under MSVC for `rl_environments_pendulum_sac_arm_test` and `rl_environments_pendulum_sac_arm_test_benchmark`
- Properly export `nn::layers::dense::DefaultParameters` in the `persist_code` operations
- Remove the memcopy in `containers/tensor/persist.h`
- Check all examples with `-fsanitize=address`
- Make `rlt::get` consistent for Matrix and Tensor (matrix should take the device as the first argument as well)

# Features
- Look into implementing TQC, DroQ and CrossQ
