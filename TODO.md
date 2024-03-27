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
- Remove the memcopy in `containers/tensor/persist.h`