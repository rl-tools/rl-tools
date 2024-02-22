# ToDos


## Broad
- Remove all `#ifdef _MSC_VER`: find, analyze and restructure so that the implementation is generic for all compilers
- Change CMake configuration to "auto-detect" instead of manually enabling features
- Move away from Tensorboard (protobuf is just too much of a liability)
- Python bindings for RLtools: Since e.g. the observation and action dimensions need to be known at compile-time use the PyTorch C++ extensions to compile it on the fly
- Consider autodetecting available dependencies (MKL, ACCELERATE, Tensorboard etc.)
## Atoms
- nn-mlp: msvc does not allow zero-sized arrays (hidden_layers are 0 if n layers = 2)
  - find a fix for all compilers (tests with n_layers = 2 are disabled for msvc for now)
- Debug why MKL fails when using Windows (MSVC) and CUDA in the `rl_environments_pendulum_sac_cuda`
- Port NaN init to the CUDA device
- Investigate MuJoCo build flags (used for the official release builds)
