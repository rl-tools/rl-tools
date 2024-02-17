# ToDos


## Broad
- Remove all `#ifdef _MSC_VER`: find, analyze and restructure so that the implementation is generic for all compilers
- Change CMake configuration to "auto-detect" instead of manually enabling features
- Move away from Tensorboard (protobuf is just too much of a liability)
## Atoms
- nn-mlp: msvc does not allow zero-sized arrays (hidden_layers are 0 if n layers = 2)
  - find a fix for all compilers (tests with n_layers = 2 are disabled for msvc for now)
- Debug why MKL fails when using Windows (MSVC) and CUDA in the `rl_environments_pendulum_sac_cuda`
- Port NaN init to the CUDA device
