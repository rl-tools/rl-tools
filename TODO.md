# ToDos


## General
- Remove all `#ifdef _MSC_VER`: find, analyze and restructure so that the implementation is generic for all compilers
## Detailed 
- nn-mlp: msvc does not allow zero-sized arrays (hidden_layers are 0 if n layers = 2)
  - find a fix for all compilers (tests with n_layers = 2 are disabled for msvc for now)