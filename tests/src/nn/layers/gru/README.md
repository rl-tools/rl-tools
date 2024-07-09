Checking bare-bonedness using the wasm32 target:

```
mkdir build
cd build
```
```
/opt/homebrew/opt/llvm/bin/clang++ -Ofast --target=wasm32 -I../include/  -std=c++17 -nostdlib -Wl,--no-entry -Wl,--export=main -mbulk-memory ../tests/src/nn/layers/gru/gru_compile.cpp -o gru_compile_test
```

Checking the compile-time of rlt::Tensor


```
 hyperfine 'clang++ -Ofast -I../include/  -std=c++17 ../tests/src/nn/layers/gru/gru_compile.cpp -o gru_compile_test'
```