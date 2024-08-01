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


Compile time with `-Ofast`: 800ms (450ms without) on Apple M3
```
g++ gru_training.cpp -I /Users/jonas/rl_tools/include  -std=c++17 -I /opt/homebrew/opt/cjson/include -L /opt/homebrew/opt/cjson/lib -lcjson -lz -Ofast && ./a.out
```