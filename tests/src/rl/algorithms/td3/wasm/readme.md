```
source ~/git/emsdk/emsdk_env.sh
```

```
 em++ --std=c++17 -O3 -s WASM=1 -I/home/jonas/phd/projects/rl_for_control/layer-in-c/include/ -s EXPORTED_FUNCTIONS="['_proxy_create_training_state', '_proxy_training_step', '_proxy_get_step', '_proxy_get_evaluation_count', '_proxy_get_evaluation_return', '_proxy_destroy_training_state']" -o wasm_interface.js wasm_interface.cpp
```
```
python3 -m http.server
```
