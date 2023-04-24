BACKPROP_TOOLS_INCLUDE_DIR=${BACKPROP_TOOLS_INCLUDE_DIR:-/home/jonas/phd/projects/rl_for_control/layer-in-c/include/}
echo BACKPROP_TOOLS_INCLUDE_DIR: $BACKPROP_TOOLS_INCLUDE_DIR
EXPORTED_FUNCTIONS="['_proxy_create_training_state', '_proxy_training_step', '_proxy_get_step', '_proxy_get_evaluation_count', '_proxy_get_evaluation_return', '_proxy_destroy_training_state', '_proxy_get_state_dim', '_proxy_get_state_value']"
em++ -DBACKPROP_TOOLS_STEP_LIMIT=10000 -DBACKPROP_TOOLS_BENCHMARK --std=c++17 -O3 -s WASM=1 -I$BACKPROP_TOOLS_INCLUDE_DIR -s EXPORTED_FUNCTIONS="$EXPORTED_FUNCTIONS" -s MODULARIZE=1 -s EXPORT_ES6=1 -s USE_ES6_IMPORT_META=0 -s ENVIRONMENT='web' --pre-js prejs.js -o build/wasm_interface_benchmark.js wasm_interface.cpp
em++ --std=c++17 -O3 -s WASM=1 -I$BACKPROP_TOOLS_INCLUDE_DIR -s EXPORTED_FUNCTIONS="$EXPORTED_FUNCTIONS" -s MODULARIZE=1 -s EXPORT_ES6=1 -s USE_ES6_IMPORT_META=0 -s ENVIRONMENT='web' --pre-js prejs.js -o build/wasm_interface.js wasm_interface.cpp
