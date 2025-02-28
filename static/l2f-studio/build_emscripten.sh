docker run -it --rm \
--mount type=bind,source=$(pwd),target=/mnt \
--mount type=bind,source=$(cd ../.. && pwd),target=/rl_tools,readonly \
--mount type=bind,source=$(cd ../../external/json && pwd),target=/json,readonly \
-w /mnt \
emscripten/emsdk:4.0.4 \
emcc -std=c++17 -O3 -I /rl_tools/include -I /json/include -DEMSCRIPTEN -DRL_TOOLS_ENABLE_JSON -s WASM=1 --bind -s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall"]' -s EXPORT_ES6=1 -s MODULARIZE=1 -o l2f-interface.js l2f.cpp
