DEBUG=${DEBUG:-0}

DEBUG_FLAGS=""
if [[ $DEBUG -eq 1 ]]; then
    DEBUG_FLAGS="-g -gsource-map -O0 -sSAFE_HEAP=1 -sASSERTIONS=1 -sNO_DISABLE_EXCEPTION_CATCHING"
else
    DEBUG_FLAGS="-O3 -ffast-math -msimd128"
fi

docker run -it --rm \
--mount type=bind,source=$(pwd)/wasm,target=/mnt,readonly \
--mount type=bind,source=$(pwd)/blob,target=/blob \
--mount type=bind,source=$(cd ../../rl-tools/ && pwd),target=/rl_tools,readonly \
--mount type=bind,source=$(cd external/json && pwd),target=/json,readonly \
-w /mnt \
emscripten/emsdk:4.0.17 \
emcc -std=c++17 -I /rl_tools/include -I /json/include -DEMSCRIPTEN -DRL_TOOLS_ENABLE_JSON \
$DEBUG_FLAGS \
-s WASM=1 --bind -s EXPORTED_RUNTIME_METHODS='["cwrap", "ccall"]' -s EXPORT_ES6=1 -s MODULARIZE=1 -o /blob/l2f-interface.js l2f.cpp
