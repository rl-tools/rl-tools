```
./build.sh
docker run --rm -it -p 8000:8000 --mount type=bind,source=$(cd ../../..; pwd),target=/backprop_tools,readonly backprop_tools:wasm
cp -r /backprop_tools/tests/src/rl/algorithms/td3/wasm/static .
BACKPROP_TOOLS_INCLUDE_DIR=/backprop_tools/include BACKPROP_TOOLS_BUILD_DIR=/build/static/build /backprop_tools/tests/src/rl/algorithms/td3/wasm/build.sh
python3 -m http.server --directory static
```
