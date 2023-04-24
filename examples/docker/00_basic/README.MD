```
cd backprop_tools/examples/docker/00_basic
./build.sh
docker run --rm -it --mount type=bind,source=$(cd ../../..; pwd),target=/backprop_tools,readonly backprop_tools:basic
cmake /backprop_tools/ -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_ENABLE_TESTS:BOOL=ON
cmake --build . --target test_rl_algorithms_td3_full_training_standalone -j$(nproc)
./tests/test_rl_algorithms_td3_full_training_standalone
```