```
./build.sh
docker run --rm -it -p 6006:6006 --mount type=bind,source=$(cd ../../..; pwd),target=/backprop_tools,readonly --cap-add=sys_nice backprop_tools:mkl
cmake /backprop_tools/ -DCMAKE_BUILD_TYPE=Release -DLAYER_IN_C_ENABLE_TESTS:BOOL=ON -DLAYER_IN_C_BACKEND_ENABLE_MKL:BOOL=ON -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=ON
cmake --build . --target test_rl_algorithms_td3_full_training_standalone -j$(nproc)
nice -n -20 ./tests/test_rl_algorithms_td3_full_training_standalone
tensorboard --logdir logs --bind_all
```
