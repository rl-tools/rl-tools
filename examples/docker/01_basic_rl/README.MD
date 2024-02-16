### This examples shows the most basic example of using RLtools without any dependencies.
You can verify the lack of dependencies in the [Dockerfile](Dockerfile) used for this example.
```
cd examples/docker/01_basic_rl
```
Build the docker image (tagged as `rl_tools:basic_rl`)
```
docker build -f Dockerfile -t rl_tools:basic_rl . --no-cache
```
Run the image just built (mounting the source code from the host in readonly mode)
```
docker run --rm -it --mount type=bind,source=$(cd ../../..; pwd),target=/rl_tools,readonly rl_tools:basic_rl
```
Configure the RLtools using CMake
```
cmake /rl_tools/ -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS:BOOL=ON
```
Build the rl_algorithms_td3_full_training_standalone target which trains the Pendulum-v1 environment using TD3 on the CPU
```
cmake --build . --target rl_environments_pendulum_td3_standalone -j$(nproc)
```
Run the built binary
```
./src/rl/environments/pendulum/td3/cpu/rl_environments_pendulum_td3_standalone
```

Expected output:
```
Mean return: -1269.58
step_i: 1000 3.93819s
Mean return: -1674.61
step_i: 2000 8.39101s
Mean return: -1411.22
step_i: 3000 12.849s
Mean return: -1329.09
step_i: 4000 17.3239s
Mean return: -1031.09
step_i: 5000 21.789s
Mean return: -452.913
step_i: 6000 26.2582s
Mean return: -318.911
step_i: 7000 30.719s
Mean return: -167.269
step_i: 8000 35.1831s
Mean return: -94.4109
step_i: 9000 39.6544s
Mean return: -211.826
total time: 44.1208s
```

We can see that the policy learned to swing up the pendulum by the mean return increasing to approximately -200.
