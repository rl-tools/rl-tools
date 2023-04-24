### This examples shows the most basic example of using BackpropTools without any dependencies.
You can verify the lack of dependencies in the [Dockerfile](Dockerfile) used for this example.
```
cd backprop_tools/examples/docker/00_basic
```
Build the docker image (tagged as `backprop_tools:basic`)
```
./build.sh
```
Run the image just built (mounting the source code from the host in readonly mode)
```
docker run --rm -it --mount type=bind,source=$(cd ../../..; pwd),target=/backprop_tools,readonly backprop_tools:basic
```
Configure the BackpropTools using CMake
```
cmake /backprop_tools/ -DCMAKE_BUILD_TYPE=Release
```
Build the test_rl_algorithms_td3_full_training_standalone target which trains the Pendulum-v1 environment using TD3 on the CPU
```
cmake --build . --target rl_environments_pendulum_td3_training_standalone -j$(nproc)
```
Run the built binary
```
./src/rl/environments/pendulum/td3/cpu/rl_environments_pendulum_td3_training_standalone
```