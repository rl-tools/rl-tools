Change to the current directory
```
cd backprop_tools/examples/docker/02_mkl
```
Build the docker image (saved as `backprop_tools:mkl`)
```
./build.sh
```
Run a docker container based on the image just built (exposing port `6006` for tensorboard and mounting the source code from the host in readonly mode)
```
docker run --rm -it -p 6006:6006 --mount type=bind,source=$(cd ../../..; pwd),target=/backprop_tools,readonly --cap-add=sys_nice backprop_tools:mkl
```
Configure the BackpropTools using CMake
```
cmake /backprop_tools/ -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON
```
<!--
Alternatively: you can enable Tensorboard using the `BACKPROP_TOOLS_ENABLE_TENSORBOARD` flag (which might slow down the training)
```
cmake /backprop_tools/ -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON
```
-->
Build the test_rl_algorithms_td3_full_training_standalone target which trains the Pendulum-v1 environment using TD3 on the CPU
```
cmake --build . --target rl_environments_pendulum_td3_training_standalone -j$(nproc)
```
Optional: Start tensorboard in the background
```
tensorboard --logdir logs --bind_all 2> /dev/null > /dev/null &
```
Execute the built binary with a high priority
```
nice -n -20 ./src/rl/environments/pendulum/td3/cpu/rl_environments_pendulum_td3_training_standalone
```
You can observe the training using the browser on your host machine to access Tensorboard at `http://localhost:6006/
