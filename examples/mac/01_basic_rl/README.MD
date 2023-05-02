### This examples shows the most basic example of using BackpropTools without any dependencies.

In the terminal, go to the BackpropTools repository you cloned via `git` before
```
cd path/to/backprop_tools
```

Configure the BackpropTools using CMake
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```
Build the rl_algorithms_td3_full_training_standalone target which trains the Pendulum-v1 environment using TD3 on the CPU
```
cmake --build . --target rl_environments_pendulum_td3_training_standalone -j$(nproc)
```
Run the built binary
```
./src/rl/environments/pendulum/td3/cpu/rl_environments_pendulum_td3_training_standalone
```