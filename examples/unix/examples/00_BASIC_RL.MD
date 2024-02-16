### This examples shows the most basic example of using RLtools without any dependencies.

In the terminal, go to the RLtools repository you cloned via `git` before
```
cd path/to/rl_tools
```

Configure the RLtools using CMake
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS:BOOL=ON
```
Build the rl_algorithms_td3_full_training_standalone target which trains the Pendulum-v1 environment using TD3 on the CPU
```
cmake --build . --target rl_environments_pendulum_td3_standalone -j$(nproc)
```
Run the built binary
```
./src/rl/environments/pendulum/td3/cpu/rl_environments_pendulum_td3_standalone
```