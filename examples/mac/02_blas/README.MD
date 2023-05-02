
In the terminal, go to the BackpropTools repository you cloned via `git` before
```
cd path/to/backprop_tools
```

Configure the BackpropTools using CMake
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE:BOOL=ON
```

#### Optional: Using Tensorboard
You can build the example with Tensorboard support by activating the `BACKPROP_TOOLS_ENABLE_TENSORBOARD` option
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON
```
Start tensorboard in a new terminal (make sure to navigate to the build directory first such that it can locate the logs directory)
```
tensorboard --logdir logs
```
You can observe the training using the browser to access Tensorboard at `http://localhost:6006/

#### Example: Pendulum-v1 using TD3
Build the rl_algorithms_td3_full_training_standalone target which trains the Pendulum-v1 environment using TD3 on the CPU
```
cmake --build . --target rl_environments_pendulum_td3_training_standalone -j$(nproc)
```
Execute the built binary (without Tensorboard logging)
```
./src/rl/environments/pendulum/td3/cpu/rl_environments_pendulum_td3_training_standalone
```
Alternatively, you can build the target with Tensorboard logging (after enabling it like specified before)
```
cmake --build . --target rl_environments_pendulum_td3_training_tensorboard -j$(nproc)
```
Execute the built binary (with Tensorboard logging)
```
./src/rl/environments/pendulum/td3/cpu/rl_environments_pendulum_td3_training_tensorboard
```

#### Example: Benchmarking MuJoCo Ant-v4 using PPO
Configure BackpropTools to enable MuJoCo
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON
```
Built the benchmark version of the PPO training target (which disables Tensorboard logging checkpointing etc.)
```
cmake --build . --target rl_environments_mujoco_ant_training_ppo_benchmark -j$(nproc)
```
Execute the built binary
```
./src/rl/environments/mujoco/ant/ppo/cpu/rl_environments_mujoco_ant_training_ppo_benchmark
```

#### Example: MuJoCo Ant-v4 using PPO (including checkpointing and Tensorboard logging)
Configure BackpropTools to enable MuJoCo
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON
```
Build the normal version of the PPO training target which includes checkpointing and Tensorboard logging
```
cmake --build . --target rl_environments_mujoco_ant_training_ppo_standalone -j$(nproc)
```
Execute the built binary
```
./src/rl/environments/mujoco/ant/ppo/cpu/rl_environments_mujoco_ant_training_ppo_standalone
```

#### Example: MuJoCo Ant-v4 using TD3
Configure BackpropTools to enable MuJoCo
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON
```
```
cmake --build . --target rl_environments_mujoco_ant_td3_training_standalone -j$(nproc)
```
Execute the built binary
```
./src/rl/environments/mujoco/ant/td3/rl_environments_mujoco_ant_td3_training_standalone
```

#### Example: MuJoCo Ant-v4 PPO policy evaluation in the UI
Configure BackpropTools to enable MuJoCo and the MuJoCo UI (using `-DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON`). Additionally the command line argument parsing library CLI11 (contained as a submodule in /external) is activated
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DBACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE:BOOL=ON -DBACKPROP_TOOLS_ENABLE_TENSORBOARD:BOOL=ON -DBACKPROP_TOOLS_ENABLE_HDF5:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DBACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ENABLE_UI:BOOL=ON -DBACKPROP_TOOLS_ENABLE_CLI11:BOOL=ON
```
Build the evaluation target
```
cmake --build . --target rl_environments_mujoco_ant_evaluation_ppo -j$(nproc)
```
Execute the built binary (it should use the most recent checkpoint of the previous standalone training run by default)
```
./src/rl/environments/mujoco/ant/rl_environments_mujoco_ant_evaluation_ppo
```

