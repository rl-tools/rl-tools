### Go to the current directory
```
cd examples/docker/03_cuda
```
### Building the environment
Build the docker container (based on an NVIDIA cuda ubuntu image) installing Intel MKL as well as a recent version of CMake. The image is tagged as `rl_tools:cuda`
```
docker build -f Dockerfile -t rl_tools:cuda . --no-cache
```
### Running the container and configuring RLtools
Run the container in interactive mode, mounting the source code from the host in readonly mode. We expose port 6006 for tensorboard and allow the sys_nice capability to be able to run the tests with a high priority
```
docker run --rm -it -p 6006:6006 --mount type=bind,source=$(cd ../../..; pwd),target=/rl_tools,readonly --cap-add=sys_nice --mount type=bind,source=$(pwd)/../checkpoints,target=/build/checkpoints --runtime=nvidia --gpus all rl_tools:cuda
```
Configure the RLtools using CMake
```
CC=clang CXX=clang++ cmake /rl_tools/ -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_ENABLE_TARGETS:BOOL=ON -DRL_TOOLS_BACKEND_ENABLE_MKL:BOOL=ON -DRL_TOOLS_BACKEND_ENABLE_CUDA:BOOL=ON -DRL_TOOLS_RL_ENVIRONMENTS_ENABLE_MUJOCO:BOOL=ON -DRL_TOOLS_ENABLE_HDF5:BOOL=ON -DRL_TOOLS_ENABLE_CLI11:BOOL=ON -DRL_TOOLS_ENABLE_TENSORBOARD:BOOL=ON
```
And build the targets
```
cmake --build . -j$(nproc)
```
Execute PPO based Ant-v4 training in the benchmark configuration (without checkpoints and tensorboard logging) with a high priority
```
nice -n -20 /build/src/rl/environments/mujoco/ant/ppo/cuda/rl_environments_mujoco_ant_ppo_cuda_benchmark
```
If you want to save checkpoints (e.g. to replay the behavior in the MuJoCo UI as described in [03_mujoco_ui](../03_mujoco_ui/README.MD)) and Tensorboard logs run the following
Optional: Start tensorboard in the background
```
tensorboard --logdir runs --bind_all 2> /dev/null > /dev/null &
```
Run the executable with checkpointing and Tensorboard logging enabled:
```
nice -n -20 /build/src/rl/environments/mujoco/ant/ppo/cuda/rl_environments_mujoco_ant_ppo_cuda_full
```
If you ran Tensorboard in the background before, you can observe the training using the browser on your host machine to access Tensorboard at `http://localhost:6006/`.


Expected output (for `rl_environments_mujoco_ant_ppo_cuda_full`):

```
Saving actor checkpoints to: checkpoints/ppo_ant
Logging to runs/2024-02-16T21_13_51Z/logs.tfevents
Observation means: 
    0.637259     0.905689    -0.001548     0.007913    -0.074319     0.112792     0.868252     0.148552    -0.748996     0.166856    -0.838683     0.136560     0.795459    -0.000869    -0.059214    -0.032415    -0.005872     0.022958    -0.097306     0.052908     0.291650     0.027677    -0.248768     0.034197    -0.233549     0.056375     0.241284 
Observation std: 
    0.116831     0.195826     0.136421     0.129967     0.317402     0.381337     0.314262     0.383310     0.281310     0.376437     0.306142     0.380004     0.301578     0.726085     0.726442     0.879740     1.555637     1.475405     1.558524     3.970453     4.197302     3.913126     3.597429     3.944718     3.950355     3.771361     4.154008 
Evaluation return mean: 981.885559 (std: 19.772139)
PPO step:          0 environment step:       8192 elapsed:       3.27s (lifetime:    2504.60 steps/s, current:   13633.69 steps/s)
PPO step:          1 environment step:      12288 elapsed:       3.47s (lifetime:    3544.70 steps/s, current:   21554.46 steps/s)
PPO step:          2 environment step:      16384 elapsed:       3.64s (lifetime:    4498.13 steps/s, current:   24055.06 steps/s)
PPO step:          3 environment step:      20480 elapsed:       3.82s (lifetime:    5355.01 steps/s, current:   23204.80 steps/s)
PPO step:          4 environment step:      24576 elapsed:       4.02s (lifetime:    6117.96 steps/s, current:   21952.43 steps/s)
PPO step:          5 environment step:      28672 elapsed:       4.22s (lifetime:    6800.98 steps/s, current:   21205.90 steps/s)
PPO step:          6 environment step:      32768 elapsed:       4.40s (lifetime:    7453.45 steps/s, current:   23415.64 steps/s)
PPO step:          7 environment step:      36864 elapsed:       4.59s (lifetime:    8035.38 steps/s, current:   22060.89 steps/s)
.
.
.
PPO step:       2485 environment step:   10186752 elapsed:     783.08s (lifetime:   13008.54 steps/s, current:   19685.63 steps/s)
PPO step:       2486 environment step:   10190848 elapsed:     783.29s (lifetime:   13010.38 steps/s, current:   20653.62 steps/s)
PPO step:       2487 environment step:   10194944 elapsed:     783.49s (lifetime:   13012.18 steps/s, current:   20414.74 steps/s)
PPO step:       2488 environment step:   10199040 elapsed:     783.70s (lifetime:   13013.98 steps/s, current:   20412.02 steps/s)
PPO step:       2489 environment step:   10203136 elapsed:     783.90s (lifetime:   13015.83 steps/s, current:   20735.18 steps/s)
Evaluation return mean: 6526.52 (std: 669.99)
PPO step:       2490 environment step:   10207232 elapsed:     786.63s (lifetime:   12975.92 steps/s, current:   15299.73 steps/s)
PPO step:       2491 environment step:   10211328 elapsed:     786.83s (lifetime:   12977.75 steps/s, current:   20612.49 steps/s)
PPO step:       2492 environment step:   10215424 elapsed:     787.04s (lifetime:   12979.55 steps/s, current:   20395.32 steps/s)
PPO step:       2493 environment step:   10219520 elapsed:     787.25s (lifetime:   12981.37 steps/s, current:   20547.84 steps/s)
PPO step:       2494 environment step:   10223616 elapsed:     787.45s (lifetime:   12983.23 steps/s, current:   20842.82 steps/s)
PPO step:       2495 environment step:   10227712 elapsed:     787.65s (lifetime:   12985.07 steps/s, current:   20639.70 steps/s)
PPO step:       2496 environment step:   10231808 elapsed:     787.86s (lifetime:   12986.91 steps/s, current:   20700.27 steps/s)
PPO step:       2497 environment step:   10235904 elapsed:     788.07s (lifetime:   12988.62 steps/s, current:   19942.56 steps/s)
PPO step:       2498 environment step:   10240000 elapsed:     788.27s (lifetime:   12990.42 steps/s, current:   20438.23 steps/s)
PPO step:       2499 environment step:   10244096 elapsed:     788.48s (lifetime:   12992.28 steps/s, current:   20834.11 steps/s)
```


As we can observe, the policy learned a very good walking gait with a return greater than 6000. 10M steps is a bit of an overkill because it usually already reaches a very good gait after 1M - 2M steps.
