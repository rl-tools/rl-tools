This release of RLtools contains binaries for the training and evaluation of PPO and TD3 deep reinforcement learning agents in the MujoCo Ant-v4 and the Gym Pendulum-v1 environments.
Pendulum-v1 (TD3):
    rl_environments_pendulum_td3_bare:
        - Training of the Pendulum-v1 without dependecies (at all, not even stdlibc++, hence the output (final sum of rewards/return) can not be through stdout and is provided through the return code)
    rl_environments_pendulum_td3_standalone:
        - Training of the Pendulum-v1 without dependecies
    rl_environments_pendulum_td3_blas
        - Accelerated Training of the Pendulum-v1 using the CPU (BLAS, e.g. Intel MKL or Apple Accelerate) but no other dependencies
    rl_environments_pendulum_td3_blas_benchmark:
        - Same as rl_environments_pendulum_td3_blas but without evaluation episodes
    rl_environments_pendulum_td3_blas_tensorboard
        - Same as rl_environments_pendulum_td3_blas but with TensorBoard logging

MuJoCo Ant-v4 (PPO):
    rl_environments_mujoco_ant_ppo_standalone:
        - Training of the MuJoCo Ant-v4 without dependencies
    rl_environments_mujoco_ant_ppo_blas
        - Accelerated Training of the MuJoCo Ant-v4 using the CPU (BLAS, e.g. Intel MKL or Apple Accelerate), with Tensorboard loggin and checkpointing. The run can be aborted when the agent reaches > 3000 return at which point the agent already learned a decent walking gait (and the checkpoints are saved regularly so that the gait can be inspected using rl_environments_mujoco_ant_evaluation_ppo). If the training is run for longer the agent should reach returns in excess of > 5000-6000 which represents a state of the art gait (kind of jumping).
    rl_environments_mujoco_ant_ppo_blas_benchmark
        - Same as rl_environments_mujoco_ant_ppo_blas but without logging and checkpointing
    rl_environments_mujoco_ant_ppo_cuda_standalone
        - Training of the MuJoCo Ant-v4 accelerated using the GPU with CUDA but no other dependencies
    rl_environments_mujoco_ant_ppo_cuda_benchmark
        - Same as rl_environments_mujoco_ant_ppo_cuda_standalone but without evaluation episodes (and BLAS, except for Windows)
    rl_environments_mujoco_ant_ppo_cuda_full
        - Same as rl_environments_mujoco_ant_ppo_cuda_standalone but with BLAS, Tensorboard logging and checkpointing. The run can be aborted when the agent reaches > 3000 return at which point the agent already learned a decent walking gait (and the checkpoints are saved regularly so that the gait can be inspected using rl_environments_mujoco_ant_evaluation_ppo). If the training is run for longer the agent should reach returns in excess of > 5000-6000 which represents a state of the art gait (kind of jumping).
    rl_environments_mujoco_ant_evaluation_ppo
        - This executable allows for deterministic evaluation of the checkpoints of agents trained with rl_environments_mujoco_ant_ppo_blas or rl_environments_mujoco_ant_ppo_cuda_full. This executable automatically loads the latest checkpoint of the latest run from the checkpoints folder in the current working directory.

MuJoCo Ant-v4 (TD3):
    Note: The hyperparameters for this implementation have not been tuned and the implementation has not been properly optimized. At this state it is much slower to train to a good performance than the PPO implementations but nevertheless we provide them for completeness
    rl_environments_mujoco_ant_training_td3_standalone:
        - Training of the MuJoCo Ant-v4 without dependencies
    rl_environments_mujoco_ant_training_td3_blas:
        - Training of the MuJoCo Ant-v4 accelerated using the CPU (BLAS, e.g. Intel MKL or Apple Accelerate) including Tensorboard logging and checkpointing. The run can be aborted when the agent reaches > 3000 return at which point the agent already learned a decent walking gait (and the checkpoints are saved regularly so that the gait can be inspected using rl_environments_mujoco_ant_evaluation_ppo). If the training is run for longer the agent should reach returns in excess of > 5000-6000 which represents a state of the art gait (kind of jumping).
    rl_environments_mujoco_ant_evaluation_td3:
        - This executable allows for deterministic evaluation of the checkpoints of agents trained with rl_environments_mujoco_ant_training_td3_blas. This executable automatically loads the latest checkpoint of the latest run from the checkpoints folder in the current working directory.

Racing Car (PPO):
    The racing car UI is implemented using the ui_server. Start the ui_server first and navigate to localhost:8000. Then start the rl_environments_car_interactive. A race-track should show up in your browser. Now you can edit the track, try driving the car yourself and finally train it using PPO

Notes:
- MuJoCo appears to require a CPU that supports the AVX extension when running on x86_64 CPUs (hence might not work in virtual machines)
- To run the training with CUDA support a CUDA installation is NOT required (just a NVIDIA driver of version >= 527.41).
- Note on checkpoints and logs: The checkpoints and logs will be placed into a "checkpoints" and "logs" folder inside the current working directory.
Tensorboard logs can be inspected (after installing tensorboard using e.g. "pip3 install tensorboard") by "python3 -m tensorboard.main --logdir runs" in the same working directory as the executables were run from.


