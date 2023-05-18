This release of BackpropTools contains binaries for the training and evaluation of PPO and TD3 deep reinforcement learning agents in the MujoCo Ant-v4 and the Gym Pendulum-v1 environments.
Pendulum-v1 (TD3):
    rl_environments_pendulum_td3_training_standalone:
        - Training of the Pendulum-v1 without dependecies
    rl_environments_pendulum_td3_training_blas
        - Accelerated Training of the Pendulum-v1 using the CPU (BLAS, e.g. Intel MKL or Apple Accelerate) but no other dependencies
    rl_environments_pendulum_td3_training_blas_benchmark:
        - Same as rl_environments_pendulum_td3_training_blas but without evaluation episodes
    rl_environments_pendulum_td3_training_blas_tensorboard
        - Same as rl_environments_pendulum_td3_training_blas but with TensorBoard logging

MuJoCo Ant-v4 (PPO):
    rl_environments_mujoco_ant_training_ppo_standalone:
        - Training of the MuJoCo Ant-v4 without dependecies
    rl_environments_mujoco_ant_training_ppo_blas
        - Accelerated Training of the MuJoCo Ant-v4 using the CPU (BLAS, e.g. Intel MKL or Apple Accelerate), with Tensorboard loggin and checkpointing. The run can be aborted when the agent reaches > 3000 return at which point the agent already learned a decent walking gait (and the checkpoints are saved regularly so that the gait can be inspected using rl_environments_mujoco_ant_evaluation_ppo). If the training is run for longer the agent should reach returns in excess of > 5000-6000 which represents a state of the art gait (kind of jumping).
    rl_environments_mujoco_ant_training_ppo_cuda_standalone
        - Training of the MuJoCo Ant-v4 accelerated using the GPU with CUDA but no other dependencies
    rl_environments_mujoco_ant_training_ppo_cuda_benchmark
        - Same as rl_environments_mujoco_ant_training_ppo_cuda_standalone but without evaluation episodes (and BLAS, except for Windows)
    rl_environments_mujoco_ant_training_ppo_cuda_full
        - Same as rl_environments_mujoco_ant_training_ppo_cuda_standalone but with BLAS, Tensorboard logging and checkpointing. The run can be aborted when the agent reaches > 3000 return at which point the agent already learned a decent walking gait (and the checkpoints are saved regularly so that the gait can be inspected using rl_environments_mujoco_ant_evaluation_ppo). If the training is run for longer the agent should reach returns in excess of > 5000-6000 which represents a state of the art gait (kind of jumping).
    rl_environments_mujoco_ant_evaluation_ppo
        - This executable allows for deterministic evaluation of the checkpoints of agents trained with rl_environments_mujoco_ant_training_ppo_blas or rl_environments_mujoco_ant_training_ppo_cuda_full. This executable automatically loads the latest checkpoint of the latest run from the checkpoints folder in the current working directory.

MuJoCo Ant-v4 (TD3):
    Note: The hyperparameters for this implementation have not been tuned and the implementation has not been properly optimized. At this state it is much slower to train to a good performance than the PPO implementations but nevertheless we provide them for completeness
    rl_environments_mujoco_ant_training_td3_standalone:
        - Training of the MuJoCo Ant-v4 without dependencies
    rl_environments_mujoco_ant_training_td3_blas:
        - Training of the MuJoCo Ant-v4 accelerated using the CPU (BLAS, e.g. Intel MKL or Apple Accelerate) including Tensorboard logging and checkpointing. The run can be aborted when the agent reaches > 3000 return at which point the agent already learned a decent walking gait (and the checkpoints are saved regularly so that the gait can be inspected using rl_environments_mujoco_ant_evaluation_ppo). If the training is run for longer the agent should reach returns in excess of > 5000-6000 which represents a state of the art gait (kind of jumping).
    rl_environments_mujoco_ant_evaluation_td3:
        - This executable allows for deterministic evaluation of the checkpoints of agents trained with rl_environments_mujoco_ant_training_td3_blas. This executable automatically loads the latest checkpoint of the latest run from the checkpoints folder in the current working directory.

Notes:
- MuJoCo appears to require a CPU that supports the AVX extension (hence might not work in virtual machines)
- To run the training with CUDA support a CUDA installation is NOT required (just a NVIDIA driver of version > 460).
- Note on checkpoints and logs: The checkpoints and logs will be placed into a "checkpoints" and "logs" folder inside the current working directory.
Tensorboard logs can be inspected (after installing tensorboard using e.g. "pip3 install tensorboard") by "python3 -m tensorboard.main --logdir logs" in the same workind directory as the executables were run from.


