# Target-frame training

All targets use Hyperdrone's target-frame task. Each family keeps compile-time settings in `config.h` and executable wiring in `training.cu` or `training.cpp`.

Target names below share the prefix `rl_environments_hyperdrone_tasks_target_frame_`.

| Directory | Target suffixes | Compute | Short profile |
| --- | --- | --- | --- |
| `on_policy` | `on_policy_cuda`, `on_policy_cuda_smoke` | CUDA + OptiX | 14 rollouts, including scene rotation and checkpoint export |
| `mixed` | `mixed_training_cuda`, `mixed_training_cuda_small` | CUDA + OptiX | 600 mixed PPO/imitation updates |
| `imitation` | `imitation`, `imitation_small` | CUDA + OptiX when available; CPU otherwise | 3 epochs |
| `imitation` | `state_estimation`, `state_estimation_small` | Same as imitation | 3 epochs predicting privileged state |
| `imitation` | `imitation_cpu` | CPU compute in a CUDA + OptiX build | Production profile |

Every executable accepts `<scene_directory or scene.glb> [seed]`. A directory supplies the numerically ordered scene corpus; a single GLB is replicated across the configured scene slots. Set `RL_TOOLS_EXTRACK_PATH` to the output parent; runs are written beneath its `experiments/` directory. Checkpoints identify the environment as `hyperdrone` and the run as the exact executable target. Studio supports the mixed policy's paired-image and 65-value sensor inputs.

Mixed training records the first scene set and every 20th thereafter to `steps/<end-step>/video.mp4`, also linked as `latest/video.mp4`, using FFmpeg with `libx264` on `PATH`. The 100 fps mosaic shows actual PPO training episodes across all active scenes, with each target image on the left and current camera image on the right; imitation instances reset every step and are omitted. Recording spans rollout updates and episode resets until the scene set or training ends (20.48 seconds per full scene set). Set `VIDEO_SAVE_INTERVAL_SCENE_SETS` in `mixed/config.h` to change the cadence or to `0` to disable recording; encoder failures disable recording and leave training running.

```bash
cmake --build build --target rl_environments_hyperdrone_tasks_target_frame_mixed_training_cuda_small -j5
CUDA_VISIBLE_DEVICES=0 ./build/src/rl/environments/hyperdrone/tasks/target_frame/mixed/rl_environments_hyperdrone_tasks_target_frame_mixed_training_cuda_small /path/to/scenes 0
```

`RL_TOOLS_HYPERDRONE_IMITATION_N_RANKS=1|2` selects the CUDA device count at CMake configuration time. Two ranks split the two active Worlds and the training batch equally; expose both devices with `CUDA_VISIBLE_DEVICES=0,1`. CPU variants use one rank. Production imitation stores its full rollout dataset, so its memory requirement is substantially larger than the small profile.

The shared boundaries follow their consumers: mixed training and Studio use the portable inertial-velocity observer in `l2f`; mixed training owns the target-frame position-hold initializer; on-policy training and its finite-difference tests share the target-local `on_policy/loss.h`. Hyperdrone's World, task operations, render backends, demos, and Python bindings provide the simulation API independently of these training profiles. Retired consumers impose no compatibility constraints: legacy PPO/mixed training and evaluation, imitation/state-estimation/yaw training, Dyn inference and stitching, unused Zoo configurations, OpenMV/sim2sim deployment, and Vast.ai/internal launchers.
