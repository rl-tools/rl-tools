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
| `yaw` | `yaw_cuda`, `yaw_cuda_smoke` | CUDA + OptiX + cuDNN + HDF5 + JSON | 3 epochs with validation and checkpoint export |

The on-policy, mixed, and imitation executables accept `<scene_directory or scene.glb> [seed]`. A directory supplies the numerically ordered scene corpus; a single GLB is replicated across the configured scene slots. Set `RL_TOOLS_EXTRACK_PATH` to the output parent; runs are written beneath its `experiments/` directory. Checkpoints identify the environment as `hyperdrone` and the run as the exact executable target. Studio supports the mixed policy's paired-image and 65-value sensor inputs.

`yaw_cuda <scene_directory> [seed]` reproduces the supervised yaw algorithm from `c0c7c31d95e9be0a1e5e926e3b419fe54e0fb164`: RAPTOR flies the rollouts, and one BF16 CNN receives current/target RGB at 80×50 with eight padded channels and predicts yaw cosine/sine. The production split uses 25 training and 5 held-out scenes; each epoch collects 6,400 rows and trains four passes over the first 6,144 rows in twelve batches of 512. The smoke target uses 2 training and 1 validation scene, accepts a single GLB replicated across those slots, and validates/checkpoints every epoch; replicated smoke scenes do not measure generalization. Both profiles use the same model, batch size, camera randomization, High shading, two motion-blur samples, and a 2×2 antialiasing grid. Current and target pixels use Hyperdrone’s float observations before brightness scaling; validation evaluates its 64 images without padding to the training batch size. Current rendering and free-space sampling prevent a bit-exact replay. HDF5 exports contain 512- and 2-example inference checkpoints, not optimizer-resume state.

Yaw refreshes its target bank every epoch; early resets reuse that epoch's position, scene yaw, camera and target setup while resampling dynamics. `metrics.jsonl` measures this reuse among collected episodes, validation frontier error, curriculum progress, timings and checkpoint reload error. `run.json` records configuration, scene content hashes, teacher identity and accepted differences; the ExTrack snapshot records the source revision and working diff. The smoke profile also requires the first epoch to reduce a fixed batch's loss. Yaw training and its CPU/CUDA tests share label and curriculum operations; its model, validation fixtures and HDF5 export stay target-local. Other Hyperdrone trainers, demos, Python callers and annotation tools constrain the shared observation and free-space APIs, which use their standard behavior without yaw-specific options.

Free-space sampling stops at 50 accepted positions instead of finishing the historical 64-candidate block; this changes the spawn distribution. Pixel encoding, spawn stopping and validation batch size are deliberate approximations, while scene/batch shuffling, epoch bank refresh and checkpoint example selection preserve the historical algorithm.

Yaw's dataset adapter restores ambient RGB `(0.5, 0.5, 0.5)`, always prepends the three historical directional fill lights to authored lights, and treats unlit materials as lit. This target-local interpretation preserves the normal dataset loaders and other rendering consumers. The modern metallic ambient term and black background remain differences from the historical renderer's metallic shading and red/white checker background.

On-policy targets default to the first `N_TOTAL_SCENES` entries of the header-pinned `procthor::Train200` catalog; pass `procthor_train_200 [seed]` to set the seed explicitly. Conta resolves its GLBs, and metra records the selected references.

Mixed training records the first scene set and every 20th thereafter to `steps/<end-step>/video.mp4`, also linked as `latest/video.mp4`, using FFmpeg with `libx264` on `PATH`. The 100 fps mosaic shows actual PPO training episodes across all active scenes, with each target image on the left and current camera image on the right; imitation instances reset every step and are omitted. Recording spans rollout updates and episode resets until the scene set or training ends (20.48 seconds per full scene set). Set `VIDEO_SAVE_INTERVAL_SCENE_SETS` in `mixed/config.h` to change the cadence or to `0` to disable recording; encoder failures disable recording and leave training running.

```bash
cmake --build build --target rl_environments_hyperdrone_tasks_target_frame_mixed_training_cuda_small -j5
CUDA_VISIBLE_DEVICES=0 ./build/src/rl/environments/hyperdrone/tasks/target_frame/mixed/rl_environments_hyperdrone_tasks_target_frame_mixed_training_cuda_small /path/to/scenes 0
```

`RL_TOOLS_HYPERDRONE_IMITATION_N_RANKS=1|2` selects the CUDA device count at CMake configuration time. Two ranks split the two active Worlds and the training batch equally; expose both devices with `CUDA_VISIBLE_DEVICES=0,1`. CPU variants use one rank. Production imitation stores its full rollout dataset, so its memory requirement is substantially larger than the small profile.

The shared boundaries follow their consumers: mixed training and Studio use the portable inertial-velocity observer in `l2f`; mixed training owns the target-frame position-hold initializer; on-policy training and its finite-difference tests share the target-local `on_policy/loss.h`. Hyperdrone's World, task operations, render backends, demos, and Python bindings provide the simulation API independently of these training profiles. Retired consumers impose no compatibility constraints: legacy PPO/mixed training and evaluation, imitation/state-estimation training, Dyn inference and stitching, unused Zoo configurations, OpenMV/sim2sim deployment, and Vast.ai/internal launchers.
