### Assets (conta)

Scenes referenced as `conta:<sha1>` are resolved through the conta client (`include/conta/conta.h`):
- By default missing blobs are downloaded from `https://huggingface.co/datasets/rl-tools/conta/resolve/main/data/<sha1>` into `${XDG_CACHE_HOME:-$HOME/.cache}/rl_tools/conta` (shared with the render server's Node cache).
- `CONTA_ROOT` (optional): path to a read-only, pre-populated store (e.g. a checkout of the `rl-tools/conta` dataset, or `/data/conta` on infra). Blobs are expected at `$CONTA_ROOT/data/<sha1>`; downloading is disabled and a missing blob is a hard error.
- `CONTA_CACHE` (optional): relocate the writable download cache.
- `CONTA_URL` (optional): alternative download base URL (mirror, `file://` store).

The `conta` CLI target (`src/conta/cli.cpp`) prefetches blobs: `conta <sha1> [more ...]` prints one local path per line.

### On-policy training

```
rl_environments_l2f_visual_training_cuda <scene_directory | scene.glb> [seed]
```

`training_cuda.cu` uses each rollout sample once. It collects 320 steps from 32 environments (16 per scene), computes GAE, and accumulates actor and critic gradients over ten 1,024-sample minibatches. Each rollout produces one averaged Adam update over 10,240 fresh transitions. Actor and critic parameters remain fixed during collection and backward passes; each rollout's image buffers are overwritten after its backward pass, retaining the preceding frames needed for frame stacking. Episodes continue across updates. The two active scenes rotate after seven rollouts (2,240 steps per environment), resetting remaining episodes. Advantage normalization remains per minibatch.

The actor uses the on-policy log-probability objective with entropy regularization. There is one actor/critic update per 10,240 fresh samples, with no sample reuse. This is 25.6 times as many optimizer updates per sample as the previous 262,144-sample accumulated configuration. The `training/optimizer_updates`, `training/accumulated_samples`, and `training/sample_uses` TensorBoard metrics expose the schedule; the existing `ppo/*` diagnostics retain their names. Extrack runs carry `algorithm=on-policy`, and metra metrics use `l2f_visual_training_on_policy/*`.

### Mixed imitation and reinforcement learning

`rl_environments_l2f_visual_training_mixed_cuda` is the full-size mixed target. It compiles `training_cuda.cu` with `RL_TOOLS_L2F_VISUAL_TRAINING_MIXED`; the ordinary `rl_environments_l2f_visual_training_cuda` target always uses pure RL. Both targets coexist in the same build without a CMake mode toggle:

```bash
cmake -B build
cmake --build build --target rl_environments_l2f_visual_training_mixed_cuda -j5
./build/src/rl/environments/l2f_visual/rl_environments_l2f_visual_training_mixed_cuda /data/procthor-train-200-glb 0
```

The RAPTOR teacher uses the same converted `src/nn_models/port_checkpoint/raptor/policy.h` as `imitation_cuda.cu`. With the original RAPTOR assets already present, generate that header by building and running `nn_models_port_checkpoint_raptor`.

Edit `MIXED_PARAMETERS` near the top of `training_cuda.cu` to configure the run:

| Constant | Default | Meaning |
| --- | --- | --- |
| `IMITATION_ENVIRONMENTS_PER_SCENE` | 8 of 16 | IL share in each scene and every minibatch; must leave at least one IL and one RL environment per scene |
| `TEACHER_FORCING_UPDATES` | 10 | First this many optimizer updates use teacher actions in IL environments |
| `IMITATION_WEIGHT` | 1 | Weight of the half-MSE teacher loss relative to the RL objective |
| `BALANCING_MODE` | `rlt::Mode<mixed_imitation::FixedWeight<>>` | Change to `rlt::Mode<mixed_imitation::FixedNormRatio<>>` to balance each minibatch |
| `OUTPUT_GRADIENT_NORM_RATIO` | 1 | Desired weighted IL / RL action-mean gradient L2 norm ratio in `FixedNormRatio` mode |
| `MAX_IMITATION_WEIGHT` | 1000 | Upper bound on the adaptive IL multiplier |
| `NORM_EPSILON` | 1e-12 | Threshold for treating an output-gradient norm as zero |

The first IL slots of each scene remain IL slots throughout training. Each 1,024-row minibatch contains complete environment steps, giving exactly 512 IL and 512 RL samples with the defaults, even after batch-order shuffling. Both subsets feed **one actor forward/backward pass**, and every accumulated Adam update includes both. Each loss is averaged over its own subset, then scaled by the reciprocal of the number of accumulated minibatches. Thus the share controls sampling allocation independently of the loss weight. The RL policy objective and entropy term retain their existing form; IL uses `0.5 * mean((action_mean - teacher_action)^2)`, matching the imitation target's action loss.

RL environments always execute sampled student actions. IL environments execute only teacher actions during warmup, then execute student mean actions, as in the imitation target. RAPTOR labels the current, pre-action state in both phases; its recurrent state resets with each environment and persists across rollout boundaries. The warmup switch happens at an optimizer-update boundary, including for ongoing episodes. IL rows never contribute to RL advantage normalization, policy/entropy loss, or critic loss. No IL sample is treated as an on-policy RL transition.

TensorBoard `mixed/*` metrics use **gradient-minibatch index** as their step; `mixed/environment_step` records the corresponding collected-step count. Each minibatch also logs to metra under `l2f_visual_training_mixed/*`. Diagnostics include raw `rl_output_gradient_norm` and `imitation_output_gradient_norm`, `weighted_imitation_output_gradient_norm`, each subset's raw output-gradient RMS, `imitation_weight`, achieved `output_gradient_norm_ratio`, `imitation_mse`, and `teacher_forcing`. These norms include subset averaging and accumulation scaling and are measured before the model backward pass. The RL-only `rl_log_std_gradient_norm` includes the entropy term and is logged separately.

For fixed-ratio balancing, the IL multiplier is `target_ratio * ||g_RL|| / ||g_IL||`, capped by `MAX_IMITATION_WEIGHT` and held constant during backward. A target of 1 equalizes action-mean output-gradient L2 norms. If either norm is at most `NORM_EPSILON`, the fixed `IMITATION_WEIGHT` is used and `balancing_fallback=1`; `weight_clamped` reports capping. `ratio_valid=0` marks a zero RL denominator, for which the logged ratio is a placeholder 0. The achieved ratio should always be inspected alongside these flags. Equal output-gradient norms do not imply equal parameter gradients or Adam updates: the subsets have different network Jacobians, and the policy's separate log-standard-deviation gradient is outside this balancing rule.

Architecture consumers considered: the existing pure-RL visual target constrains the non-mixed path and optimizer schedule; side-by-side RL and mixed runs require distinct executable names and target-local compile definitions; the RAPTOR imitation target constrains teacher observations, recurrence, action selection and MSE; per-scene rendering and frame-stack collection constrain the deterministic grouping; the actor and asymmetric critic require separate loss masks; the runner remains collection-only; TensorBoard/metra consumers require diagnostics before backward; CPU loss tests require a freestanding, device-first helper in `include/rl_tools/rl/algorithms/ppo/`. Other PPO targets and the hyperdrone and multi-GPU imitation targets retain their existing APIs and behavior.

`rl_environments_l2f_visual_training_mixed_cuda_small` is excluded from the default build. It runs three updates over two scenes, with two mixed minibatches per update and one teacher-forced update, to exercise both phases:

```bash
cmake --build build --target test_rl_algorithms_ppo_mixed_imitation rl_environments_l2f_visual_training_mixed_cuda_small -j5
ctest --test-dir build -R '^PPO_MIXED_IMITATION\.' --output-on-failure --timeout 20 -j5
./build/src/rl/environments/l2f_visual/rl_environments_l2f_visual_training_mixed_cuda_small /data/procthor-train-200-glb 0
```

### Imitation

```
rl_environments_l2f_visual_imitation_cuda <scene_directory | scene.glb> [seed]
```

The run is defined by the `*_TOTAL` constants in `imitation_cuda.cu` (`N_ACTIVE_SCENES_TOTAL`, hence `N_ENVIRONMENTS_TOTAL`, and `BATCH_SIZE_TOTAL`, plus `STEPS_PER_ENV`) and does not depend on the number of GPUs. The compile-time rank count `RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS` (CMake cache variable, default 1; `cmake -B build -DRL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS=2`) splits that run into equal static shards, one host thread + `devices::CUDA` + renderers + epoch dataset + student/Adam replica per GPU (`0..N_RANKS-1`): rank `r` hosts the contiguous block of global environments `r*N_ENVIRONMENTS..(r+1)*N_ENVIRONMENTS`, whose CUDA RNG streams are keyed by the global environment index, renders the global active-scene slots `r*N_ACTIVE_SCENES..(r+1)*N_ACTIVE_SCENES`, and contributes `BATCH_SIZE = BATCH_SIZE_TOTAL / N_RANKS` rows to each gradient step. Scenes are sampled from `N_SCENE_BUCKETS` (compile-time, default `N_ACTIVE_SCENES_TOTAL`) contiguous buckets of the sorted corpus: every epoch each bucket is shuffled with the shared scene RNG and its first `SLOTS_PER_BUCKET` scenes fill the bucket's active slots, so slot `a` always draws from bucket `a / SLOTS_PER_BUCKET`; rank `r` hosts buckets `r*BUCKETS_PER_RANK..(r+1)*BUCKETS_PER_RANK` and loads only their scenes (about `N_TOTAL_SCENES / N_RANKS` renderers per GPU). The bucketing is part of the run definition, so a single GPU samples the same stratified way as several. Because dataset rows are step-major and environment-minor and a gradient batch is a whole number of steps (`BATCH_SIZE_TOTAL % N_ENVIRONMENTS_TOTAL == 0` is asserted), rank-local batch `k` is exactly the rank's slice of global batch `k`, and all ranks shuffle batches with the same RNG stream, so what enters each gradient step is the same for any rank count. The only coupling is the gradient all-reduce after every backward pass (`rlt::copy_gradient` pushes each rank's gradient into the peers' shadow models, `rlt::add_gradient` reduces the rank-indexed buffers with a fixed pairwise tree; the MSE loss weight carries the `1/N_RANKS` of the mean). The reduced gradient is bit-identical on every rank, so the replicas stay in lockstep without a parameter broadcast; at checkpoint epochs rank 0 verifies this (`[consistency] ... parameter abs_diff` must be exactly 0, logged to metra as `l2f_visual_imitation/replica_parameter_abs_diff`). Results across rank counts agree in distribution, not bitwise: the per-rank partial sums are reduced in a different order than a single GPU's full-batch reduction. Rank 0 alone logs, checkpoints and records video/trajectories (its own scenes); loss and episode statistics are aggregated over ranks, the timing breakdown is rank 0's own, and `allreduce` in the epoch line is the wall-clock spent in the collective including barrier waits. Peer access is not required (GeForce cards stage the transfer through host memory), no NCCL dependency. Per-rank device memory for the epoch dataset is `STEPS_PER_ENV * N_ENVIRONMENTS * 416000` bytes (~27 GB for the default run on one GPU, ~13 GB per GPU on two).

`rl_environments_l2f_visual_imitation_cuda_small` (`RL_TOOLS_L2F_VISUAL_IMITATION_SMALL`: 2 scenes, 2 active scenes, 100 steps, 3 epochs, checkpoint cadence 2; 5.3 GB of epoch dataset, halved per rank on two GPUs) is the functional smoke profile: `rl_environments_l2f_visual_imitation_cuda_small /data/procthor-train-200-glb 0`.

### Imitation on hyperdrone (`imitation_hyperdrone.cpp`)

`rl_environments_l2f_visual_imitation_hyperdrone` is the port of `imitation_cuda.cu` onto `hyperdrone::tasks::target_frame::World` (frame stack, cached target frame, channel padding, camera/brightness randomization and free-space placement all come from the World; the driver keeps the teacher, student, training loop, logging and artifacts of the original). One device-generic source builds against every raytracing backend:

| Build | Compute | Type policy |
|---|---|---|
| OptiX backend + CUDA toolchain | CUDA (env verbs, teacher, student, training on the GPU) | bf16 parameters/activations/gradients, fp32 master (as `imitation_cuda.cu`) |
| Metal, Vulkan, WebGPU, generic | host CPU (the renderer stays on the backend device) | fp32 |
| `..._hyperdrone_cpu` (CUDA machines only) | host CPU against the OptiX backend, for A/B against the CUDA-compute build | fp32 |

Targets: `rl_environments_l2f_visual_imitation_hyperdrone`, `rl_environments_l2f_visual_imitation_state_estimation_hyperdrone` (`RL_TOOLS_L2F_VISUAL_IMITATION_STATE_ESTIMATION`, teacher drives, 15-dim body-frame label), and their `_small` variants (`RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_SMALL`: 2 scenes, 16 instances per scene, 100 steps, one training pass, 64 checkpoint examples, 3 epochs, checkpoint cadence 2). The small profile's 3,200 global dataset rows supply six 512-row training batches per pass; the remaining 128 rows are unused for training at either rank count. The full profile needs about 20 GB of device memory in bf16 or 41 GB of host memory in fp32 for the epoch dataset. Same CLI as the original:

```
rl_environments_l2f_visual_imitation_hyperdrone <scene_directory | scene.glb> [seed]
```

CUDA-compute Hyperdrone targets use the same `RL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS` CMake setting as the original CUDA targets (default 1). With two active Worlds, supported rank counts are 1 and 2. Each rank owns a host thread, CUDA device, target-frame Worlds and scene library, epoch dataset, teacher, student, and Adam state. The global workload stays fixed: 128 instances, 500 steps per instance, and a 512-row gradient batch in the full profile; two ranks each run 64 instances and contribute 256 rows per update. World 0 retains scenes `[0,12)` and World 1 retains `[12,25)` of the full corpus, with the same round-robin schedule regardless of rank count. Only locally owned scenes are loaded. CPU-compute variants always use one rank, including `_cpu` when CUDA targets are configured for two.

Each global instance has its own CUDA RNG stream: the driver sets `World::rng_offset` to the global World index times the instances per World. Hyperdrone CUDA verbs validate that this range fits the supplied RNG allocation; the default offset is zero for existing callers. Episode decisions use per-instance host RNGs; batch shuffling and checkpoint evaluation have separate RNGs, so rank-local resets and rank-0 artifacts cannot change batch alignment. This separates streams that earlier Hyperdrone runs shared, changing historical seeded trajectories. Within the new scheme, scene ownership, RNG identity, and batch membership are independent of rank count. Trained results across rank counts are compared statistically because floating-point reduction order changes.

After every backward pass, ranks use `copy_gradient` and a fixed source-rank pairwise `add_gradient` reduction, with loss weight `0.5 / N_RANKS` before summation. No NCCL or peer access is required. Rank 0 aggregates loss and episode counters and owns all logging and artifacts; its videos, trajectories, and checkpoint examples use its local scenes. Checkpoint step numbers and epoch FPS count global environment steps; collection FPS and timing components describe rank 0. Every checkpoint epoch verifies exactly zero parameter difference between replicas. Additional metra metrics are `l2f_visual_imitation/{n_ranks,batch_size,epoch_dataset_bytes_per_rank,all_reduce_time_s,replica_parameter_abs_diff}`. Dataset memory halves on two ranks; model replicas and peer-gradient buffers remain allocated per GPU.

```bash
CUDACXX=/usr/local/cuda-13.1/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release -DRL_TOOLS_L2F_VISUAL_IMITATION_N_RANKS=2
cmake --build build --target rl_environments_l2f_visual_imitation_hyperdrone_small -j5
./build/src/rl/environments/l2f_visual/rl_environments_l2f_visual_imitation_hyperdrone_small /data/procthor-train-200-glb 0
```

Deliberate differences from `imitation_cuda.cu`: the two active scenes per epoch rotate round-robin over each World's contiguous partition of the 25-scene corpus instead of being drawn at random; the random draw order differs, so runs are compared statistically; CPU-compute builds use fp32 instead of bf16; the target frame is rendered once per episode (identical output, since its pose and brightness are fixed per episode); the timing breakdown is collection/training wall-clock instead of per-kernel CUDA events. Everything else (dynamics, camera geometry and FOV randomization, shading/AA, brightness, target roll/pitch and mismatch, frame-stack layout, teacher, student, optimizer, batch scheme, loss, action-selection rule, checkpoint/video/trajectory artifacts) matches.

Both targets log the same per-epoch metrics to metra under `l2f_visual_imitation/{mse_loss, episode_length, episodes, terminated_share, complete_episode_length, complete_terminated_share, fps, epoch_time_s}` (plus `state_estimation/*` for the SE variants) and tag each run with `l2f_visual_imitation/{target, compute_device, n_environments, steps_per_env, seed}`, so learning curves overlay directly when `METRA_URL` is set. Equivalence check: run `rl_environments_l2f_visual_imitation_cuda` and `rl_environments_l2f_visual_imitation_hyperdrone` on the same corpus for a few seeds each and compare the `mse_loss`, `episode_length` and `terminated_share` curves; the port should lie within the seed-to-seed band of the original.
