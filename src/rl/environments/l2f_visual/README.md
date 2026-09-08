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

Targets: `rl_environments_l2f_visual_imitation_hyperdrone`, `rl_environments_l2f_visual_imitation_state_estimation_hyperdrone` (`RL_TOOLS_L2F_VISUAL_IMITATION_STATE_ESTIMATION`, teacher drives, 15-dim body-frame label), `rl_environments_l2f_visual_imitation_hyperdrone_small` (`RL_TOOLS_L2F_VISUAL_IMITATION_HYPERDRONE_SMALL`: 2 scenes, 16 instances per scene, 100 steps, one training pass, 64 checkpoint examples — the functional smoke profile for CPU-compute builds; the full profile needs 20 GB of device memory in bf16 or 41 GB of host memory in fp32 for the epoch dataset). Same CLI as the original:

```
rl_environments_l2f_visual_imitation_hyperdrone <scene_directory | scene.glb> [seed]
```

Deliberate differences from `imitation_cuda.cu`: the two active scenes per epoch rotate round-robin over each World's contiguous partition of the 25-scene corpus instead of being drawn at random; the random draw order differs, so runs are compared statistically; CPU-compute builds use fp32 instead of bf16; the target frame is rendered once per episode (identical output, since its pose and brightness are fixed per episode); the timing breakdown is collection/training wall-clock instead of per-kernel CUDA events. Everything else (dynamics, camera geometry and FOV randomization, shading/AA, brightness, target roll/pitch and mismatch, frame-stack layout, teacher, student, optimizer, batch scheme, loss, action-selection rule, checkpoint/video/trajectory artifacts) matches.

Both targets log the same per-epoch metrics to metra under `l2f_visual_imitation/{mse_loss, episode_length, episodes, terminated_share, complete_episode_length, complete_terminated_share, fps, epoch_time_s}` (plus `state_estimation/*` for the SE variants) and tag each run with `l2f_visual_imitation/{target, compute_device, n_environments, steps_per_env, seed}`, so learning curves overlay directly when `METRA_URL` is set. Equivalence check: run `rl_environments_l2f_visual_imitation_cuda` and `rl_environments_l2f_visual_imitation_hyperdrone` on the same corpus for a few seeds each and compare the `mse_loss`, `episode_length` and `terminated_share` curves; the port should lie within the seed-to-seed band of the original.
