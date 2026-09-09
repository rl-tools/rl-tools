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

### Hyperdrone on-policy training

```
rl_environments_l2f_visual_training_cuda_hyperdrone <scene_directory | scene.glb> [seed]
```

Hyperdrone follows the pure-RL `rl_environments_l2f_visual_training_cuda` defaults: float32, 32 environments, 320 steps per rollout, ten 1,024-sample minibatches accumulated into one actor and critic Adam update, and one use of every transition. Its target-local `training_hyperdrone_config.h` pins the reference parameters. The actor/critic wiring stays in `training_cuda_hyperdrone.cu`; `training_hyperdrone_loss.h` implements the reference advantage-weighted log-probability objective, entropy regularization and minibatch advantage normalization. The Hyperdrone port does not modify `training_cuda.cu`, shared PPO operations or neural-network backends. Mixed imitation/RL is a separate CUDA target described below; Hyperdrone remains pure RL.

Both targets shuffle the numerically ordered 25-scene corpus with a dedicated seeded scene RNG and select two scenes every seven rollouts (2,240 steps per environment). `training_hyperdrone_scenes.h` mirrors the reference enumeration. Both Hyperdrone Worlds can select any scene through the existing shared scene library and explicit `select_scene`, using 50 renderer slots with shared geometry builds. Their RNG offsets are 0 and 16. The existing `rotate_scene` behavior is preserved for other callers.

Hyperdrone retains same-step autoreset and pre-reset critic observations for truncation bootstrapping. A raw observation refreshes the task target cache before composed observations are requested. The target is cached for each episode; history has 411 frames to accommodate the final observation. Episode completion reports follow the CUDA target's next-reset convention, including scene boundaries without double-counting task endings. RNG draw order and reset timing differ between targets, so historical seeded trajectories are not interchangeable.

Extrack uses `algorithm_environments_accumulation=on-policy_32_1` and the distinct Hyperdrone target name. The first checkpoint/video endpoint is 71,680 environment steps; subsequent saved groups are 184 scene sets apart. Checkpoint examples preserve `[TIME, BATCH, ...]` axes, and trajectories include scene yaw/hash and per-episode parameters captured before autoreset.

The Hyperdrone smoke target retains the full corpus and production rollout/batch sizes, completes two scene sets (143,360 transitions and 14 updates), and enables HDF5/tar exports. It checks all stacked/target/padding channels at five image positions for every transition against the actual rollout inputs, including history wraparound and resets, and compares training action means with collection means before the update.

```bash
cmake --build build --target rl_environments_l2f_visual_training_cuda_hyperdrone_smoke test_rl_environments_l2f_visual_hyperdrone_gradient -j5
timeout 300 ./build/src/rl/environments/l2f_visual/rl_environments_l2f_visual_training_cuda_hyperdrone_smoke /data/procthor-train-200-glb 0
ctest --test-dir build -R '^L2F_VISUAL_HYPERDRONE_ON_POLICY_GRADIENT\.' --output-on-failure --timeout 20 -j5
```

### Patch64 mixed visual position hold

`rl_environments_l2f_visual_training_mixed_cuda` trains Patch64 from `training_mixed_cuda.cu`; compile-time defaults live in `training_mixed_config.h`. The current setup uses a nominal camera pitch of 45 degrees downward, horizontal FOV 110 degrees, and the five-second inertial-velocity observer. Training uses 2×2 antialiasing and evaluation uses 4×4, selected at compile time. The original forward-camera Patch64 integration is preserved in commit `cbc1b3e956d53002c745723b9e774aafbfdf96f8`; its recorded checkpoint and preserved binaries are described below. Building the training target also builds its matching 32-view evaluator. The pure-RL and Hyperdrone targets retain their own sources and defaults.

```bash
CUDACXX=/usr/local/cuda-13.2/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rl_environments_l2f_visual_training_mixed_cuda -j1
RL_TOOLS_EXTRACK_PATH=/vm/sesions/l2f-visual-hold ./build/src/rl/environments/l2f_visual/rl_environments_l2f_visual_training_mixed_cuda /data/procthor-train-200-glb/ 0
```

Use the installed CUDA compiler path on your machine. Build serially on memory-constrained VMs and finish compilation before starting GPU training. The 600-second training budget starts after scene/model initialization; initialization and final persistence add wall time. The optional seed defaults to 0. `RL_TOOLS_EXTRACK_PATH` is the parent of `experiments/`; optionally set `RL_TOOLS_EXTRACK_EXPERIMENT` to a sortable timestamp such as `2026-09-08_10-00-00`. The run records its source snapshot and prints the final checkpoint directory.

| Default | Value |
| --- | --- |
| Scene corpus / active scenes | 25 / 8 |
| Environments / fresh samples per update | 128 / 512 (4 steps each) |
| IL / RL samples per update | 256 / 256, fixed slots in each scene |
| Adam learning rate / epochs | `3e-4` / 1 |
| Initial RL action standard deviation | `0.05` |
| IL action schedule | Teacher for 500 updates; blend toward student through update 2000 |
| Teacher action noise | `0.01` standard deviation |
| IL / RL action-gradient norm ratio | 50, adaptive IL weight capped at `1e8` |
| Scene rotation / physical episode limit | 2,048 / 10,000 control steps |
| Control / inner dynamics frequency | 100 Hz / 1,000 Hz |
| Training time | 600 seconds |
| Nominal camera pitch / horizontal FOV | 45 degrees downward / 110 degrees |
| Antialiasing grid, training / evaluation | 2×2 / 4×4 |

Each IL row samples an independent pose: position within ±3 cm per axis, attitude angle at most 0.015 radians, zero initial velocity and angular rate, and independently sampled camera calibration, brightness, scene placement and heading. RL rows follow physical trajectories. Each fresh sample is used once. IL rows are excluded from advantage normalization, the RL objective, entropy and critic loss. A geometric teacher labels CTBR actions using privileged body-frame position and heading together with the same inertial observations available to the actor.

The actor receives an 80×50 RGB view and its target view, plus 65 state features: world-up direction (3), gyroscope (3), CTBR action history (32), body-frame acceleration history (24), and estimated body-frame velocity (3). It does not receive true position, velocity or heading. The simulator supplies the world-up direction from orientation; gyroscope bias is uniform within ±0.02 rad/s and accelerometer noise/bias are zero in this baseline. Velocity starts at zero, integrates gravity-corrected acceleration with gyro-based frame rotation, and decays with a five-second time constant. These assumptions matter when interpreting hold and recovery performance.

Camera preprocessing uses the actual randomized calibration to rectify roll and pitch while preserving yaw. A paired `[current,target] / [target,current]` image layout feeds a fixed grayscale/local matching prefix. A learned 64-channel 1×1 patch convolution and 64-channel spatial convolution use batch normalization and GELU; a four-channel readout and fixed antisymmetric difference produce visual corrections. A fixed sensor-feedback branch stabilizes tilt, estimated velocity and collective thrust. Their summed outputs are collective thrust and three body-rate commands, consumed by the existing CTBR controller. The inner controller uses simulator angular velocity and dynamics calibration. The fixed prefix and feedback layers are frozen; backward propagation stops at the first learned image layer. The initial visual readout is zero. No RAPTOR checkpoint, auxiliary head or synthetic-warp training pass is required.

Rendering uses High shading, 2×2 antialiasing during training and 4×4 during evaluation, no motion blur, nominal camera pitch 45° downward, horizontal FOV 110° ±5°, mount offsets ±1 cm, additional mount rotations ±5°, and shared current/target brightness in `[0.5,1.5]`. Target roll and pitch are zero. The fixed-workload `_small` target uses the full training workload for exactly 600 optimizer updates (307,200 environment steps), crossing the first scene rotation and teacher-blending boundary:

```bash
cmake --build build --target rl_environments_l2f_visual_training_mixed_cuda_small test_rl_environments_l2f_visual_position_hold_helpers -j1
ctest --test-dir build -R '^(L2F_VISUAL_POSITION_HOLD|PPO_MIXED_IMITATION)\.' --output-on-failure --timeout 20 -j1
RL_TOOLS_EXTRACK_PATH=/vm/sesions/l2f-visual-hold ./build/src/rl/environments/l2f_visual/rl_environments_l2f_visual_training_mixed_cuda_small /data/procthor-train-200-glb/ 0
```

TensorBoard `mixed/*` metrics use gradient-minibatch index; `mixed/environment_step` provides collected-step count. Metra records learning, timing and evaluation metrics. The balancing diagnostics report raw RL/IL action-mean gradient norms, adaptive weight, achieved ratio, fallback and capping flags; the separate RL log-standard-deviation gradient is outside this ratio. Two otherwise unused historical random-draw sequences are retained to preserve the recorded seed behavior.

Architecture consumers considered: independent training and evaluation require one actor/configuration with distinct environment counts; the recorded checkpoint constrains initialization, preprocessing, RNG ordering and canonical input shapes; pure-RL, Hyperdrone and RAPTOR imitation consumers constrain keeping this target separate; embedded/device consumers constrain freestanding helpers for velocity, camera projection and fixed model initialization; sequential-model consumers constrain a reusable backward operation that stops at a specified trainable layer; CUDA convolution caches require one backward pass through the learned image branch; Extrack and model viewers constrain canonical `[time,batch,...]` saved examples; video reviewers constrain finalized regular MP4 artifacts. The existing mixed-loss helper remains responsible for subset masks and gradient balancing, while the target owns progress and reporting. For the downward-camera setup, rendering and rectification consume the same actual randomized calibration; rotate the nominal mount once before parameter sampling to retain RNG sequencing. Training/evaluation and raw-input deployment require matching camera geometry, while checkpoint model viewers retain the existing canonical input shapes. `camera_setup.json` in each run records camera geometry, observer time constant, actual antialiasing grid and configured training/evaluation defaults. The ten-minute training budget constrains training sampling to 2×2; evaluation and raw-input deployment use 4×4. The existing `EVALUATION_BUILD` selection also keeps `_small` at the training grid. Checkpoint rendering metadata preserves the actual grid used during training and adds `evaluation_default_anti_aliasing_grid_size` as a replay recommendation; tar and HDF5 receive the same metadata string. This is target-local wiring through existing renderer specifications, with no shared renderer or other target changes.

### Patch64 evaluation and recorded checkpoint

```bash
RL_TOOLS_EXTRACK_PATH=/vm/sesions/l2f-visual-hold ./build/src/rl/environments/l2f_visual/rl_environments_l2f_visual_evaluate_mixed_cuda /data/procthor-train-200-glb/ 42 /path/to/checkpoint_2examples.h5 600
```

The duration is simulated seconds (default 600). Before flight, CPU and CUDA predictions must match the checkpoint's two saved examples within maximum absolute action differences of `1e-4` and `1e-3`, respectively. Evaluation uses 32 trials across two scenes, starts at the target with zero velocity, and disables scene/time-limit resets. The first termination or non-finite state permanently fails a trial. `evaluation.json` reports survival, position/speed RMS, bias, variation and drift after a five-second settling window; drift compares the mean position at seconds 5–10 with the final five seconds. `evaluation.csv` records states and actions at 10 Hz for the full run, stopping each trial's trace after failure.

Use an evaluator with the checkpoint's camera configuration and the intended evaluation sampling grid. Canonical-example checks validate network inference; they do not check physical camera or renderer compatibility. The current evaluator uses the downward camera and 4×4 antialiasing. Archived checkpoints retain their original training metadata; the designated checkpoint's adjacent setup instructions specify AA4 replay. Normal evaluation records target/live mosaics at 25 fps while control remains at 100 Hz. The encoder runs with one thread and writes `video.partial.mp4`; only successful finalization promotes it to `steps/<step>/video.mp4`. Use that regular file for copying or review. Extrack's `latest` entries are navigation links. Shared review exports must be real file copies and should be checked with `ffprobe` for duration and frame count (a 600-second normal run has 15,000 frames).

Append `recovery` for initial per-axis position and velocity offsets in `[-0.2,0.2]`, `blind` to zero images, or `blind-recovery` for both. Recovery diagnostics suppress video because the renderer can reset a failed environment after its first failure. Use `teacher` in place of the checkpoint for the geometric privileged reference. Hold success does not establish recovery from arbitrary displacement or velocity, or indefinite stability.

The designated checkpoint is `/vm/sesions/l2f-visual-hold/review/aa2train-aa4eval-checkpoint/checkpoint_2examples.h5`. These are the unchanged down45/FOV110 training-seed-1 weights, trained for 600 wall seconds and 5,733,376 environment steps with AA2. Its archived training source SHA-256 is `48372467b82b7cba18750c3270ae0bd27d251fc29fa5d26eecfa119e55ca7dd2` and configuration SHA-256 is `d266933ef2c05508e962ad61a1d39480b65b71479b3f23c5c57c4676ba997603`. Current defaults preserve that training computation and use AA4 for evaluation. The original checkpoint metadata is preserved; the adjacent README/provenance records the intended runtime grid and matching evaluator. Pass `1` for the designated training seed; the default dynamic seed remains `0`.

Fixed-update validation compares 600 optimizer updates (307,200 steps) on seed 0. New main and a fresh build of the unchanged prior source/configuration match all 41 checkpoint datasets exactly. The preserved old binary reproduces its archived smoke exactly, but fresh builds differ from that binary in 17 datasets; visual-weight differences reach `0.001727`, log-standard-deviation differences `0.003811`, and saved output differences `0.000155`. The build-level cause is not isolated. This limits exact reproduction of historical training weights; the source/configuration edits do not explain the difference. Checkpoint replay still reproduces the reviewed evaluation report, CSV and video exactly. Both the preserved binary and fresh-build controls remain archived in the session.

| Same checkpoint, nine evaluation seeds (42–50), 600 s each | AA2 evaluation | AA4 evaluation |
| --- | --- | --- |
| Surviving trials | 288/288 | 288/288 |
| Mean per-trial position RMS | 2.70 mm | 1.06 mm |
| Worst per-trial position RMS | 14.62 mm | 6.71 mm |
| Maximum drift | 10.39 mm | 7.56 mm |
| Worst position variation RMS | 6.82 mm | 2.94 mm |
| RMS and drift both ≤5 cm | 288/288 | 288/288 |
| Recovery survival, seed 43 | 21/32 | 21/32 |

The 5-cm row is descriptive and does not change termination criteria. These runs use the first 25 scenes in the numerically ordered directory. Recovery survivors may retain substantial errors; counts alone do not imply accurate recovery. The fixed camera, observer, model, optimizer and imitation pose distribution are unchanged. Deterministic denser spatial sampling reduces the measured residual movement; it does not remove all offsets or oscillation.

The AA4 one-hour hold on evaluation seed 50 completes 32/32 trials with mean/worst RMS 1.39/4.57 mm and maximum drift 3.27 mm. In the final five-minute window, the worst displacement between five-second boundary averages is 2.93 mm and worst position variation RMS is 2.94 mm.

The AA4 one-hour hold on evaluation seed 49 completes 32/32 trials with mean/worst RMS 1.39/6.27 mm and maximum drift 3.30 mm. In the final five-minute window, the worst displacement between five-second boundary averages is 1.53 mm and worst position variation RMS is 2.57 mm.

The existing seed-0 checkpoint from the same 600-second AA2 training configuration was also checked with AA4 on evaluation seeds 50/49/47: 96/96 normal trials survived, with mean/worst RMS 1.43/6.84 mm and recovery survival 27/32. These are two recorded training seeds and finite fixed-pool tests; universal or indefinite stability and generalization to every scene are not established.

Fresh training with AA3 and AA4 collected 5,069,312 and 4,328,960 steps in the same ten-minute budget. On the matched seeds 50/49/47, their mean RMS was 2.43/1.70 mm and worst variation was 5.28/5.40 mm, compared with 1.25/2.94 mm for AA2 training with AA4 evaluation. Recovery survival improved to 25/24 versus 21, but the normal hold tradeoff favored retaining AA2 training. Fresh AA4 also introduced a sustained slow oscillation on seed 47/view 24. The comparison and alternative checkpoints remain documented in the session SUMMARY.md.

Verified regular videos and full provenance are indexed by `/vm/sesions/l2f-visual-hold/SUMMARY.md` and `review/aa2train-aa4eval-checkpoint/README.md`. Every review records size, SHA256, duration, frame count and full-decode checks. Hour reviews include 60× previews and full 90,000-frame captures. Failed views remain marked in previews; raw captures are separately identified.

The prior AA2 training/evaluation setup is preserved at isolated commit `b3624532dbbffff2f6cfabe3c843c116a2de3f6e`, with its validated main binaries/source in `scratch/down45-aa2-main-binaries`. It completed 288/288 normal holds and an hour on seed 50, with nine-seed mean/worst RMS 2.70/14.62 mm and worst variation 6.82 mm. The earlier down30 setup is preserved at `38ae01bde396d40009779066210bbb967dd26093`, with checkpoint `review/camera-down30-checkpoint/checkpoint_2examples.h5` and main binaries in `scratch/down30-main-binaries`; it failed seed 47/view 21 and drifted in two nearly featureless views. The camera change to down45/FOV110 restored position/yaw sensitivity across all 128 geometry probes on seeds 43/44/46/47. Those local information checks motivated camera training but are not stability proofs. Other camera and training-seed alternatives remain in SUMMARY.md.

### Original forward-camera checkpoint

The original checkpoint requires its original forward-camera configuration. On this VM, its previously validated evaluator is preserved at `/vm/sesions/l2f-visual-hold/scratch/baseline-main-binaries/rl_environments_l2f_visual_evaluate_mixed_cuda`; the training/small binaries and exact committed source/configuration are preserved beside it. The original source is also available at commit `cbc1b3e956d53002c745723b9e774aafbfdf96f8`.

The original reviewed checkpoint is `/vm/experiments/2026-09-08_08-44-02.571427/c75e5f8_l2f_visual_ctbr_patch_context64_real_algorithm_environments_accumulation/mixed_128_1/0000/steps/000000005827072/checkpoint_2examples.h5`. It trained from scratch for 600 seconds and collected 5,827,072 environment steps. Its run directory archives the exact prototype source and build metadata; the source SHA-256 is `391ccae9b5ed0ba02b048b29d2c4568094d6986eb278fa90f81b4d4055962357`. The earlier commit titled `interesting results` contained shared backend fixes but did not yet contain this external prototype. Commit `cbc1b3e95` integrated that prototype into the main source before the later camera change.

| Recorded 600-second evaluation | Seed 42 | Seed 43 |
| --- | --- | --- |
| Surviving trials | 32/32 | 32/32 |
| Mean per-trial position RMS | 1.32 cm | 3.21 cm |
| Worst per-trial position RMS | 10.99 cm | 22.98 cm |
| Maximum positional drift | 12.68 cm | 38.90 cm |

These are results of the original archived checkpoint, not a claim that all ten-minute runs produce identical final checkpoints: a wall-time limit yields hardware-dependent update counts. The preserved forward-camera evaluator exactly reproduces both original evaluation reports and raw MP4 files. Training across the source refactor has small numerical differences: first-update advantages differ by at most `4.77e-7` and readout weights by `5.82e-11`, which accumulate over subsequent updates; it is not byte-identical to the archived prototype. Fixed-seed, fixed-update repeats should be compared separately from wall-time runs. Original review videos and full evaluation files are in `/vm/data/rl-tools/visual-hold/patch64-video-review/`; the hardest-view montage exposes the residual drift hidden by averages. The HDF5 actor includes canonical preprocessed image/state examples for model inspection; running it from raw sensor/image input also requires the camera rectification, paired layout, velocity observer and CTBR interface described above.

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
