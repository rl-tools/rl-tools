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

`rl_environments_l2f_visual_training_mixed_cuda` trains Patch64 from `training_mixed_cuda.cu`; compile-time defaults live in `training_mixed_config.h`. The baseline uses a forward-facing camera with zero nominal pitch, 90-degree horizontal FOV, and raw images without mechanical or digital stabilization. Estimated gravity and bias-corrected gyro feed the existing fixed controller and five-second inertial-velocity observer; the gyro-bias estimate is additionally tracked in flight with a ten-second low-pass of the raw gyro (see the startup-excursion section below). Training uses 2×2 antialiasing and evaluation uses 4×4. Building the training target also builds its matching 32-view evaluator.

```bash
CUDACXX=/usr/local/cuda-13.2/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rl_environments_l2f_visual_training_mixed_cuda -j1
RL_TOOLS_EXTRACK_PATH=/vm/sesions/l2f-visual-hold ./build/src/rl/environments/l2f_visual/rl_environments_l2f_visual_training_mixed_cuda /data/procthor-train-200-glb/ 0
```

Use the installed CUDA compiler path on your machine. Build serially on memory-constrained VMs and finish compilation before GPU work. The 600-second training budget starts after scene/model initialization; initialization and persistence add wall time. The optional seed defaults to 0. `RL_TOOLS_EXTRACK_PATH` is the parent of `experiments/`; an optional `RL_TOOLS_EXTRACK_EXPERIMENT` must be a sortable timestamp such as `2026-09-09_09-00-00`. The run records its source snapshot and prints the final checkpoint directory.

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
| Independent IL pose | Position ±20 cm per axis; attitude angle at most 0.05 rad |
| Scene rotation / physical episode limit | 2,048 / 10,000 control steps |
| Control / inner dynamics frequency | 100 Hz / 1,000 Hz |
| Training time | 600 seconds |
| Nominal camera pitch / horizontal FOV | 0 degrees / 90 degrees |
| Image stabilization / antialiasing grid | None / 2×2 training, 4×4 evaluation |
| Mahony proportional / integral gain | 0.4 / 0.01 |
| Simulated gyro calibration residual | Gaussian σ0.001 rad/s |
| In-flight gyro-bias tracking time constant | 10 seconds (`GYRO_BIAS_TRACKING_TAU`) |
| Inertial-velocity observer decay | 5 seconds |

Each IL row samples an independent pose with zero initial velocity and angular rate, and randomized camera calibration, brightness, scene placement and heading. RL rows follow physical trajectories. Each fresh sample is used once. IL rows are excluded from advantage normalization, the RL objective, entropy and critic loss. A geometric teacher labels CTBR actions using privileged body-frame position and heading together with the same inertial observations available to the actor.

The actor receives an 80×50 RGB view and its target view, plus 65 state features: Mahony estimated world-up direction (3), bias-corrected gyro (3), CTBR action history (32), body-frame acceleration history (24), and estimated body-frame velocity (3). It does not receive true position, velocity or heading. Physical gyro bias is uniform within ±0.02 rad/s. At reset, the simulator initializes the bias estimate to physical bias plus a Gaussian residual with σ0.001 rad/s, modeling a stationary calibration; it does not actually simulate or measure the calibration interval. The Mahony filter then adapts that estimate, and during flight the prologue additionally low-passes the raw gyro into the bias estimate with a ten-second time constant, which removes the calibration residual under the hold assumption of zero mean rotation without relying on the accelerometer (whose direction is tilt-blind for a drag-free multirotor). The corrected gyro is used consistently by the policy and velocity observer. Velocity starts at zero, integrates gravity-corrected acceleration with gyro-based frame rotation, and decays with a five-second time constant.

Remaining simulator assumptions are explicit: initial estimated gravity is ideal; in-flight gyro white noise/bias drift and accelerometer noise/bias are zero; the inner CTBR controller uses simulator angular velocity and calibrated dynamics. The calibration residual is hypothetical and has not been measured on the platform, and the in-flight bias tracker assumes that the vehicle does not rotate on average over about ten seconds, which holds for a position hold but not for aggressive flight. This baseline satisfies the requested camera constraint and establishes a reproducible starting point for deployment work; it is not a hardware-validated controller.

Raw images are copied directly into frame history without roll/pitch rectification or gravity-driven warping. A paired `[current,target] / [target,current]` layout feeds a fixed grayscale/local matching prefix. Learned 64-channel patch and spatial convolutions use batch normalization and GELU; a four-channel readout and fixed antisymmetric difference produce visual corrections. A fixed sensor-feedback branch supplies tilt, estimated velocity and collective-thrust feedback. Their summed outputs are collective thrust and three body-rate commands. The prefix and feedback layers are frozen by model-group selection in the actor optimizer step. Backward propagation computes all gradients; the optimizer preserves excluded parameters and their moments. The actor uses `mode::Freeze` for fixed parameters except the exploration scale, plus readout biases; all other parameters train, and the critic uses the default selection. The exploration exception is explicit because `log_std` inherits the head's `Fixed` group as well as its own `Exploration` group. The initial visual readout is zero. No RAPTOR checkpoint or auxiliary head is required.

Rendering uses High shading, no motion blur, horizontal FOV 90° ±5°, mount offsets ±1 cm, additional mount rotations ±5°, and shared current/target brightness in `[0.5,1.5]`. Target roll and pitch are zero. The fixed-workload `_small` target uses the full training workload for exactly 600 optimizer updates (307,200 environment steps), crossing the first scene rotation and teacher-blending boundary:

```bash
cmake --build build --target rl_environments_l2f_visual_training_mixed_cuda_small test_rl_environments_l2f_visual_position_hold_helpers -j1
ctest --test-dir build -R '^(L2F_VISUAL_POSITION_HOLD|PPO_MIXED_IMITATION)\.' --output-on-failure --timeout 20 -j1
RL_TOOLS_EXTRACK_PATH=/vm/sesions/l2f-visual-hold ./build/src/rl/environments/l2f_visual/rl_environments_l2f_visual_training_mixed_cuda_small /data/procthor-train-200-glb/ 0
```

TensorBoard `mixed/*` metrics use gradient-minibatch index; `mixed/environment_step` provides collected-step count. Metra records learning, timing and evaluation metrics. Balancing diagnostics report raw RL/IL action-mean gradient norms, adaptive weight, achieved ratio, fallback and capping flags; the separate RL log-standard-deviation gradient is outside this ratio. Historical random-draw sequences are retained to preserve the recorded seeded behavior, including calibration-error draws on steps where they are not applied.

Consumers and architecture decisions: training, evaluation and the fixed-update smoke share one configuration with compile-time environment counts and AA2/AA4 selection. Recorded checkpoints constrain initialization, RNG ordering, the frozen controller and canonical `[time,batch,...]` examples: image `[1,B,50,160,8]`, state `[1,B,65]`, output `[1,B,4]`; the image width 160 is the paired 80-pixel layout, not the FOV. Sensor and deployment consumers require consistent estimated gravity and corrected gyro, and no image stabilization. Extrack and model viewers require identical metadata for tar/HDF5 and canonical examples. Video reviewers require finalized regular MP4s. Pure RL, Hyperdrone, shared rendering and model operations retain their existing interfaces; this baseline changes only the mixed target's wiring and diagnostics. `camera_setup.json` records the actual camera, estimator gains, calibration residual and rendering grid; checkpoint metadata identifies the raw images, corrected gyro and AA4 replay recommendation.

### Patch64 evaluation and recorded checkpoint

```bash
RL_TOOLS_EXTRACK_PATH=/vm/sesions/l2f-visual-hold ./build/src/rl/environments/l2f_visual/rl_environments_l2f_visual_evaluate_mixed_cuda /data/procthor-train-200-glb/ 50 /vm/sesions/l2f-visual-hold/review/raw90-main-baseline/checkpoint_2examples.h5 600
```

The designated checkpoint was trained for 600 wall seconds with training seed 0, collecting 5,826,048 environment steps. Its SHA256 is `a403dd32669afed3ab261a3bc6a5fb7b6fb00226bfb84a21d08ccfbf3f9e95b0`. Main source SHA256 `ec856980d2600801e80ac6f1c08c4d4285d8f606c08c1d6dd7660d9e5bd0e74f` and configuration SHA256 `0f6733bcecfd09d4ab38ebf632c4a30d453172c6196f9d33b5fa53ca2a60772d` match its archived training source exactly. The checkpoint, replay instructions, verified videos and source provenance are indexed by `/vm/sesions/l2f-visual-hold/review/raw90-main-baseline/README.md` and `current-best.json` in that session. The weights came from the archived raw90 experiment; integration validation checks the main build separately.

The duration is simulated seconds (default 600). Before flight, CPU and CUDA predictions must match the checkpoint's two saved examples within maximum absolute action differences of `1e-4` and `1e-3`. Evaluation uses 32 trials across two scenes, starts at the target with zero velocity, and disables scene/time-limit resets. First termination or non-finite state permanently fails a trial. `evaluation.json` reports survival and position/motion statistics after a five-second settling window. `evaluation.csv` records states, policy and same-state teacher actions, estimated gravity, corrected gyro and estimated velocity at 10 Hz, stopping each trial's trace after failure. `sensor_startup.json` exposes physical bias, estimated bias and policy/observer gyro consistency.

| Raw90 normal hold, 600 seconds | All-view survival | Textured-view survival | Textured survivor mean / worst position RMS | Textured survivor mean speed |
| --- | --- | --- | --- | --- |
| Training seed 0, first25, evaluation seed 50 | 29/32 | 15/17 | 44.980 / 165.598 mm | 9.210 mm/s |
| Training seed 1, first25, evaluation seed 50 | 23/32 | 13/17 | 53.723 / 115.590 mm | 19.168 mm/s |
| Training seed 0, first25, evaluation seed 49 | 28/32 | 21/24 | 79.207 / 709.254 mm | 21.658 mm/s |
| Training seed 0, pool 75–99, evaluation seed 50 | 21/32 | 5/5 | 36.756 / 51.089 mm | 8.084 mm/s |
| Training seed 1, pool 75–99, evaluation seed 50 | 9/32 | 5/5 | 70.115 / 158.492 mm | 22.467 mm/s |

Textured subsets are fixed diagnostics, with every trial retained; no feature-based target rejection is enabled. Metrics in the last two columns include survivors only, so failures must be read alongside them. These views include textured failures and substantial drift/oscillation. Pool 75–99 has now been used for selection and is no longer an untouched test set; each run exercises two actual scenes. Recovery, hour-long holds and real hardware validation have not been established for this baseline.

KI0.02 and KI0.04 are competing runtime tradeoffs, preserved in the session records. Higher gains improve some survival/error outcomes but increase motion on other matched textured views. Fresh KI0.04 training also introduces a textured failure in pool75–99 for training seed 1. KI0.01 is selected as the common baseline for its replicated lower motion and matching trained/evaluated settings. The earlier stabilized FOV160 results remain archived in `historical-best-level160.json`; they are not the baseline for further deployment work.

Normal evaluation records target/live mosaics at 25 fps while control stays at 100 Hz. The encoder uses one thread and writes `video.partial.mp4`; only successful finalization promotes it to `steps/<step>/video.mp4`. Shared review exports must be regular file copies, with full decode, byte count, SHA256, duration and frame-count verification. A 600-second normal run has 15,000 frames; a 10× preview has 1,500 frames over 60 seconds. Extrack's `latest` links are navigation aids, not exported review files. Native/example agreement does not validate camera or sensor compatibility: replay this checkpoint with the matching raw90/KI0.01 evaluator and AA4 sampling.

Append `recovery` for per-axis initial position/velocity offsets in `[-0.2,0.2]`, `blind` to zero images, or `blind-recovery` for both. Recovery diagnostics suppress video because the renderer can reset a failed environment after its first failure. Use `teacher` in place of the checkpoint for the privileged geometric reference; normal teacher evaluation uses the same at-target initialization and records video. Finite normal holds do not establish recovery or indefinite stability.

### Startup excursion and the in-flight gyro-bias tracker

With the estimated gravity and the σ0.001 rad/s calibration residual, every hold of the raw90 baseline starts with an excursion: the mean position error over the 32 views is about 110 mm at 5 s and 200 mm at 10 s before decaying over roughly two minutes (evaluation-seed 50, `transient_profile.py` in the parallel session). The simulated accelerometer is the exact specific force of a drag-free rigid body, always along body z, so the Mahony filter can only integrate the residual away through its KP/KI = 40 s integral term; meanwhile its proportional term holds a tilt-estimate error of order residual/KP that the fixed feedback turns into a sustained lateral acceleration. Narrow imitation ranges (3 cm, 8 cm, a 75 percent-narrow mixture) lose most trials inside this excursion, a slower KP makes residual/KP larger, and the surviving trials' "after 5 s" RMS of about 50 mm is the tail of the same excursion. The in-flight tracker (`state.bias_estimate += (gyro - bias_estimate) * dt / GYRO_BIAS_TRACKING_TAU` on non-reset steps of the prologue, Mahony gains unchanged) halves the excursion; 5 s is worse than no tracker because it absorbs the vehicle's own corrective rotations, 20 s halves the benefit, 10 s is the selected value. All numbers below are training seed 0, 600 wall seconds, AA2 training / AA4 evaluation, first-25 pool unless noted, from `/vm/sesions/l2f-visual-hold/parallel/claude/README.md` (round 2).

| Normal hold, 600 s, all views retained | Baseline (this commit before the tracker) | With 10 s tracker |
| --- | --- | --- |
| Survival, evaluation seeds 50 / 49 / 47 | 29 / 26 / 23 of 32 | 32 / 31 / 26 of 32 |
| Survivor mean RMS, seeds 50 / 49 / 47 | 51.968 / 53.705 / 45.814 mm | 42.134 / 43.356 / 24.144 mm |
| Survivor worst RMS, seeds 50 / 49 / 47 | 170.418 / 173.631 / 276.470 mm | 148.112 / 132.299 / 80.257 mm |
| Mean error at 5 / 10 / 20 / 40 s, seed 50 | 112 / 201 / 183 / 113 mm | 86 / 107 / 63 / 35 mm |
| Recovery survival, seed 43 | 23/32 | 28/32 |
| Pool 75-99 survival, seed 50 | 21/32 (session record) | 26/32 |
| Pool 25-49 survival, seed 50 | not run | 25/32 |

Designated tracker checkpoint: `/vm/sesions/l2f-visual-hold/parallel/claude/review/il02bt10-checkpoint/checkpoint_2examples.h5` (SHA256 `d11e22896b66090c…`, 5,886,976 steps), with `provenance.json`, `camera_setup.json`, training record and replay command beside it; verified regular-file videos are under `review/il02bt10/2026-09-09_10-04-32.145827/` in that session. The weights were trained by the archived variant source (`checkouts/il02bt10/`, hashes in `provenance.json`), whose training semantics equal the patch in this commit; the variant additionally dumps evaluation target frames and carries unused metadata, so its binaries are not byte-identical to a main build. One training seed per variant: the survival gains are consistent across three evaluation seeds, recovery and both held-out pools but not yet replicated with a second training seed. Integration of the patch into main was validated on 2026-09-10: serial build of the training, evaluation, `_small` and helper-test targets, `ctest -R '^(L2F_VISUAL_POSITION_HOLD|PPO_MIXED_IMITATION)\.'` (2/2), a 600-update `_small` run (307,200 steps, checkpoint and `camera_setup.json` with `gyro_bias_tracking_tau_s = 10`), and a 600 s seed-50 replay of the designated checkpoint through the main-built evaluator that reproduces the variant evaluator's per-view position RMS and failure times exactly (32/32, mean / worst 42.134 / 148.112 mm; canonical-example errors 0 and 7.2e-5).

Deployment specifications established alongside: with a 0.3 mrad/s residual the excursion shrinks to 40 / 46 mm at 5 / 10 s and survival is 32/32 on seeds 50 and 49, so the platform's post-calibration gyro residual must be measured; the remaining 33-40 mm error floor of the wide-imitation policy is then the limiting factor, and the imitation range can only be narrowed once the excursion is gone. Camera timing is not limiting (the true-gravity raw90 policy kept 32/32 through 33 Hz / 30 ms up to 10 Hz / 300 ms image age). Image appearance is: clean-trained policies drop to 24-25/32 with survivor RMS tripling under sensor noise (σ up to 0.02), a 3x3 blur and a 10 percent target/live exposure mismatch, training under that randomization does not recover it within the budget, and even noise σ0.005 / half blur / 5 percent mismatch leaves 62-116 mm, which points at the frozen fine-matching frontend as the next sim2real target.

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
