# Raytracing benchmarks

The simulator benchmark supports OptiX, Metal, Vulkan, and WebGPU through the configured renderer backend. Its default workload matches `python -m hyperdrone.examples.benchmark`: ProcTHOR-Train-1, 4096 static cameras at 64×64, medium RGB, 80° FOV, no probes/AA/motion blur, seed 0, 2 seconds of warmup, and 10 seconds of measurement with synchronization every 10 launches. Python and C++ sample the same orientation distribution using different RNGs.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rendering_raytracing_sim_benchmark_medium -j5
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_medium
```

`AUTO` selects the available backend. Use `-DRL_TOOLS_RENDERING_RAYTRACING_BACKEND=METAL`, `VULKAN`, `WEBGPU`, or `OPTIX` to require one explicitly. OptiX requires CUDA; Vulkan requires headers and `glslangValidator`, plus a device with ray-query support at runtime. WebGPU uses the pinned wgpu-native runtime on Linux x86_64, runs headlessly, and selects its adapter through `RL_TOOLS_WEBGPU_DEVICE_INDEX` when set. Vulkan and WebGPU can run on lavapipe for correctness testing. Metal, Vulkan, and WebGPU do not require CUDA.

The default scene resolves `conta:75d1d67a17ecd16bc3a214baa895240da8060df7`, the same content hash as the Python example; `--procthor-path` accepts a file or a conta reference. The procedural `--scene 20_objects` case needs no downloaded scene.

## Targets and cases

Canonical targets are `rendering_raytracing_sim_benchmark_{medium,high,very_high,low}`, `rendering_raytracing_sim_benchmark_sweep`, and `rendering_raytracing_procthor_frame_{medium,high,very_high,low}`. They are built explicitly. RGB and depth share each executable; select `--output rgb|depth|all`.

The render matrix is two scenes × two outputs:

```bash
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_medium \
  --scene all --output all --output-dir /tmp/rltools-render-matrix
```

Set the compile-time camera count with `-DRL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=16` for small correctness runs. The normal resolution is 64×64; frame tools use one camera at 1000×1000. Target configuration is generated from `simulator_matrix_config.h.in` by CMake.

The sweep covers resolutions 64–2048, medium/high/low RGB and depth, with AA off and 2×2 AA. Camera count scales to preserve the normal target's pixel count, with a minimum of one camera. `--scene all --output all` produces 96 records. Configuration indices follow the selected outputs, with scenes sharing each configuration index.

```bash
./build/src/rendering/raytracing/benchmark/rendering_raytracing_sim_benchmark_sweep \
  --scene all --output all --output-dir /tmp/rltools-render-sweep
```

For a slice, use `--sweep-config-start` and `--sweep-config-count`; `--cooldown-seconds` defaults to 10 between configurations.

Frame tools accept a position and either a forward vector, look-at point, or quaternion, as shown by `--help`. With no pose arguments, they look from `[-3.92, -5.67, 1.0]` toward the scene center. `--output-png` requires a single output; otherwise artifact names are generated.

## May 2026 paper benchmarks

The throughput table in `~/papers/hyperdrone/example.tex` comes from `figures/benchmark/results{,_5090,_h200}.csv`, aggregated by `figures/benchmark/make_results_table.py`. The launcher, `benchmarks/run_all_benchmarks.py`, selected these executables from `src/rendering/raytracing/benchmark/`:

| Paper fidelity | CSV variant | Historical target | Current option |
|---|---|---|---|
| Low | `fast_flat` | `rendering_raytracing_sim_benchmark_fast_flat` | `--fidelity low` |
| Medium | `basic` | `rendering_raytracing_sim_benchmark` | `--fidelity medium` |
| High | `high_fidelity` | `rendering_raytracing_sim_benchmark_high_fidelity` | `--fidelity high` |

The historical targets compiled `simulator_matrix.cpp` and, after the May 25 dynamics fix, `simulator_matrix_physics.cu`. The separate rendering benchmark, sweep, frame, dynamic-scene, and yaw-training targets did not generate this table. Each saved GPU CSV contains exactly one 20-second measurement per configuration (24 HyperDrone rows), after 2 seconds of warmup; there is no averaging across repeated runs or seeds. RGB cells use that single measurement. Depth cells average the three fidelity measurements for each scene/mode, then repeat that mean in all three fidelity rows (`make_results_table.py::average_hyperdrone_depth_fps`).

The saved CSV paths establish the runs below. They contain no Git revision or dirty-state record, and neither did the launcher's manifest; the original logs/manifests are absent from this paper checkout. Source revisions below are reconstructions, not recorded executable identities.

| Published rows | Saved run | Source evidence and camera contract |
|---|---|---|
| RTX 4090 ProcTHOR | `hyperdrone_physics_fix_rerun_20260525_171252` | Strongly inferred `6c7fc4c6e0c912c451e40856b80740e65f129635` (May 25, 17:10:58): physics updates rendered cameras. |
| RTX 4090 objects20 | `static_camera_20s` | Revision unresolved; static-camera contract matches `10bee5cdc6616e6c42c16cf8966abff3f007ef06`. Paper commit `277eafeb3a20ea664346b938303deadbcf8443df` imported these rows four minutes before that source commit, so a clean committed build cannot be assumed. |
| H200 | `h200_static_camera_20s_20260528_234935` | Inferred `10bee5cdc6616e6c42c16cf8966abff3f007ef06` (May 28, 23:24:26): physics runs into a separate camera buffer; rendered cameras remain static. |
| RTX 5090 | `h200_static_camera_20s_20260529_004330` | Same inferred revision/static-camera contract; the CSV GPU name identifies the 5090 despite the run-name prefix. |

Paper commit `77f4c5dfe33fb79d4b68841c87605eeedcc95ee3` imported the May 25 physics rerun; `277eafeb3a20ea664346b938303deadbcf8443df` subsequently replaced only its objects20 rows. Commits `e729a4fde8cb3405d6ee75f63a60ec69a1a5502f` and `d446c793dac5831825756f12f9c30bbef2f012ed` imported the final H200 and 5090 CSVs. The published 4090 table therefore mixes camera contracts. Before RLtools commit `a8d98a52da105740b772a1f7d014b7fd385f46b8`, `render_physics` only rewrote fixed camera poses and did not execute multirotor dynamics.

The paper target preserves the two scenes, RGB/depth matrix, 4096 cameras at 64×64, 80° FOV, seed 0, uniform SO(3) orientations sampled once per camera, no probes/AA/motion blur, 2-second warmup, 20-second measurement, and synchronization every 10 launches. Spawns are fixed at `[0,0,0]` for `canonical_staggered_v1` objects20 (alternating boxes and spheres), and `[-3.92,-5.67,1]` FLU for ProcTHOR; the paper's rejection-sampling discussion describes training, not this benchmark.

The ProcTHOR asset is the May test-data revision `d39a1ede85a505af01afd897aa51779f87db3cc1`, file `ProcTHOR-Train-1.glb`, SHA-256 `76c463017f2a25683abaea3b29786294602bd2629b2f0725b7215dabf58f35aa`. The paper target rejects a different asset. On an OptiX, Metal, Vulkan, or WebGPU build, download it once, then run the full 24-cell matrix:

```bash
.venv/bin/python src/rendering/raytracing/benchmark/download_paper_scene.py
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rendering_raytracing_paper_benchmark -j5
./build/src/rendering/raytracing/benchmark/rendering_raytracing_paper_benchmark \
  --output-dir /tmp/rltools-paper-benchmark
```

The defaults run 3 complete passes through 3 fidelities × 2 scenes × 2 outputs × 2 workloads = 24 cells, producing 72 measurements in the order `1…24, 1…24, 1…24`. Each cell starts fresh with the same seed and warmup; `--repeats N` overrides the pass count. `--fidelity low|medium|high|all`, `--scene 20_objects|procthor|all`, `--output rgb|depth|all`, and `--workload` all default to `all`. For workloads, `all` means `render_only` plus `render_physics_static`, matching the later H200/5090 camera contract. Select `--workload render_physics_coupled` separately for the May 25 moving-camera variant. Physics runs on CUDA for OptiX and on one CPU thread for Metal/Vulkan/WebGPU, selected at build time. Each fidelity uses its original compiled renderer configuration; selection happens outside timing. Pass `--procthor-path` for the pinned asset at another location; its default is `build/paper-benchmark/ProcTHOR-Train-1.glb`.

Physics uses the Crazyflie direct-motor model, float precision, 100 Hz integration with one substep, integrated rotor dynamics, zero disturbances/noise, zero initial velocities, and hovering initial motor speeds. Actions apply the historical ±0.010 sinusoidal roll/pitch perturbations; initial states are restored every 10 iterations or on displacement squared greater than 16 or vertical displacement greater than 3. Warmup state carries into measurement while the timed iteration index restarts at zero.

Both physics backends execute the same frozen contract, including scratch-camera generation in static mode. CPU storage is allocated before warmup; coupled mode copies the generated cameras through the renderer's synchronized memory API. Physics stepping and camera transfers are included in elapsed time, so CPU/GPU combinations measure their complete execution path.

This target preserves the task and historical shading flags while using the current renderer. Lighting/material-import fixes, ambient/background colors, and depth semantics have evolved: current depth is forward-projected distance with infinite misses, whereas May used ray distance and finite misses. Results are not promised to match May pixels or FPS exactly.

Paper results are written to stdout and one `rendering_raytracing_paper_benchmark_results_<timestamp>.jsonl` in `--output-dir` (default: current directory), with identical JSON lines flushed after each completed cell, outside timing. Each raw measurement records its one-based `repeat`, total `repeats`, and `physics_backend` (`cpu`, `cuda`, or `null` for rendering alone); verification PNG filenames include the repeat number, and physics metric paths include the execution backend. Records include the reconstructed historical source revision, build revision, tracked dirty state (when git snapshotting is enabled), and a SHA-256 fingerprint of the benchmark sources, including untracked additions. The fingerprint covers the target wiring, generated-header template, scene/camera driver, and both physics implementations/contract; it does not identify the entire renderer or toolchain. `historical_source_commit` describes the reconstructed workload, not a verified revision of the original executable.

Consumers considered: the published throughput table constrains paper scene, spawn, dynamics, camera-coupling, timing, and separate depth measurements for all three fidelities; the benchmark CLI and contract tests require three complete matrix passes to separate repeated cells in time; JSONL analysis and image inspection require repeat identifiers and distinct PNGs. Metal/Vulkan/WebGPU require CPU stepping and synchronized host camera transfer; OptiX retains its CUDA stream producer. Headless WebGPU result labels identify the selected native adapter. The later H200/5090 runs determine the default physics-camera contract, while the mixed 4090 results retain explicit coupled-camera selection. The current simulator benchmark, Python benchmark, sweep, and frame tools constrain keeping their target names, defaults, and `render-v1` output contract.

## Measurement and results

Warmup completes before timing. With `--warmup-seconds 0`, general targets prime once and the paper target retains the historical ten warmup iterations. `--iterations N` selects a fixed count instead of duration; the final batch is drained even when it is shorter than `--sync-interval`. Scene loading, image saving, and reporting are outside the measured interval. Vulkan and WebGPU currently wait for the previous submission in each `render_launch`; WebGPU also copies every frame through staging buffers into host tensors, including mapping waits and readback in elapsed time. Equal sync intervals do not imply equal queue depth across backends.

Vulkan keeps one instance and selected physical device alive across all cases to avoid repeated driver loading; each case still allocates and frees its own logical device and rendering resources.

Normal, sweep, and frame tools emit one `render-v1` JSON object per result on its own line. Numbers and booleans retain their JSON types; camera vectors are arrays, with unavailable forward/up vectors represented by `null`. Records identify the workload, backend, selected device, scene source, scene content identity, camera configuration, sampling, timing, and verification PNG. Ray throughput counts primary samples; shading may trace additional rays. Frame measurements have workload `frame`; general benchmark measurements have workload `render_only`, while the paper target reports its selected workload. Extract JSON Lines with `rg '^\{' output.log`. Throughput and elapsed time are also logged to metra when `METRA_URL` is set. Backend, device, workload, scene reference, profile, resolution, camera count, orientation, seed, and sync interval distinguish metric series.

This replaces the old CUDA-specific CSV/JSON layouts. General renderer targets no longer expose `render_physics`, redundant target aliases, separate RGB/depth targets, `--step-mode`, `--gpu-label`, `--warmup-iterations`, or runtime `--num-envs`/`--resolution` options. The dedicated paper target above identifies its physics/camera contract explicitly.

## Validation

After building the medium benchmark, sweep, and medium frame targets:

```bash
.venv/bin/python tests/src/rendering/raytracing/benchmark.py --build-dir build --backend metal --with-procthor
```

Select the matching backend and omit `--with-procthor` for procedural-only checks. Each subprocess has a 20-second timeout. Vulkan and WebGPU CI use 16 cameras on lavapipe; production defaults remain 4096.

For headless WebGPU correctness checks, configure a separate build with `-DRL_TOOLS_RENDERING_RAYTRACING_BACKEND=WEBGPU -DRL_TOOLS_RENDERING_RAYTRACING_SIM_BENCHMARK_NUM_ENVS=16`, use `--backend webgpu`, and set `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json` for lavapipe. Use a separate 4096-camera build on the selected physical GPU for throughput measurements; reduced-camera runs do not qualify the production workload. Validate the pinned paper asset separately from the general benchmark's newer scene.

Paper contract checks build the unified target plus the general medium target:

```bash
cmake --build build --target rendering_raytracing_paper_benchmark \
  rendering_raytracing_sim_benchmark_medium test_rendering_raytracing_paper_physics \
  test_rendering_raytracing_paper_physics_cpu -j5
.venv/bin/python tests/src/rendering/raytracing/paper_benchmark.py --build-dir build --backend metal --with-procthor
ctest --test-dir build -L '^paper-benchmark$' --output-on-failure --timeout 20 -j5
```

The CPU dynamics test compares against trajectories captured from the May 25 source under strict floating-point flags; the CPU batch driver test uses production compiler flags and covers historical camera positions, reset boundaries, and reinitialization. Python checks verify static/coupled behavior, synchronization, and all 72 ordered records on every supported paper backend. Vulkan and WebGPU CI run these checks on lavapipe, including the pinned May asset.
