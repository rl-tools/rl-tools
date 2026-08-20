# Raytracing Renderer

Batch raytracing for visual RL: N cameras over one static scene plus per-camera dynamic
overlays, rendered by one of four backends (OptiX, Vulkan, Metal, generic CPU) behind a single
`Renderer<SPEC, BACKEND>` interface (`BACKEND` defaults to the configured backend).
Everything is wired for a training step in which dynamics,
rendering, and observation never touch the CPU: the host's only per-step role is issuing
asynchronous enqueues.

## Coordinate frame

The renderer operates in the same FLU frame as L2F: camera principal axis = +X (forward), image
horizontal = Y, image vertical = Z. GLB meshes (Y-up) are swizzled to FLU at load time.

## Convention

- **Backend selection is typed dispatch.** The renderer's second template argument is the
  backend tag (`backends::{Generic,Optix,Metal,Vulkan,Webgpu}`, defaulting to `backends::Default`,
  which the CMake backend macro resolves). Each backend header overloads the public verbs
  directly on `Renderer<SPEC, backends::X>`, so `render(device, renderer)` resolves to the
  backend baked into the renderer's type, and the same tag selects the backend state types.
  The `DEVICE` argument keeps its usual RLtools meaning: it says where caller-owned tensors
  live and (on CUDA) carries the producer stream — it does not select the render backend.
- **State is a tensor behind an accessor.** Renderer inputs and outputs are backend-native
  `Tensor` members exposed by accessors — `cameras()`, `cameras_open()`/`cameras_close()`
  (motion blur), `transforms()` (overlays), `transforms_motion()` (dynamic motion blur:
  per-sample overlay transforms, sample-major so slab `s` shares the `transforms()` layout),
  `transforms_pair()` (dynamic motion blur producer input: shutter-open/close entries),
  `frame_buffer()`, `depth_buffer()`, `segmentation_buffer()`, `normals_buffer()`,
  `flow_buffer()`, `flow_deltas()`, `collision_results()`, `observation()`. Residency is a
  backend property: CUDA device memory on OptiX, host on generic and WebGPU (WebGPU settles
  outputs from staging at the sync boundary), shared/mapped on Metal/Vulkan.
- **Data moves via tensor copies and kernels.** No raw backend handles or `cudaStream_t` appear in
  public signatures. Device producers (extraction kernels) write the input tensors in place;
  device consumers read the output tensors in place; host readers/writers cross the residency
  boundary with a regular tensor copy against the renderer's memory-domain device:
  `rlt::copy(renderer.device, device, rlt::frame_buffer(device, renderer), host_tensor)`
  (readback) and `rlt::copy(device, renderer.device, host_tensor, rlt::cameras(device, renderer))`
  (upload). `renderer.device` (`backends::Device<BACKEND>`, populated in `malloc`) carries only
  what the copy needs — the renderer-owned streams on OptiX, the in-flight context on
  Metal/Vulkan/WebGPU — and the copy orders itself after the renderer's in-flight work, so no separate
  `synchronize` is needed at readback/upload boundaries. It is a copy-dispatch handle only:
  backend selection for the verbs remains the renderer's template argument.
- **Verbs enqueue; waiting is separate.** `render`/`probe`/`update` each split into `_launch`
  (pure enqueue) and `_sync` (true boundary: readback, benchmarks, teardown); the fused verb is
  launch + sync. On OptiX, calling a launch verb with a CUDA device makes the backend stream
  wait on `device.stream` via an event, so producer kernels need no user-facing stream plumbing;
  `stream(device, renderer)` exists for power users. Write inputs only between frames — a launch
  in flight may still be reading them.
- **Spec-driven observation output.** With `OUTPUT_OBSERVATION`, the RGB ray gen writes float
  pixels (frame-buffer color before 8-bit quantization) directly to `observation()` — no
  format-conversion pass. The packed `uint32` frame buffer remains the video/golden output.
- **Normals output.** With `OUTPUT_NORMALS`, a segmentation-style single-sample pass
  (shutter-close camera, pixel-center ray) writes the world-frame (FLU) geometric unit normal
  of the hit surface, oriented against the ray, as float3 per pixel to `normals_buffer()`
  (miss = zero). Unit normals cannot be averaged, so anti-aliasing and motion-blur settings do
  not apply, and the pass stays out of the dynamic-motion-blur accumulate/resolve machinery.
- **Flow output.** With `OUTPUT_FLOW`, another single-sample pass writes the backward optical
  flow of the shutter-close frame in pixels to `flow_buffer()` (float2, miss = zero): the hit
  point is carried to shutter open by its instance's shutter motion and projected through the
  shutter-open camera. The two instants are the motion-blur pair inputs — `cameras_open()`
  (whose storage flow enables independently of motion blur) and the `set_transform_pair`
  shutter poses, composed per overlay slot into `flow_deltas()` (world_open ∘ world_close⁻¹;
  single-pose verbs replicate, so held objects contribute pure camera flow). A producer drives
  frame-to-frame flow by writing last frame's state into the open slots. On the device-resident
  pair path the deltas are expanded on-device alongside `expand_motion_transforms`.
- **Determinism is a contract.** Fixed seed ⇒ identical results, no atomics. The global
  instance-id layout (scene instances `[0,S)`, overlay `o` slot `s` at `S + o*MAX + s`) is
  cross-backend API surface consumed by segmentation.
- **Dynamic motion blur** (`ENABLE_DYNAMIC_MOTION_BLUR`, requires motion blur + overlays):
  `render_launch` internally runs `MOTION_BLUR_SAMPLES` sequential passes, rebuilding the
  overlay acceleration structures from the per-sample `transforms_motion()` slabs between
  passes (camera lerped at the same `(i+0.5)/N` shutter times), accumulating linear radiance
  per pixel (each thread owns its pixel — no atomics), then restores the shutter-close overlay
  state (segmentation and probes stay single-sample at shutter close, ids unchanged) and
  resolves with the standard transfer curve + quantization. `set_transform_pair` expands
  shutter-open/close entries into the samples by constant-twist screw interpolation (exact for
  constant-velocity rigid motion below 180° per shutter — an off-origin rotation axis such as a
  prop hub stays fixed); faster motion writes `transforms_motion()` directly. The single-pose verbs replicate across samples, so a
  dynamic spec driven only by them renders pixel-identically to camera-only blur. All passes
  are enqueue-only: one fenced submit on Vulkan, one command buffer on Metal, stream-ordered
- **GPU-resident dynamic motion blur** (OptiX): a sim kernel writes the shutter pair into
  `transforms_pair()` (entry layout `[2][num_overlays * max_instances][12]`, open then close)
  and the camera pair in place; `expand_motion_transforms_launch` then expands the samples into
  `transforms_motion()` and the close state into `transforms()` on the render stream
  (`transforms_generic.h` is the single interpolant shared by the host verb and the kernel).
  The steady-state step becomes kernel → expand → update → render with no host data path —
  the host's per-frame role is enqueues only (kernel launches, `optixAccelBuild` enqueues, and
  OWL's constant 48-byte launch-param upload per launch). Per overlay, use either the
  `set_transform*` host verbs or the pair path, not both. On generic/Vulkan/Metal the same verb
  expands on the CPU into the host-resident tensors (residency is a backend property; Vulkan's
  and Metal's per-frame instance-descriptor composition is CPU-side by architecture). End-to-end example:
  `src/rendering/raytracing/example/drone_device.cu`.

## Lifecycle

Standalone renderer (examples, tests, single scene):

```cpp
Renderer<SPEC> renderer;
rlt::malloc(device, renderer);
Scene scene;                       // load() asserts the scene is empty; add() composes
rlt::load(device, scene, "scene.glb");
rlt::init(device, renderer, scene);
```

Many renderers over (possibly repeated) scenes — the multi-scene training case:

```cpp
AssetLibrary<SPEC> library;        // one backend context, one build per unique scene
rlt::malloc(device, library);
for(TI s = 0; s < N; s++){
    rlt::malloc(device, renderers[s], library);
    TI scene_id = rlt::init(device, renderers[s], library, scene_paths[s]);  // content-hash dedup
    // scene_id keys caller-side per-scene data; identical files share one host scene,
    // one device build (BLAS/textures/vertex data), and return the same id
}
```

The library owns the host scenes and device builds; renderers own only their cameras, outputs,
ray gens, and launch params. Renderer storage, looping, and environment metadata (e.g. procthor
indoor-position scoring) are caller-side. Ownership is symmetric: `free` every renderer, then
`free` the library.

## Steady-state step (device-resident)

```cpp
// producers on device.stream write the input tensors in place
extract_cameras_kernel<<<..., device.stream>>>(..., rlt::data(rlt::cameras(device, renderer)));
rlt::update_launch(device, renderer);   // overlays: fill kernel + TLAS rebuilds
rlt::render_launch(device, renderer);   // waits device.stream internally (CUDA device arg)
rlt::probe_launch(device, renderer);
// consumers read rlt::data(rlt::observation(device, renderer)) etc. in place
```

## Backends

| backend | device code | notes |
|---|---|---|
| OptiX | `backends/optix` (PTX via OWL) | production training path; shared `AssetLibrary` context |
| Vulkan | `backends/vulkan` (GLSL → SPIR-V) | `VK_KHR_ray_query` compute; headless; runs on lavapipe for CI; genuinely-async `update` |
| generic | `backends/generic` | freestanding CPU reference and determinism oracle |
| Metal | `backends/metal` (MSL) | macOS; **not buildable on the Linux dev machine — changes are pattern-exact and must be validated on macOS before release** |
| WebGPU | `backends/webgpu` (WGSL) | standard webgpu.h only (browser/WASM-ready): host-built BVH traversed in WGSL compute (binned SAH, ordered near-first; `RL_TOOLS_WEBGPU_BVH=median` = generic baseline builder), leaf-packed triangle stream, content-deduplicated texture pool sampled in-shader; wgpu-native runtime (hash-pinned prebuilt), headless, runs on lavapipe |

Backend selection: `-DRL_TOOLS_RENDERING_RAYTRACING_BACKEND=AUTO|OPTIX|METAL|VULKAN|WEBGPU|GENERIC`
(mux in `operations_cpu_mux.h`). Code that needs a non-default backend passes an explicit
`backends::{Generic,Optix,Metal,Vulkan,Webgpu}` as the renderer's second template argument and
includes that backend's operations header directly. The cross-backend parity suite is
`tests/src/rendering/raytracing/generic/scene.cpp`, compiled once pinned to generic and once
against the active backend; golden image/probe comparisons live next to it.

Each successful renderer allocation writes one plain-text line to stderr:
`#rl_tools::rendering::raytracing: backend=<name>`. This makes the resolved runtime
backend visible to both native applications and language bindings without maintaining
announcement state; `backends::name<BACKEND>()` exposes the same lowercase name to code.

Golden candidates are generated by `test_rendering_raytracing_generate_golden_generic` and
`test_rendering_raytracing_generate_golden_<active-backend>`. They default to
`tests/data/rendering_raytracing_golden/backend/<backend>`; pass `--output-dir` explicitly when
promoting a selected backend's output to the canonical
`tests/data/rendering_raytracing_golden/procthor_static_scene` directory. Generator targets are excluded from the default build and are not registered with
CTest.

Python (hypert) consumers JIT against these headers out of tree: `data()` on the accessor
tensors yields stable device pointers suitable for dlpack/`__cuda_array_interface__` wrapping,
and the `_launch`/`_sync` verb split maps onto phase-based scheduling.
