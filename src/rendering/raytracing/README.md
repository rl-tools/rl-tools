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
  (motion blur), `transforms()` (overlays: one geometry-to-world 3x4 per slot),
  `transforms_motion()` (dynamic motion blur: per-sample overlay transforms, sample-major so
  slab `s` shares the `transforms()` layout),
  `cameras_previous()`/`transforms_previous()` (flow: the previous instant),
  `frame_buffer()`, `depth_buffer()`, `segmentation_buffer()`, `normals_buffer()`,
  `flow_buffer()`/`flow_status_buffer()`, `flow_forward_buffer()`/`flow_forward_status_buffer()`,
  `collision_results()`, `observation()`. Residency is a
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
- **Flow output.** With `OUTPUT_FLOW`, a single-sample pass writes the backward optical flow of the current frame to `flow_buffer()` (float2 per pixel, in pixels, sub-pixel exact) and a `FlowStatus` code per pixel to `flow_status_buffer()` (uint32). The renderer holds two instants of the published state: `cameras()`/`transforms()` are the current instant and `cameras_previous()`/`transforms_previous()` the previous one, each with its own resident overlay acceleration structures. Per pixel the pass traces the pixel-centre ray against the current instant, carries the hit point by its instance's rigid motion between the instants (composed per overlay slot at `update`; scene geometry and objects that did not move contribute pure camera flow), projects it through the previous camera and shoots a visibility ray from the previous camera towards the carried point against the previous instant. The status resolves in priority order: `MISS` (no hit; flow zero), `BEHIND` (carried point at or behind the other camera; flow zero), `OUT_OF_FRAME` (projects outside the other image; visibility not tested), `OCCLUDED` (a closer hit along the visibility ray within the relative margin `FLOW_VISIBILITY_EPSILON`), `VALID`. The flow value is the displacement to add to a pixel to reach its correspondence (`p_previous - p_current`) and is retained on `OUT_OF_FRAME` and `OCCLUDED`, so the status is the mask. `OUTPUT_FLOW_FORWARD` adds the mirrored pass on the previous camera grid (`flow_forward_buffer()`/`flow_forward_status_buffer()`, `p_current - p_previous`), which is what a bidirectional-consistency consumer needs. Host verbs: `spawn` initialises previous == current, the `set_transform*` verbs touch only the current instant, and `advance_flow` publishes the current instant as the previous one (cameras and poses; resident rows copy at once, rows the host has written but not yet published stage through their mirror at the next `update`). The frame-to-frame steady state is `advance_flow` → move/write → `update` → `render`; producers that keep last frame's state themselves write `cameras_previous()` and the `transforms_previous()` rows they own directly, under the same ownership contract as `transforms()`. Not modelled: attachment changes between the instants, non-rigid motion beyond per-part articulation, rolling shutter, and flow through motion-blur sample poses. Examples: `src/rendering/raytracing/example/flow.cpp` (one steady-state step to PNGs) and the `rendering_raytracing_interactive_flow` viewer (walk the scene with an orbiting cube; RGB on the left half, flow on the right half, Tab cycles backward flow, backward status, forward flow and forward status).
- **Determinism is a contract.** Fixed seed ⇒ identical results, no atomics. The global
  instance-id layout (scene instances `[0,S)`, overlay `o` slot `s` at `S + o*MAX + s`) is
  cross-backend API surface consumed by segmentation.
- **Dynamic motion blur** (`ENABLE_DYNAMIC_MOTION_BLUR`, requires motion blur + overlays):
  `render_launch` internally runs `MOTION_BLUR_SAMPLES` sequential passes, rebuilding the
  overlay acceleration structures from the per-sample `transforms_motion()` slabs between
  passes (camera lerped at the same `(i+0.5)/N` shutter times), accumulating linear radiance
  per pixel (each thread owns its pixel — no atomics), then restores the shutter-close overlay
  state (segmentation and probes stay single-sample at shutter close, ids unchanged) and
  resolves with the standard transfer curve + quantization. `set_transform_pair` expands a
  rigid shutter-open/close pose into the samples by constant-twist screw interpolation
  (`rendering::screw_interpolate`: exact for constant-velocity rigid motion below 180° per
  shutter — an off-origin rotation axis such as a prop hub stays fixed); `set_motion_transforms`
  takes per-part samples and the shutter-close poses directly. The single-pose verbs replicate
  across samples, so a dynamic spec driven only by them renders pixel-identically to camera-only
  blur. All passes are enqueue-only: one fenced submit on Vulkan, one command buffer on Metal,
  stream-ordered
- **Articulation is a producer** (`rendering/rig/`): a `rig::Model` is a fixed-capacity,
  plain-value rigid-link tree (links declared parent before child; fixed, revolute, prismatic
  and spherical joints; one `origin` per link and one `offset` per part) that `evaluate`s to
  geometry-to-world poses in part order and `sample`s a shutter — coordinates interpolate before
  the kinematics (a revolute joint keeps every revolution), the base follows the screw. The host
  verbs are `set_transforms(device, renderer, overlay, placement, poses)` and
  `set_motion_transforms(device, renderer, overlay, placement, samples, close)`; a device kernel
  calls `sample` with the renderer's `transforms_motion()`/`transforms()` rows as its outputs.
  `init(device, model, assembly)` builds the all-fixed rig of an asset's rest frames (nested
  GLB parts recorded by the loader become joint origins), and the hyperdrone rotorcraft is one
  such model built by its body/prop_* convention.
- **GPU-resident dynamic motion blur** (OptiX): a sim kernel writes the per-sample world poses
  into `transforms_motion()`, the shutter-close poses into `transforms()` and the camera pair in
  place, all on the render stream; the steady-state step is kernel → update → render with no
  host data path — the host's per-frame role is enqueues only (kernel launches,
  `optixAccelBuild` enqueues, and OWL's constant 48-byte launch-param upload per launch). Host
  verbs and producers may share a renderer: a verb writes the host mirror and marks its rows
  pending, `update` publishes exactly the pending rows (compacted into one scatter on OptiX),
  and a producer owns every row the host has not marked. On generic/Vulkan/Metal the publish is
  a memcpy into the host-resident tensors (residency is a backend property; Vulkan's and Metal's
  per-frame instance-descriptor composition is CPU-side by architecture). End-to-end example:
  `src/rendering/raytracing/example/drone_device.cu`.

## Lifecycle

Standalone renderer (examples, tests, single scene). Scenes come from the datasets layer
(`rendering/datasets/`, e.g. the glb loader) as a `rendering::Bundle` — Scene plus loader-filled
`SceneMetadata` (bounds, max ray length, SHA-1 content hash); the renderer never touches files:

```cpp
Renderer<SPEC> renderer;
rlt::malloc(device, renderer);
rendering::Bundle<T> bundle;       // load() asserts the bundle is empty; add() composes
rlt::load(device, bundle, "scene.glb");   // or a "conta:HASH" reference
rlt::init(device, renderer, bundle);
```

Many renderers over (possibly repeated) scenes — the multi-scene training case:

```cpp
AssetLibrary<SPEC> library;        // one backend context, one build per unique scene
rlt::malloc(device, library);
for(TI s = 0; s < N; s++){
    rlt::malloc(device, renderers[s], library);
    rendering::Bundle<T> bundle;
    rlt::load<typename SPEC::SHADING, SPEC::HAS_RGB>(device, bundle, references[s]);
    TI scene_id = rlt::insert(device, library, bundle);   // content-hash dedup; library takes the scene
    rlt::init(device, renderers[s], library, scene_id);
    // scene_id keys caller-side per-scene data; identical content shares one host scene,
    // one device build (BLAS/textures/vertex data), and returns the same id
}
```

The library owns the host scenes and device builds; renderers own only their cameras, outputs,
ray gens, and launch params. Renderer storage, looping, and environment metadata (e.g. procthor
indoor-position scoring) are caller-side. Ownership is symmetric: `free` every renderer, then
`free` the library.

## Steady-state step (device-resident)

```cpp
rlt::advance_flow_launch(device, renderer);   // flow only: cameras/transforms -> *_previous (device-to-device on OptiX)
// producers on device.stream write the input tensors in place
extract_cameras_kernel<<<..., device.stream>>>(..., rlt::data(rlt::cameras(device, renderer)));
rlt::update_launch(device, renderer);   // overlays: fill kernel + TLAS rebuilds (both instants with flow)
rlt::render_launch(device, renderer);   // waits device.stream internally (CUDA device arg)
rlt::probe_launch(device, renderer);
// consumers read rlt::data(rlt::observation(device, renderer)), rlt::data(rlt::flow_buffer(device, renderer)) etc. in place
```

The host-verb form of the same step (`advance_flow` → `set_transform`/camera upload → `update` → `render`) is `src/rendering/raytracing/example/flow.cpp`.

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

Goldens are generated by `test_rendering_raytracing_generate_golden_<active-backend>` straight into
`tests/data/rendering/raytracing/goldens/procthor-train-1` (`--output-dir` renders a candidate elsewhere)
and by the OptiX-only overlay generator; generator targets are excluded from the default build and are not registered with
CTest. Each corpus carries a `manifest.json` with a `golden_version` pinned in
`tests/src/rendering/raytracing/golden_manifest.h`; the comparators hold the reference backend
(OptiX) to bit-exact reproduction and every other backend to measured cross-backend floors, and
`tests/src/rendering/raytracing/GOLDEN_LEDGER.md` lists every corpus version with its date,
bit-exact commit and the reason for regenerating.

Python (hypert) consumers JIT against these headers out of tree: `data()` on the accessor
tensors yields stable device pointers suitable for dlpack/`__cuda_array_interface__` wrapping,
and the `_launch`/`_sync` verb split maps onto phase-based scheduling.

Browser demo: `src/rendering/raytracing/example/web/build.sh` compiles the WebGPU backend to
WASM with an activated emsdk (`--use-port=emdawnwebgpu`, ASYNCIFY, `-fexceptions` — assimp
relies on internal try/catch) and the page in `static/raytracing/` runs the drone
motion-blur demo (`drone.cpp` semantics) live on canvases; assets are fetched by sha1 from the
conta-data store (`?assets=local` serves `static/raytracing/assets/`, populated by
`link_local_assets.sh`). The blocking-wait seams and adapter acquisition switch on
`__EMSCRIPTEN__` in `backends/webgpu/operations_cpu.h`; browsers cap
`maxStorageBuffersPerShaderStage` at the spec default of 8, which is why the bind layout packs
per-frame inputs and outputs into the `FRAME_INPUTS`/`OUTPUTS` arenas.
