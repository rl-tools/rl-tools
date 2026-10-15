# Raytracing Renderer

Batch raytracing for visual RL: N cameras over one static scene plus per-camera dynamic
overlays, rendered by one of five backends (OptiX, Vulkan, Metal, WebGPU, generic CPU) behind a single
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
  and `output(device, renderer, descriptor)`. Residency is a
  backend property: CUDA device memory on OptiX, host on generic and WebGPU (WebGPU settles
  outputs from staging at the sync boundary), shared/mapped on Metal. Vulkan uses mapped device-local outputs when they fit; otherwise, device-local outputs are copied into stable host tensors at synchronization and output-copy boundaries.
- **Data moves via tensor copies and kernels.** No raw backend handles or `cudaStream_t` appear in
  public signatures. Device producers (extraction kernels) write the input tensors in place;
  device consumers read the output tensors in place; host readers/writers cross the residency
  boundary with a regular tensor copy against the renderer's memory-domain device:
  `rlt::copy(renderer.device, device, rlt::output(device, renderer, outputs::RGB<>{}), host_tensor)`
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
- **Outputs determine storage and work.** `Config::OUTPUTS` is a compile-time list of descriptors;
  `Specification::PLAN` derives producers, history, and representations. The generic backend
  dispatches encoders through C++ overloads; GPU backends lower the same plan into their shader
  constants and bindings. Raw and colored forms share the quantity calculation, and requesting
  only a colored form allocates no raw image. AA and motion blur apply only to RGB.
  OptiX and Vulkan share the shutter-close primary intersection across depth, segmentation, normals, and backward flow; forward flow traces the previous instant separately.
- **Flow has two independent directions.** Backward flow is `p_previous - p_current` on the
  current grid; forward flow is `p_current - p_previous` on the previous grid. Either direction
  requires previous cameras and overlay poses, but neither requires the other output. A flow
  vector alone omits the visibility ray; status or colored flow adds it. Status priority is
  `MISS`, `BEHIND`, `OUT_OF_FRAME`, `OCCLUDED`, `VALID`; vectors are zero for the first two
  and retain their geometric displacement for the others. Visibility uses the relative margin
  `FLOW_VISIBILITY_EPSILON`. `advance_flow` → move/write → `update` → `render` publishes the
  two instants; explicit producers can instead write `cameras_previous()` and
  `transforms_previous()`. Spawn initializes previous equal to current. Attachment changes,
  non-rigid motion, rolling shutter, and flow through motion-blur samples are not modeled.
- **Determinism is a contract.** Fixed seed ⇒ identical results, no atomics. The global
  instance-id layout (scene instances `[0,S)`, overlay `o` slot `s` at `S + o*MAX + s`) is
  cross-backend API surface consumed by segmentation.
- **Dynamic motion blur** (`ENABLE_DYNAMIC_MOTION_BLUR`, requires motion blur + overlays):
  `render_launch` internally runs `MOTION_BLUR_SAMPLES` sequential passes, rebuilding the
  overlay acceleration structures from the per-sample `transforms_motion()` slabs between
  passes (camera lerped at the same `(i+0.5)/N` shutter times), accumulating linear radiance
  per pixel (each thread owns its pixel — no atomics), then restores the shutter-close overlay
  state (depth, segmentation and probes stay single-sample at shutter close, ids unchanged) and
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
  such model built by its body/prop_* convention, its props matched to the dynamics rotors.
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

## Declaring outputs

```cpp
namespace outputs = rl_tools::rendering::raytracing::outputs;
struct Config: rl_tools::rendering::raytracing::config::Default<float, unsigned int> {
    static constexpr unsigned int CAM_WIDTH = 128, CAM_HEIGHT = 128, NUM_CAMERAS = 1;
    static constexpr unsigned int NUM_PROBES = 0;
    using OUTPUTS = outputs::List<
        outputs::RGB<outputs::format::Float32>,
        outputs::Depth<outputs::format::RGBA8>,
        outputs::Flow<outputs::direction::Forward>>;
};
// After malloc/init/render:
auto& depth_preview = output(device, renderer, outputs::Depth<outputs::format::RGBA8>{});
```

An undeclared descriptor is a compile error. Each descriptor selects a tensor shape and scalar
without a channel-specific accessor. List order is arbitrary; one quantity/representation slot
may appear once. `outputs::Concat` and `outputs::If` compose lists. The default requests only
packed RGB; probes must be explicit and require `NUM_PROBES > 0`.

`outputs::Cameras<SELECTION, DESCRIPTOR>` restricts an entry to the cameras its selection picks
(`cameras::Every<PERIOD, OFFSETS...>`, or any type with a `constexpr selected(camera)`); the
descriptor alone stays the accessor key. `N` below is the number of cameras an entry covers, one
row each in camera order. Each shared per-pixel pass (primary ray, surface hit, flow visibility)
runs on the union of the cameras whose outputs need it.

| Descriptor | Raw representation and shape | RGBA8 representation |
|---|---|---|
| `RGB<FORMAT>` | Float32 `[N,H,W,3]`, shading transfer curve before quantization | Default; packed uint32 `[N,H,W]` |
| `Depth<FORMAT, PARAMETERS>` | Default Float32 `[N,H,W]`, camera-forward metres; miss = +infinity | Grayscale from 0 to `PARAMETERS::SCALE` metres (default 10); miss = white |
| `Segmentation<FORMAT, LABELS>` | Default UInt32 `[N,H,W]`; `labels::Instance` or `labels::Semantic`; miss = 0xffffffff | Pinned ID palette; miss = black |
| `Normals<FORMAT>` | Default Float32 `[N,H,W,3]`, world FLU unit normal against the ray; miss = zero | Pinned component mapping; miss = gray |
| `Flow<DIRECTION, FORMAT, PARAMETERS>` | Default backward Float32 `[N,H,W,2]`, pixel displacement | Hue = direction, saturation = magnitude / `PARAMETERS::SCALE` (default 20 pixels); invalid pixels use status colors |
| `FlowStatus<DIRECTION, FORMAT>` | Default backward UInt32 `[N,H,W]` | Pinned status palette |
| `Probes` | `CollisionResult` `[N,NUM_PROBES]`, distance and hit flag | — |

Depth, labels, normals, and flow use pixel centers at shutter close independently of RGB
sampling. Instance and semantic labels may coexist. `DepthParameters` and `FlowParameters`
provide display-scale defaults; the full configured descriptor, including its parameter type,
is the accessor key. `save_image(device, renderer, descriptor, filename)` saves any declared
RGBA8 image. Existing documented raw export helpers retain their image normalization and
binary formats for the examples and golden tooling that consume them.

| Consumer | Constraint on the design |
|---|---|
| HyperDrone CPU/CUDA training, rigs, tasks, spectators | Float RGB and image history only when requested; explicit probes; backend-resident tensor copies |
| Browser display | Packed display outputs directly from the rendering backend; no CPU conversion or raw intermediate; auxiliary outputs for the onboard mount only |
| Python Render/Env | Existing Python names, shapes, and dtypes map to explicit descriptor lists |
| Native interactive viewer, dataset annotation, examples, benchmarks, goldens | Stable raw values, instance IDs, export formats, and deterministic workloads |
| Generic, Metal, OptiX, Vulkan, WebGPU | Shared plan and accessor; native allocation, shader lowering, and synchronization remain backend operations |

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
// consumers read data(output(device, renderer, descriptor)) in place
```

The host-verb form of the same step (`advance_flow` → `set_transform`/camera upload → `update` → `render`) is `src/rendering/raytracing/example/flow.cpp`.

## Backends

| backend | device code | notes |
|---|---|---|
| OptiX | `backends/optix` (PTX via OWL) | production training path; shared `AssetLibrary` context |
| Vulkan | `backends/vulkan` (GLSL → SPIR-V) | `VK_KHR_ray_query` compute; headless; runs on lavapipe for CI; genuinely-async `update` |
| generic | `backends/generic` | freestanding CPU backend; built as a WASI module it is the golden reference (`tests/src/rendering/raytracing/wasm`) |
| Metal | `backends/metal` (MSL) | macOS; **not buildable on the Linux dev machine — changes are pattern-exact and must be validated on macOS before release** |
| WebGPU | `backends/webgpu` (WGSL) | standard webgpu.h only (browser/WASM-ready): host-built BVH traversed in WGSL compute (binned SAH, ordered near-first; `RL_TOOLS_WEBGPU_BVH=median` = generic baseline builder), leaf-packed triangle stream, content-deduplicated texture pool sampled in-shader; wgpu-native runtime (hash-pinned prebuilt), headless, runs on lavapipe |

Backend selection: `-DRL_TOOLS_RENDERING_RAYTRACING_BACKEND=AUTO|OPTIX|METAL|VULKAN|WEBGPU|GENERIC`
(mux in `operations_cpu_mux.h`). Code that needs a non-default backend passes an explicit
`backends::{Generic,Optix,Metal,Vulkan,Webgpu}` as the renderer's second template argument and
includes that backend's operations header directly. The cross-backend parity suite is
`tests/src/rendering/raytracing/generic/scene.cpp`, compiled once pinned to generic and once
against the active backend; golden image/probe comparisons live next to it.

At the default verbosity, each successful renderer allocation writes one plain-text line to stderr:
`#rl_tools::rendering::raytracing: backend=<name>`. This makes the resolved runtime
backend visible to both native applications and language bindings without maintaining
announcement state; `backends::name<BACKEND>()` exposes the same lowercase name to code.
Rendering and dataset messages pass a level, `log(device, device.logger, level, ...)`, which
`device.logger.verbosity` (`error|warning|info|debug`, default `info`) filters to stderr, with
the backend line at `info`; `log` without a level remains unconditional stdout.

Goldens are generated by the wasm reference module (`tests/src/rendering/raytracing/wasm`, `render.cmake` with `-DOUTPUT_DIR`) straight into
`tests/data/rendering/raytracing/goldens/procthor-train-1`; the native `test_rendering_raytracing_generate_golden_<backend>` targets render candidates (`--output-dir`, `--dump-dir`),
the OptiX-only overlay generator its own corpus; generator targets are excluded from the default build and are not registered with
CTest. Each corpus carries a `manifest.json` with a `golden_version` pinned in
`tests/src/rendering/raytracing/golden_manifest.h`; the comparators hold the reference backend
(wasm) to bit-exact reproduction and every other backend to measured cross-backend floors, and
`tests/src/rendering/raytracing/GOLDEN_LEDGER.md` lists every corpus version with its date,
bit-exact commit and the reason for regenerating.

Python (hypert) consumers JIT against these headers out of tree: `data()` on the accessor
tensors yields stable device pointers suitable for dlpack/`__cuda_array_interface__` wrapping,
and the `_launch`/`_sync` verb split maps onto phase-based scheduling.

Browser demo: `src/rl/environments/hyperdrone/web/build.sh` compiles the WebGPU backend to
WASM with an activated emsdk (`--use-port=emdawnwebgpu`, ASYNCIFY, `-fexceptions` — assimp
relies on internal try/catch) and the page in `static/hyperdrone/` uses HyperDrone's
MultiEnvironment for interactive flight with two mounted RGB cameras and onboard depth,
semantic segmentation, normals, and optical flow. RAPTOR runs at 100 Hz: WASD moves horizontally,
E ascends, and Q descends through current-relative position and velocity targets; releasing
the keys holds position and altitude. Combined input is normalized to the configured speed. Click a
camera to capture or release the mouse, horizontal mouse motion controls yaw, and RAPTOR
holds position without user input; hidden tabs pause. The scene selector uses the shared
`procthor::Train200` catalog. `static/hyperdrone/registry.json` pins the rigged airframes and
RAPTOR checkpoint; `pose.json` pins the default scene and its starting position in FLU meters
and yaw in degrees (other scenes use MultiEnvironment's free-space annotations).
The browser conta `load` resolver reuses verified gzip assets from persistent browser Cache Storage;
on a miss it requests `blob/<sha1>.gz`, then the format-1 conta providers, preferring advertised gzip
variants. Scene, airframe, and RAPTOR downloads share this cache; raw-only assets are gzipped before
storage, corrupt entries are discarded, and storage failures leave downloading available.
Run `static/hyperdrone/download_blob.sh` to clone the separate asset repository into
`static/hyperdrone/blob/`; existing checkouts are left untouched. The build script writes its module
to `blob/build/`. Serve the page over HTTPS or localhost; `?scene=ProcTHOR-Train-0` selects a
scene, and `?frames=N` renders a deterministic finite sequence for verification.
The blocking-wait seams and adapter acquisition switch on
`__EMSCRIPTEN__` in `backends/webgpu/operations_cpu.h`; browsers cap
`maxStorageBuffersPerShaderStage` at the spec default of 8, which is why the bind layout packs
per-frame inputs and outputs into the `FRAME_INPUTS`/`OUTPUTS` arenas.
