#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_MUX_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_MUX_H

// backend contract: renderer.h maps the same macro to backends::Default (the Renderer/AssetLibrary
// template default), and the selected header provides, in namespace rl_tools for that tag:
//   types: backends::{RendererState, LibraryState, SceneState}<BACKEND, SPEC> specializations
//   verbs on Renderer<SPEC, BACKEND>: malloc, init(device, renderer, scene, pool, metadata), free,
//          synchronize, generate_probe_directions, {render, probe, update, advance_flow}_{launch, sync}
//   copy(...) with backends::Device<BACKEND> as source or destination
//   optional: device_name; Runtime malloc/free and renderer malloc for shared driver lifetime (Vulkan);
//             AssetLibrary malloc/free/init overloads for cross-renderer sharing (OptiX)
// It includes operations_cpu_common.h (backend-independent host verbs) first and
// operations_cpu_post.h last, which derives the blocking composites (render = render_launch +
// render_sync, ...), the library/Bundle init overloads and the non-sharing library fallbacks.
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL)
    #include "backends/metal/operations_cpu.h"
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
    #include "backends/optix/operations_cuda.h"
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_VULKAN)
    #include "backends/vulkan/operations_cpu.h"
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_WEBGPU)
    #include "backends/webgpu/operations_cpu.h"
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_GENERIC)
    #include "backends/generic/operations_cpu.h"
#else
    #error "RLtools raytracing: no backend configured (RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX, RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL, RL_TOOLS_RENDERING_RAYTRACING_BACKEND_VULKAN, RL_TOOLS_RENDERING_RAYTRACING_BACKEND_WEBGPU, or RL_TOOLS_RENDERING_RAYTRACING_BACKEND_GENERIC)"
#endif

#endif
