#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_POST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_POST_H

#include "renderer.h"

// backend-generic wrappers over each backend's 5-arg init(device, renderer, scene, pool,
// metadata): included at the BOTTOM of every backend operations header so the forwarded-to
// overloads are visible at the wrappers' definition context (the verbs live in plain rl_tools,
// which argument-dependent lookup at the point of instantiation never associates)
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::AssetLibrary<SPEC, BACKEND>& library, typename SPEC::TI scene_id){
        init(device, renderer, library.scenes[scene_id], library.pool, library.metadata[scene_id]);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND, typename T>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const rendering::Bundle<T>& bundle, const rendering::raytracing::AssetPool& pool){
        init(device, renderer, bundle.scene, pool, bundle.metadata);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND, typename T>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, const rendering::Bundle<T>& bundle){
        static const rendering::raytracing::AssetPool empty_pool{};
        init(device, renderer, bundle.scene, empty_pool, bundle.metadata);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
