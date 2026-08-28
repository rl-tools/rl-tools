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

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void render(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        render_launch(device, renderer);
        render_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void probe(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        probe_launch(device, renderer);
        probe_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        update_launch(device, renderer);
        update_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void expand_motion_transforms(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer){
        expand_motion_transforms_launch(device, renderer);
        expand_motion_transforms_sync(device, renderer);
    }

    // shared-asset-library fallbacks for backends without cross-renderer sharing: the library is
    // empty and every renderer builds its own copy — the API stays uniform. A backend with real
    // sharing (OptiX) declares its own overloads, which win by partial ordering.
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void malloc(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, BACKEND>& library){
        library.backend = new rendering::raytracing::backends::LibraryState<BACKEND, SPEC>{};
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void free(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, BACKEND>& library){
        for(auto* assets : library.assets){
            delete assets;
        }
        library.assets.clear();
        library.scenes.clear();
        library.metadata.clear();
        delete library.backend;
        library.backend = nullptr;
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, BACKEND>& renderer, rendering::raytracing::AssetLibrary<SPEC, BACKEND>& library){
        malloc(device, renderer);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
