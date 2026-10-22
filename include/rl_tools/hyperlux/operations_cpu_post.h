#include "../version.h"
#include "../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_OPERATIONS_CPU_POST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_OPERATIONS_CPU_POST_H

#include "hyperlux.h"

// backend-generic wrappers over each backend's 5-arg init(device, renderer, scene, pool,
// metadata): included at the BOTTOM of every backend operations header so the forwarded-to
// overloads are visible at the wrappers' definition context (the verbs live in plain rl_tools,
// which argument-dependent lookup at the point of instantiation never associates)
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename BACKEND>
    void malloc(DEVICE&, hyperlux::Runtime<BACKEND>&){
    }

    template <typename DEVICE, typename BACKEND>
    void free(DEVICE&, hyperlux::Runtime<BACKEND>&){
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void malloc(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, hyperlux::Runtime<BACKEND>&){
        malloc(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void init(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, hyperlux::AssetLibrary<SPEC, BACKEND>& library, typename SPEC::TI scene_id){
        init(device, renderer, library.scenes[scene_id], library.pool, library.metadata[scene_id]);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND, typename T>
    void init(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, const hyperlux::Bundle<T>& bundle, const hyperlux::AssetPool& pool){
        init(device, renderer, bundle.scene, pool, bundle.metadata);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND, typename T>
    void init(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, const hyperlux::Bundle<T>& bundle){
        static const hyperlux::AssetPool empty_pool{};
        init(device, renderer, bundle.scene, empty_pool, bundle.metadata);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void render(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer){
        render_launch(device, renderer);
        render_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void probe(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer){
        probe_launch(device, renderer);
        probe_sync(device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void update(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer){
        update_launch(device, renderer);
        update_sync(device, renderer);
    }

    // Snapshot current cameras and staged transforms as the previous instant before the next pose writes, whether or not an image was acquired.
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void advance_flow(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer){
        advance_flow_launch(device, renderer);
        advance_flow_sync(device, renderer);
    }

    // shared-asset-library fallbacks for backends without cross-renderer sharing: the library is
    // empty and every renderer builds its own copy — the API stays uniform. A backend with real
    // sharing (OptiX) declares its own overloads, which win by partial ordering.
    template <typename DEVICE, typename SPEC, typename BACKEND>
    void malloc(DEVICE& device, hyperlux::AssetLibrary<SPEC, BACKEND>& library){
        library.backend = new hyperlux::backends::LibraryState<BACKEND, SPEC>{};
    }

    template <typename DEVICE, typename SPEC, typename BACKEND>
    void free(DEVICE& device, hyperlux::AssetLibrary<SPEC, BACKEND>& library){
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
    void malloc(DEVICE& device, HyperLux<SPEC, BACKEND>& renderer, hyperlux::AssetLibrary<SPEC, BACKEND>& library){
        malloc(device, renderer);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
