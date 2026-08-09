#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_H

#include "renderer.h"

#include <cstddef>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE, typename T>
    void copy_to_renderer(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, const T* source, T* destination, std::size_t count){
        copy_to_renderer(device.render, device, renderer, source, destination, count);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE, typename T>
    void copy_from_renderer(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, const T* source, T* destination, std::size_t count){
        copy_from_renderer(device.render, device, renderer, source, destination, count);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        malloc(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void malloc(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, RENDER_DEVICE>& library){
        malloc(device.render, device, library);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void malloc(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, rendering::raytracing::AssetLibrary<SPEC, RENDER_DEVICE>& library){
        malloc(device.render, device, renderer, library);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void free(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        free(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void free(DEVICE& device, rendering::raytracing::AssetLibrary<SPEC, RENDER_DEVICE>& library){
        free(device.render, device, library);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, const rendering::raytracing::Scene& scene, const rendering::raytracing::AssetPool& pool){
        init(device.render, device, renderer, scene, pool);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void init(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, const rendering::raytracing::Scene& scene){
        init(device.render, device, renderer, scene);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    typename SPEC::TI init(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, rendering::raytracing::AssetLibrary<SPEC, RENDER_DEVICE>& library, const char* scene_path){
        return init(device.render, device, renderer, library, scene_path);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void update(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        update(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void update_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        update_launch(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void update_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        update_sync(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void generate_cameras(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, const typename SPEC::T center[3], typename SPEC::T radius, const typename SPEC::T up[3], typename SPEC::T fov){
        generate_cameras(device.render, device, renderer, center, radius, up, fov);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void generate_probe_directions(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        generate_probe_directions(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void render_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        render_launch(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void render_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        render_sync(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void render(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        render(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void probe_launch(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        probe_launch(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void probe_sync(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        probe_sync(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void probe(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        probe(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void synchronize(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        synchronize(device.render, device, renderer);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void save_segmentation_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, const char* filename){
        save_segmentation_image(device.render, device, renderer, filename);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void save_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, const char* filename){
        save_image(device.render, device, renderer, filename);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void save_depth_image(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, const char* filename){
        save_depth_image(device.render, device, renderer, filename);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void save_depth(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, const char* filename){
        save_depth(device.render, device, renderer, filename);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    void save_probes(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer, const char* filename){
        save_probes(device.render, device, renderer, filename);
    }

    template <typename DEVICE, typename SPEC, typename RENDER_DEVICE>
    auto stream(DEVICE& device, rendering::raytracing::Renderer<SPEC, RENDER_DEVICE>& renderer){
        return stream(device.render, device, renderer);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
