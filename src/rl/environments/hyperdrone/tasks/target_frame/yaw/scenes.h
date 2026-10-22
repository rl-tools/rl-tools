#pragma once

#include <rl_tools/hyperlux/datasets/procthor/operations_cpu.h>

namespace rl_tools::rl::environments::hyperdrone::tasks::target_frame::yaw{
    struct Scenes: hyperlux::datasets::procthor::GLB{};
}

namespace rl_tools{
    template<typename DEVICE>
    void enumerate(DEVICE& device, const rl::environments::hyperdrone::tasks::target_frame::yaw::Scenes& dataset, rl::environments::hyperdrone::tasks::target_frame::yaw::Scenes::Corpus& corpus){
        enumerate(device, static_cast<const hyperlux::datasets::procthor::GLB&>(dataset), corpus);
    }

    template<typename SHADING = hyperlux::High, bool HAS_RGB = true, typename DEVICE, typename T, typename MODE = mode::Default<>>
    bool load(DEVICE& device, const rl::environments::hyperdrone::tasks::target_frame::yaw::Scenes&, const rl::environments::hyperdrone::tasks::target_frame::yaw::Scenes::Corpus& corpus, size_t index, hyperlux::Bundle<T>& bundle, const Mode<MODE>& = Mode<MODE>{}){
        using hyperlux::datasets::resolve_reference;
        using hyperlux::datasets::compute_bounds;
        using hyperlux::datasets::content_hash;
        static_assert(!mode::is<MODE, mode::hyperlux::SceneRoots>, "Yaw scenes use welded geometry");
        utils::assert_exit(device, index < corpus.references.size(), "yaw::Scenes: corpus index out of range");
        utils::assert_exit(device, bundle.scene.objects.empty() && bundle.scene.instances.empty() && bundle.scene.lights.empty(), "yaw::Scenes: bundle must be empty");
        std::string path;
        if(!resolve_reference(device, corpus.references[index], path)){
            return false;
        }
        if(!add<SHADING, HAS_RGB>(device, bundle, path)){
            return false;
        }
        constexpr float inv_sqrt2 = 0.70710678f;
        const hyperlux::SceneLight fill[] = {
            {0, {0,0,0}, {-inv_sqrt2, 0.f, inv_sqrt2}, {0.4f, 0.4f, 0.4f}, 0,0,0, 0,0},
            {0, {0,0,0}, {0.f, -inv_sqrt2, inv_sqrt2}, {0.3f, 0.3f, 0.3f}, 0,0,0, 0,0},
            {0, {0,0,0}, {0.f, inv_sqrt2, inv_sqrt2}, {0.2f, 0.2f, 0.2f}, 0,0,0, 0,0}
        };
        bundle.scene.lights.insert(bundle.scene.lights.begin(), fill, fill + 3);
        for(auto& light: bundle.scene.lights){
            light.radius = 0;
        }
        bundle.scene.environment.mode = hyperlux::Scene::Environment::Mode::SOLID;
        for(unsigned int channel = 0; channel < 3; channel++){
            bundle.scene.environment.ambient[channel] = 0.5f;
            bundle.scene.environment.horizon[channel] = 0.0f;
        }
        for(auto& object: bundle.scene.objects){
            for(auto& mesh: object.meshes){
                mesh.unlit = false;
            }
        }
        compute_bounds(device, bundle);
        utils::assert_exit(device, content_hash(device, path, bundle.metadata.content_hash), "yaw::Scenes: failed to hash scene file");
        return true;
    }
}
