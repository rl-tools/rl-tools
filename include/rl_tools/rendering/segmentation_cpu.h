#include "../version.h"
#include "../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_SEGMENTATION_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_SEGMENTATION_CPU_H

#include "scene.h"
#include <stdexcept>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE>
    std::shared_ptr<const rendering::SegmentationTaxonomy> segmentation_taxonomy(DEVICE&, const std::vector<const rendering::Object*>& objects){
        std::shared_ptr<const rendering::SegmentationTaxonomy> taxonomy;
        for(const auto* object : objects){
            const auto& candidate = object->segmentation_taxonomy;
            if(candidate){
                if(taxonomy && candidate != taxonomy && (candidate->name != taxonomy->name || candidate->classes != taxonomy->classes)){
                    throw std::invalid_argument("semantic segmentation: incompatible taxonomies; remap classes before composition");
                }
                taxonomy = candidate;
            }
        }
        return taxonomy;
    }

    template <typename DEVICE, typename CONTAINER>
    std::shared_ptr<const rendering::SegmentationTaxonomy> segmentation_taxonomy(DEVICE& device, const CONTAINER& container){
        std::vector<const rendering::Object*> objects;
        for(const auto& object : container.objects){
            objects.push_back(&object);
        }
        return segmentation_taxonomy(device, objects);
    }

    template <typename DEVICE>
    void validate_semantic_segmentation(DEVICE& device, const std::vector<const rendering::Object*>& objects){
        const auto taxonomy = segmentation_taxonomy(device, objects);
        for(const auto* object : objects){
            if(!object->segmentation_class_valid){
                throw std::invalid_argument("semantic segmentation: welded classes; load with SceneRoots (Python: preserve_instances=True)");
            }
            if(object->segmentation_class == 0xFFFFFFFFu || (taxonomy && !taxonomy->classes.count(object->segmentation_class))){
                throw std::invalid_argument("semantic segmentation: class ID is reserved or absent from the taxonomy");
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
