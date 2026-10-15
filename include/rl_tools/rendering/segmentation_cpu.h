#include "../version.h"
#include "../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_SEGMENTATION_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_SEGMENTATION_CPU_H

#include "scene.h"
#include "segmentation.h"
#include <stdexcept>
#include <type_traits>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename CONTAINER>
    auto segmentation_objects(DEVICE&, CONTAINER& container){
        using Object = std::conditional_t<std::is_const_v<CONTAINER>, const rendering::Object, rendering::Object>;
        std::vector<Object*> objects;
        if constexpr(std::is_same_v<std::remove_const_t<CONTAINER>, rendering::Object>){
            objects.push_back(&container);
        }
        else if constexpr(std::is_same_v<std::remove_const_t<CONTAINER>, rendering::AssetPool>){
            for(auto& assembly : container.assemblies){
                for(auto& object : assembly.objects){
                    objects.push_back(&object);
                }
            }
        }
        else{
            for(auto& object : container.objects){
                objects.push_back(&object);
            }
        }
        return objects;
    }

    template <typename DEVICE>
    void validate_segmentation_taxonomy(DEVICE&, const rendering::SegmentationTaxonomy& taxonomy){
        if(taxonomy.name.empty() || !taxonomy.classes.count(0)){
            throw std::invalid_argument("semantic segmentation: taxonomy must have a name and class 0");
        }
        for(const auto& entry : taxonomy.classes){
            if(entry.first >= rendering::segmentation::RESERVED || entry.second.empty()){
                throw std::invalid_argument("semantic segmentation: invalid taxonomy class");
            }
        }
    }

    template <typename DEVICE, typename OBJECT>
    std::shared_ptr<const rendering::SegmentationTaxonomy> segmentation_taxonomy(DEVICE& device, const std::vector<OBJECT*>& objects){
        std::shared_ptr<const rendering::SegmentationTaxonomy> taxonomy;
        for(const auto* object : objects){
            const auto& candidate = object->segmentation_taxonomy;
            if(candidate){
                if(candidate != taxonomy){
                    validate_segmentation_taxonomy(device, *candidate);
                    if(taxonomy && (candidate->name != taxonomy->name || candidate->classes != taxonomy->classes)){
                        throw std::invalid_argument("semantic segmentation: incompatible taxonomies; remap classes before composition");
                    }
                }
                taxonomy = candidate;
            }
        }
        return taxonomy;
    }

    template <typename DEVICE, typename CONTAINER>
    std::shared_ptr<const rendering::SegmentationTaxonomy> segmentation_taxonomy(DEVICE& device, const CONTAINER& container){
        return segmentation_taxonomy(device, segmentation_objects(device, container));
    }

    template <typename DEVICE>
    void validate_segmentation_class(DEVICE&, uint32_t id, const std::shared_ptr<const rendering::SegmentationTaxonomy>& taxonomy){
        if(id >= rendering::segmentation::RESERVED || (taxonomy && !taxonomy->classes.count(id))){
            throw std::invalid_argument("semantic segmentation: class ID is reserved or absent from the taxonomy");
        }
    }

    template <typename DEVICE, typename OBJECT>
    void validate_semantic_segmentation(DEVICE& device, const std::vector<OBJECT*>& objects){
        const auto taxonomy = segmentation_taxonomy(device, objects);
        for(const auto* object : objects){
            if(!object->segmentation_class_valid){
                throw std::invalid_argument("semantic segmentation: welded classes; load with SceneRoots (Python: preserve_instances=True) or explicitly assign one class");
            }
            validate_segmentation_class(device, object->segmentation_class, taxonomy);
        }
    }

    template <typename DEVICE>
    void validate_semantic_segmentation(DEVICE& device, const rendering::Scene& scene, const rendering::AssetPool& pool){
        auto objects = segmentation_objects(device, scene);
        const auto pool_objects = segmentation_objects(device, pool);
        objects.insert(objects.end(), pool_objects.begin(), pool_objects.end());
        validate_semantic_segmentation(device, objects);
    }

    template <typename DEVICE>
    void assign_segmentation_class(DEVICE& device, rendering::Object& object, uint32_t id, std::shared_ptr<const rendering::SegmentationTaxonomy> taxonomy){
        if(taxonomy){
            validate_segmentation_taxonomy(device, *taxonomy);
        }
        validate_segmentation_class(device, id, taxonomy);
        object.segmentation_class = id;
        object.segmentation_taxonomy = std::move(taxonomy);
        object.segmentation_class_valid = true;
    }

    template <typename DEVICE>
    void assign_segmentation_class(DEVICE& device, rendering::Object& object, uint32_t id){
        assign_segmentation_class(device, object, id, object.segmentation_taxonomy);
    }

    template <typename DEVICE, typename CONTAINER>
    void assign_segmentation_class(DEVICE& device, CONTAINER& container, size_t object_index, uint32_t id){
        assign_segmentation_class(device, container.objects.at(object_index), id, segmentation_taxonomy(device, container));
    }

    template <typename DEVICE>
    void assign_segmentation_class(DEVICE& device, rendering::AssetPool& pool, size_t asset_index, size_t object_index, uint32_t id){
        assign_segmentation_class(device, pool.assemblies.at(asset_index).objects.at(object_index), id, segmentation_taxonomy(device, pool));
    }

    // Complete remaps are transactional; assigning one class is the explicit way to replace
    // boundaries lost by welding. A null target taxonomy selects unnamed numeric classes.
    template <typename DEVICE, typename CONTAINER>
    void remap_segmentation_classes(DEVICE& device, CONTAINER& container, const std::map<uint32_t, uint32_t>& mapping, std::shared_ptr<const rendering::SegmentationTaxonomy> taxonomy){
        auto objects = segmentation_objects(device, container);
        validate_semantic_segmentation(device, objects);
        if(taxonomy){
            validate_segmentation_taxonomy(device, *taxonomy);
        }
        for(const auto* object : objects){
            const auto entry = mapping.find(object->segmentation_class);
            if(entry == mapping.end()){
                throw std::invalid_argument("semantic segmentation: incomplete class remapping");
            }
            validate_segmentation_class(device, entry->second, taxonomy);
        }
        for(auto* object : objects){
            object->segmentation_class = mapping.at(object->segmentation_class);
            object->segmentation_taxonomy = taxonomy;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
