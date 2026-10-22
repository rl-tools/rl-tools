#pragma once
#include <rl_tools/hyperlux/outputs/outputs.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hyperlux {
    namespace camera_outputs = rl_tools::hyperlux::outputs;

    struct OutputDescriptor {
        std::string quantity;
        std::string format;
        std::string direction;
        std::string labels;
        double scale = 0;
    };

    struct OutputLayout {
        std::vector<size_t> shape;
        std::vector<int64_t> strides;
        std::vector<std::string> axes;
        std::string dtype;
        size_t offset = 0;
        size_t item_size = 4;

        size_t size() const {
            size_t count = 1;
            for(size_t dimension: shape){
                count *= dimension;
            }
            return count;
        }
    };

    struct OutputMetadata {
        OutputDescriptor descriptor;
        OutputLayout layout;
        std::vector<size_t> cameras;
        std::vector<std::pair<std::string, OutputLayout>> fields;
        OutputLayout packed;
        bool has_packed = false;
        int device_type = 1;
        int device_id = 0;
        size_t block_count = 1;
    };

    template <typename FORMAT>
    std::string output_format(){
        if constexpr(std::is_same<FORMAT, camera_outputs::format::RGBA8>::value){
            return "rgba8";
        }
        else if constexpr(std::is_same<FORMAT, camera_outputs::format::Float32>::value){
            return "float32";
        }
        else {
            return "uint32";
        }
    }

    template <typename DIRECTION>
    std::string output_direction(){
        return std::is_same<DIRECTION, camera_outputs::direction::Backward>::value ? "backward" : "forward";
    }

    template <typename F>
    OutputDescriptor output_descriptor(camera_outputs::RGB<F>){
        return {"rgb", output_format<F>(), {}, {}, 0};
    }
    template <typename F, typename P>
    OutputDescriptor output_descriptor(camera_outputs::Depth<F, P>){
        return {"depth", output_format<F>(), {}, {}, std::is_same<F, camera_outputs::format::RGBA8>::value ? P::SCALE : 0};
    }
    template <typename F, typename L>
    OutputDescriptor output_descriptor(camera_outputs::Segmentation<F, L>){
        return {"segmentation", output_format<F>(), {}, std::is_same<L, camera_outputs::labels::Semantic>::value ? "semantic" : "instance", 0};
    }
    template <typename F>
    OutputDescriptor output_descriptor(camera_outputs::Normals<F>){
        return {"normals", output_format<F>(), {}, {}, 0};
    }
    template <typename D, typename F, typename P>
    OutputDescriptor output_descriptor(camera_outputs::Flow<D, F, P>){
        return {"flow", output_format<F>(), output_direction<D>(), {}, std::is_same<F, camera_outputs::format::RGBA8>::value ? P::SCALE : 0};
    }
    template <typename D, typename F>
    OutputDescriptor output_descriptor(camera_outputs::FlowStatus<D, F>){
        return {"flow_status", output_format<F>(), output_direction<D>(), {}, 0};
    }
    inline OutputDescriptor output_descriptor(camera_outputs::Probes){
        return {"probes", {}, {}, {}, 0};
    }

    inline void output_contiguous_strides(OutputLayout& layout){
        layout.strides.resize(layout.shape.size());
        int64_t stride = 1;
        for(size_t axis = layout.shape.size(); axis > 0; axis--){
            layout.strides[axis - 1] = stride;
            stride *= layout.shape[axis - 1];
        }
    }

    template <typename SHAPE, size_t... AXES>
    std::vector<size_t> output_shape(std::index_sequence<AXES...>){
        return {static_cast<size_t>(SHAPE::template GET<AXES>)...};
    }

    template <typename SPEC, typename ENTRY>
    OutputMetadata output_metadata_entry(){
        using OUTPUT = typename camera_outputs::Entry<ENTRY>::OUTPUT;
        using SELECTION = typename camera_outputs::Entry<ENTRY>::SELECTION;
        using BUFFER = camera_outputs::Buffer<SPEC, ENTRY>;
        using SHAPE = typename BUFFER::TENSOR_SPEC::SHAPE;
        OutputMetadata result;
        result.descriptor = output_descriptor(OUTPUT{});
        result.layout.shape = output_shape<SHAPE>(std::make_index_sequence<SHAPE::LENGTH>{});
        for(size_t camera = 0; camera < SPEC::NUM_CAMERAS; camera++){
            if(SELECTION::selected(camera)){
                result.cameras.push_back(camera);
            }
        }
        if constexpr(std::is_same<OUTPUT, camera_outputs::Probes>::value){
            result.layout.axes = {"camera", "probe"};
            result.layout.dtype = "record";
            result.layout.item_size = sizeof(rl_tools::hyperlux::CollisionResult);
            for(const auto& field: {std::string("distance"), std::string("hit")}){
                OutputLayout layout = result.layout;
                layout.dtype = field == "distance" ? "float32" : "int32";
                layout.offset = field == "distance" ? offsetof(rl_tools::hyperlux::CollisionResult, distance) : offsetof(rl_tools::hyperlux::CollisionResult, hit);
                layout.item_size = sizeof(float);
                output_contiguous_strides(layout);
                for(auto& stride: layout.strides){
                    stride *= sizeof(rl_tools::hyperlux::CollisionResult) / layout.item_size;
                }
                result.fields.emplace_back(field, std::move(layout));
            }
        }
        else {
            result.layout.axes = {"camera", "height", "width"};
            if(result.descriptor.format == "rgba8"){
                result.has_packed = true;
                result.packed = result.layout;
                result.packed.dtype = "uint32";
                output_contiguous_strides(result.packed);
                result.layout.dtype = "uint8";
                result.layout.item_size = 1;
                result.layout.shape.push_back(4);
            }
            else {
                result.layout.dtype = result.descriptor.format;
            }
            if(result.layout.shape.size() == 4){
                result.layout.axes.push_back("channel");
            }
        }
        output_contiguous_strides(result.layout);
        return result;
    }

    template <typename SPEC, typename... ENTRIES>
    std::vector<OutputMetadata> output_metadata_list(camera_outputs::List<ENTRIES...>){
        return {output_metadata_entry<SPEC, ENTRIES>()...};
    }

    template <typename SPEC>
    std::vector<OutputMetadata> output_metadata(){
        return output_metadata_list<SPEC>(typename SPEC::OUTPUTS{});
    }

    template <typename T>
    struct OutputType { using TYPE = T; };

    template <typename FUNCTION, typename... ENTRIES>
    void visit_output(size_t index, FUNCTION&& function, camera_outputs::List<ENTRIES...>){
        size_t cursor = 0;
        bool found = false;
        auto visit = [&](auto entry){
            using ENTRY = typename decltype(entry)::TYPE;
            if(cursor++ == index){
                function(OutputType<typename camera_outputs::Entry<ENTRY>::OUTPUT>{});
                found = true;
            }
        };
        (visit(OutputType<ENTRIES>{}), ...);
        if(!found){
            throw std::out_of_range("hyperlux: output index out of range");
        }
    }

    template <typename SPEC, typename FUNCTION>
    void visit_output(size_t index, FUNCTION&& function){
        visit_output(index, std::forward<FUNCTION>(function), typename SPEC::OUTPUTS{});
    }

    inline const OutputLayout& output_layout(const OutputMetadata& metadata, const std::string& field){
        if(field == "packed" && metadata.has_packed){
            return metadata.packed;
        }
        if(metadata.fields.empty() && field.empty()){
            return metadata.layout;
        }
        for(const auto& entry: metadata.fields){
            if(entry.first == field){
                return entry.second;
            }
        }
        throw std::invalid_argument("hyperlux: choose a configured output field");
    }
}
