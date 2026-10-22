#pragma once
#include "output.h"
#include "bindings.h"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include <functional>

namespace hyperlux {
    inline nb::dlpack::dtype output_dtype(const std::string& dtype){
        if(dtype == "float32"){
            return nb::dtype<float>();
        }
        if(dtype == "uint8"){
            return nb::dtype<uint8_t>();
        }
        if(dtype == "int32"){
            return nb::dtype<int32_t>();
        }
        if(dtype == "uint32"){
            return nb::dtype<uint32_t>();
        }
        throw std::invalid_argument("hyperlux: output has no scalar dtype");
    }

    inline nb::dict output_layout_dict(const OutputLayout& layout){
        nb::dict result;
        result["shape"] = nb::cast(layout.shape);
        result["dtype"] = nb::cast(layout.dtype);
        result["axes"] = nb::cast(layout.axes);
        return result;
    }

    inline nb::list output_metadata_dicts(const std::vector<OutputMetadata>& metadata){
        nb::list results;
        for(const auto& value: metadata){
            nb::dict result = output_layout_dict(value.layout);
            nb::dict descriptor;
            descriptor["quantity"] = nb::cast(value.descriptor.quantity);
            if(!value.descriptor.format.empty()){
                descriptor["format"] = nb::cast(value.descriptor.format);
            }
            if(!value.descriptor.direction.empty()){
                descriptor["direction"] = nb::cast(value.descriptor.direction);
            }
            if(!value.descriptor.labels.empty()){
                descriptor["labels"] = nb::cast(value.descriptor.labels);
            }
            if(value.descriptor.scale > 0){
                descriptor["scale"] = value.descriptor.scale;
            }
            result["descriptor"] = descriptor;
            result["cameras"] = nb::cast(value.cameras);
            result["device"] = nb::make_tuple(value.device_type, value.device_id);
            result["block_count"] = value.block_count;
            if(value.has_packed){
                result["packed"] = output_layout_dict(value.packed);
            }
            if(!value.fields.empty()){
                nb::dict fields;
                for(const auto& field: value.fields){
                    fields[nb::str(field.first.c_str())] = output_layout_dict(field.second);
                }
                result["fields"] = fields;
            }
            results.append(result);
        }
        return results;
    }

    inline void validate_output_array(const OutputLayout& layout, const nb::ndarray<nb::c_contig>& array){
        bool valid = array.ndim() == layout.shape.size() && array.dtype() == output_dtype(layout.dtype);
        for(size_t axis = 0; valid && axis < layout.shape.size(); axis++){
            valid = array.shape(axis) == layout.shape[axis];
        }
        if(!valid || reinterpret_cast<uintptr_t>(array.data()) % layout.item_size != 0){
            throw std::invalid_argument("hyperlux: output destination must match the compiled shape and dtype and be aligned");
        }
        if(array.device_type() != 1 && array.device_type() != 2){
            throw std::invalid_argument("hyperlux: output destinations must use CPU or CUDA memory");
        }
    }

    struct NativeOutputBlock {
        nb::object owner;
        void* pointer = nullptr;
        OutputLayout layout;
        int device_type = 1;
        int device_id = 0;
        std::function<void(uintptr_t)> order_consumer;
    };

    inline void bind_output_blocks(nb::module_& module){
        if(nb::type<NativeOutputBlock>().is_valid()){
            return;
        }
        nb::class_<NativeOutputBlock>(module, "NativeOutputBlock")
            .def("__dlpack_device__", [](const NativeOutputBlock& block){ return nb::make_tuple(block.device_type, block.device_id); })
            .def("__dlpack__", [](NativeOutputBlock& block, nb::object stream, nb::object, nb::object device, nb::object copy){
                if(!device.is_none()){
                    auto requested = nb::cast<std::pair<int, int>>(device);
                    if(requested.first != block.device_type || requested.second != block.device_id){
                        throw std::invalid_argument("hyperlux: live views cannot change device");
                    }
                }
                if(!copy.is_none() && nb::cast<bool>(copy)){
                    throw std::invalid_argument("hyperlux: use read() for owned output copies");
                }
                if(block.order_consumer && block.device_type == 2){
                    int64_t value = stream.is_none() ? 0 : nb::cast<int64_t>(stream);
                    if(value != -1){
                        block.order_consumer(value == 1 ? 0 : static_cast<uintptr_t>(value));
                    }
                }
                return nb::ndarray<>(block.pointer, block.layout.shape.size(), block.layout.shape.data(), nb::find(&block), block.layout.strides.data(), output_dtype(block.layout.dtype), block.device_type, block.device_id);
            }, nb::arg("stream") = nb::none(), nb::arg("max_version") = nb::none(), nb::arg("dl_device") = nb::none(), nb::arg("copy") = nb::none());
    }
}
