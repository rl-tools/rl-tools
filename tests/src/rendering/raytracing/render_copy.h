#ifndef TESTS_RENDERING_RAYTRACING_RENDER_COPY_H
#define TESTS_RENDERING_RAYTRACING_RENDER_COPY_H

#include <type_traits>
#include <vector>

// std::vector/array-backed staging for the render-device tensor copies: an alias tensor shaped
// like the renderer tensor points at the caller's storage
namespace golden {
    template <typename RENDER_DEVICE, typename DEVICE, typename TENSOR, typename ELEMENT>
    void copy_out(RENDER_DEVICE& render_device, DEVICE& device, const TENSOR& tensor, std::vector<ELEMENT>& out){
        using SPEC = typename TENSOR::SPEC;
        static_assert(std::is_same<ELEMENT, typename SPEC::T>::value);
        out.resize(SPEC::SIZE);
        rl_tools::Tensor<rl_tools::tensor::Specification<ELEMENT, typename SPEC::TI, typename SPEC::SHAPE>> alias;
        alias._data = out.data();
        rl_tools::copy(render_device, device, tensor, alias);
    }
    template <typename DEVICE, typename RENDER_DEVICE, typename ELEMENT, typename TENSOR>
    void copy_in(DEVICE& device, RENDER_DEVICE& render_device, const ELEMENT* source, TENSOR& tensor){
        using SPEC = typename TENSOR::SPEC;
        static_assert(std::is_same<ELEMENT, typename SPEC::T>::value);
        rl_tools::Tensor<rl_tools::tensor::Specification<ELEMENT, typename SPEC::TI, typename SPEC::SHAPE, true, rl_tools::tensor::RowMajorStride<typename SPEC::SHAPE>, true>> alias;
        alias._data = source;
        rl_tools::copy(device, render_device, alias, tensor);
    }
}

#endif
