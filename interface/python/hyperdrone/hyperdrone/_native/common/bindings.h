#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <cstring>
#include <stdexcept>

namespace hyperdrone {
    namespace nb = nanobind;

    using Vec3 = std::array<float, 3>;
    using Vec4 = std::array<float, 4>;
    using FloatArray = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
    using Transform = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;

    inline void extract_transform(const Transform& transform, float out[12]){
        if(transform.ndim() == 1 && transform.shape(0) == 12){
            std::memcpy(out, transform.data(), 12 * sizeof(float));
        }
        else if(transform.ndim() == 2 && transform.shape(0) == 3 && transform.shape(1) == 4){
            std::memcpy(out, transform.data(), 12 * sizeof(float));
        }
        else {
            throw std::invalid_argument("hyperdrone: transform must have shape (12,) or (3, 4)");
        }
    }

    inline nb::ndarray<nb::numpy, float> make_owned_array(const float* values, std::initializer_list<size_t> shape){
        size_t count = 1;
        for(size_t dim : shape){
            count *= dim;
        }
        float* buffer = new float[count];
        std::memcpy(buffer, values, count * sizeof(float));
        nb::capsule owner(buffer, [](void* pointer) noexcept { delete[] (float*)pointer; });
        return nb::ndarray<nb::numpy, float>(buffer, shape, owner);
    }
}
