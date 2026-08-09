#pragma once
// GPU staging buffer for device-resident inputs: uploads a host float32 array once; slices
// along axis 0 are handed out as DLPack CUDA views (e.g. one pre-sampled camera set per
// slice). Registered by every CUDA-capable component core so hyperdrone.cuda can resolve a
// provider from whichever domain package is in use.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace hyperdrone {
    struct CudaBuffer {
        void* pointer = nullptr;
        std::vector<size_t> shape;

        explicit CudaBuffer(nanobind::ndarray<const float, nanobind::c_contig, nanobind::device::cpu> source){
            shape.assign(source.shape_ptr(), source.shape_ptr() + source.ndim());
            const size_t bytes = source.size() * sizeof(float);
            if(cudaMalloc(&pointer, bytes) != cudaSuccess){
                throw std::runtime_error("hyperdrone: cudaMalloc failed");
            }
            if(cudaMemcpy(pointer, source.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess){
                cudaFree(pointer);
                pointer = nullptr;
                throw std::runtime_error("hyperdrone: upload to CUDA device memory failed");
            }
        }
        ~CudaBuffer(){
            if(pointer != nullptr){
                cudaFree(pointer);
            }
        }
        CudaBuffer(const CudaBuffer&) = delete;
        CudaBuffer& operator=(const CudaBuffer&) = delete;

        size_t slice_elements() const {
            size_t count = 1;
            for(size_t dimension = 1; dimension < shape.size(); dimension++){
                count *= shape[dimension];
            }
            return count;
        }
        const float* slice_pointer(size_t index) const {
            if(shape.empty() || index >= shape[0]){
                throw std::out_of_range("hyperdrone: CudaBuffer slice index out of range");
            }
            return (const float*)pointer + index * slice_elements();
        }
    };

    inline void register_cuda_staging(nanobind::module_& m){
        namespace nb = nanobind;
        nb::class_<CudaBuffer>(m, "CudaBuffer")
            .def(nb::init<nb::ndarray<const float, nb::c_contig, nb::device::cpu>>(), nb::arg("source"))
            .def_prop_ro("shape", [](const CudaBuffer& buffer){ return buffer.shape; })
            .def("slice_dlpack", [](CudaBuffer& buffer, size_t index){
                std::vector<size_t> slice_shape(buffer.shape.begin() + 1, buffer.shape.end());
                return nb::ndarray<>((void*)buffer.slice_pointer(index), slice_shape.size(), slice_shape.data(),
                                     nb::find(&buffer), nullptr, nb::dtype<float>(), nb::device::cuda::value, 0);
            }, nb::arg("index"))
            .def("dlpack", [](CudaBuffer& buffer){
                return nb::ndarray<>(buffer.pointer, buffer.shape.size(), buffer.shape.data(),
                                     nb::find(&buffer), nullptr, nb::dtype<float>(), nb::device::cuda::value, 0);
            });
    }
}
