#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

// type-only include: the operations headers define non-inline functions and may appear in
// only one translation unit of this module (scene_bindings.cpp)
#include <rl_tools/rendering/raytracing/scene.h>

#include "bindings.h"
#include "iface.h"
#include "jit_host.h"

#include <stdexcept>
#include <string>

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
#include "cuda_staging.h"
#define HYPERDRONE_RENDER_CORE_HAS_CUDA 1
#else
#define HYPERDRONE_RENDER_CORE_HAS_CUDA 0
#endif

namespace nb = nanobind;
namespace rrt = rl_tools::rendering::raytracing;
namespace hdr = hyperdrone::render;

using hyperdrone::FloatArray;
using hyperdrone::Transform;
using hyperdrone::Vec3;
using hyperdrone::extract_transform;
using hyperdrone::make_owned_array;

void register_scene_bindings(nb::module_& m);

struct JitRenderer {
    hyperdrone::JitInstance<hdr::Renderer> instance;

    JitRenderer(const std::string& path, const std::string& expected_config)
        : instance(path, "hyperdrone_render", HYPERDRONE_RENDER_IFACE_VERSION, expected_config) {}

    hdr::Renderer* operator->() const { return instance.instance; }
    hdr::Renderer& renderer() const { return *instance.instance; }

    size_t pixel_count() const {
        const hdr::Config c = renderer().config();
        return (size_t)c.num_cameras * c.height * c.width;
    }
    void check_image_shape(size_t ndim, const size_t* shape) const {
        const hdr::Config c = renderer().config();
        if(ndim != 3 || shape[0] != c.num_cameras || shape[1] != c.height || shape[2] != c.width){
            throw std::invalid_argument("hyperdrone: output array must have shape (num_cameras, height, width)");
        }
    }
    void check_cameras_dims(size_t ndim, const int64_t* dims) const {
        const hdr::Config c = renderer().config();
        const bool flat = ndim == 2 && dims[0] == c.num_cameras && dims[1] == 12;
        const bool structured = ndim == 3 && dims[0] == c.num_cameras && dims[1] == 4 && dims[2] == 3;
        if(!flat && !structured){
            throw std::invalid_argument("hyperdrone: cameras must have shape (num_cameras, 12) or (num_cameras, 4, 3) and dtype float32");
        }
    }
    void check_cameras_shape(const FloatArray& cameras) const {
        int64_t dims[3] = {0, 0, 0};
        for(size_t dimension = 0; dimension < cameras.ndim() && dimension < 3; dimension++){
            dims[dimension] = (int64_t)cameras.shape(dimension);
        }
        check_cameras_dims(cameras.ndim(), dims);
    }
};

NB_MODULE(hyperdrone_render_core, m){
    m.doc() = "hyperdrone render core: scene assembly and JIT renderer loader for the RLtools raytracer";

    register_scene_bindings(m);

    auto jit_renderer_class = nb::class_<JitRenderer>(m, "JitRenderer");
    jit_renderer_class
        .def(nb::init<const std::string&, const std::string&>(), nb::arg("library_path"), nb::arg("expected_config"))
        .def_prop_ro("backend", [](const JitRenderer& jit){ return std::string(jit->backend()); })
        .def("init", [](JitRenderer& jit, const rrt::Scene& scene, const rrt::AssetPool* pool){
            nb::gil_scoped_release release;
            jit->init(&scene, pool);
        }, nb::arg("scene"), nb::arg("asset_pool").none() = nb::none(), nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
        .def("update", [](JitRenderer& jit){
            nb::gil_scoped_release release;
            jit->update();
        })
        .def("synchronize", [](JitRenderer& jit){
            nb::gil_scoped_release release;
            jit->synchronize();
        })
        .def("set_cameras", [](JitRenderer& jit, FloatArray cameras){
            jit.check_cameras_shape(cameras);
            jit->set_cameras(cameras.data());
        }, nb::arg("cameras"))
        .def("set_motion_blur_cameras", [](JitRenderer& jit, FloatArray cameras_open, FloatArray cameras_close){
            jit.check_cameras_shape(cameras_open);
            jit.check_cameras_shape(cameras_close);
            jit->set_motion_blur_cameras(cameras_open.data(), cameras_close.data());
        }, nb::arg("cameras_open"), nb::arg("cameras_close"))
        .def("generate_cameras", [](JitRenderer& jit, Vec3 center, float radius, Vec3 up, float fov){
            jit->generate_cameras(center.data(), radius, up.data(), fov);
        }, nb::arg("center"), nb::arg("radius"), nb::arg("up"), nb::arg("fov"))
        .def("generate_probe_directions", [](JitRenderer& jit){ jit->generate_probe_directions(); })
        .def("render", [](JitRenderer& jit, int target, int phase){
            nb::gil_scoped_release release;
            jit->render((hdr::RenderTarget)target, (hdr::RenderPhase)phase);
        }, nb::arg("target"), nb::arg("phase"))
        .def("read_frame_buffer", [](JitRenderer& jit, nb::ndarray<uint32_t, nb::c_contig, nb::device::cpu> out){
            size_t shape[3] = {(size_t)out.shape(0), out.ndim() > 1 ? (size_t)out.shape(1) : 0, out.ndim() > 2 ? (size_t)out.shape(2) : 0};
            jit.check_image_shape(out.ndim(), shape);
            nb::gil_scoped_release release;
            jit->read_frame_buffer(out.data());
        }, nb::arg("out"))
        .def("read_depth_buffer", [](JitRenderer& jit, nb::ndarray<float, nb::c_contig, nb::device::cpu> out){
            size_t shape[3] = {(size_t)out.shape(0), out.ndim() > 1 ? (size_t)out.shape(1) : 0, out.ndim() > 2 ? (size_t)out.shape(2) : 0};
            jit.check_image_shape(out.ndim(), shape);
            nb::gil_scoped_release release;
            jit->read_depth_buffer(out.data());
        }, nb::arg("out"))
        .def("read_segmentation_buffer", [](JitRenderer& jit, nb::ndarray<uint32_t, nb::c_contig, nb::device::cpu> out){
            size_t shape[3] = {(size_t)out.shape(0), out.ndim() > 1 ? (size_t)out.shape(1) : 0, out.ndim() > 2 ? (size_t)out.shape(2) : 0};
            jit.check_image_shape(out.ndim(), shape);
            nb::gil_scoped_release release;
            jit->read_segmentation_buffer(out.data());
        }, nb::arg("out"))
        .def("read_collision_results", [](JitRenderer& jit, nb::ndarray<float, nb::c_contig, nb::device::cpu> distances, nb::ndarray<int32_t, nb::c_contig, nb::device::cpu> hits){
            const hdr::Config config = jit->config();
            const size_t expected = (size_t)config.num_cameras * config.num_probes;
            if(distances.size() != expected || hits.size() != expected){
                throw std::invalid_argument("hyperdrone: collision output arrays must have num_cameras * num_probes elements");
            }
            nb::gil_scoped_release release;
            jit->read_collision_results(distances.data(), hits.data());
        }, nb::arg("distances"), nb::arg("hits"))
        .def("framebuffer_device_ptr", [](JitRenderer& jit){ return (uintptr_t)jit->framebuffer_device_ptr(); })
        .def("depthbuffer_device_ptr", [](JitRenderer& jit){ return (uintptr_t)jit->depthbuffer_device_ptr(); })
        .def("frame_view", [](JitRenderer& jit, bool refresh){
            const hdr::Config c = jit->config();
            uint32_t* pointer;
            {
                nb::gil_scoped_release release;
                pointer = jit->frame_buffer_host(refresh);
            }
            return nb::ndarray<nb::numpy, uint32_t>(pointer, {c.num_cameras, c.height, c.width}, nb::find(&jit));
        }, nb::arg("refresh") = true)
        .def("depth_view", [](JitRenderer& jit, bool refresh){
            const hdr::Config c = jit->config();
            float* pointer;
            {
                nb::gil_scoped_release release;
                pointer = jit->depth_buffer_host(refresh);
            }
            return nb::ndarray<nb::numpy, float>(pointer, {c.num_cameras, c.height, c.width}, nb::find(&jit));
        }, nb::arg("refresh") = true)
        .def("segmentation_view", [](JitRenderer& jit, bool refresh){
            const hdr::Config c = jit->config();
            uint32_t* pointer;
            {
                nb::gil_scoped_release release;
                pointer = jit->segmentation_buffer_host(refresh);
            }
            return nb::ndarray<nb::numpy, uint32_t>(pointer, {c.num_cameras, c.height, c.width}, nb::find(&jit));
        }, nb::arg("refresh") = true)
        .def("frame_dlpack", [](JitRenderer& jit){
            const hdr::Config c = jit->config();
            uint32_t* pointer = jit->frame_buffer_live();
            return nb::ndarray<>(pointer, {c.num_cameras, c.height, c.width}, nb::find(&jit), {},
                                 nb::dtype<uint32_t>(), jit->buffer_device_type(), 0);
        })
        .def("depth_dlpack", [](JitRenderer& jit){
            const hdr::Config c = jit->config();
            float* pointer = jit->depth_buffer_live();
            return nb::ndarray<>(pointer, {c.num_cameras, c.height, c.width}, nb::find(&jit), {},
                                 nb::dtype<float>(), jit->buffer_device_type(), 0);
        })
        .def("buffer_device_type", [](JitRenderer& jit){ return jit->buffer_device_type(); })
        .def("save", [](JitRenderer& jit, int target, const std::string& path){
            nb::gil_scoped_release release;
            jit->save((hdr::SaveTarget)target, path.c_str());
        }, nb::arg("target"), nb::arg("path"))
        .def("scene_bounds", [](JitRenderer& jit){
            float center[3], half_extent[3], camera_radius;
            jit->scene_bounds(center, half_extent, camera_radius);
            return nb::make_tuple(make_owned_array(center, {3}), make_owned_array(half_extent, {3}), camera_radius);
        })
        .def("can_attach", [](JitRenderer& jit, size_t camera, size_t overlay){ return jit->can_attach(camera, overlay); })
        .def("attach", [](JitRenderer& jit, size_t camera, size_t overlay){ jit->attach(camera, overlay); })
        .def("detach", [](JitRenderer& jit, size_t camera, size_t overlay){ jit->detach(camera, overlay); })
        .def("can_spawn", [](JitRenderer& jit, size_t overlay, size_t asset){ return jit->can_spawn(overlay, asset); })
        .def("spawn", [](JitRenderer& jit, size_t overlay, size_t asset, Transform transform){
            float values[12];
            extract_transform(transform, values);
            hdr::OverlayPlacementData placement = jit->spawn(overlay, asset, values);
            return nb::make_tuple(placement.first_slot, placement.num_parts, placement.first_part);
        }, nb::arg("overlay"), nb::arg("asset"), nb::arg("transform"))
        .def("despawn", [](JitRenderer& jit, size_t overlay, std::array<size_t, 3> placement){
            jit->despawn(overlay, hdr::OverlayPlacementData{placement[0], placement[1], placement[2]});
        }, nb::arg("overlay"), nb::arg("placement"))
        .def("set_transform", [](JitRenderer& jit, size_t overlay, std::array<size_t, 3> placement, Transform transform){
            float values[12];
            extract_transform(transform, values);
            jit->set_transform(overlay, hdr::OverlayPlacementData{placement[0], placement[1], placement[2]}, values);
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("transform"))
        .def("set_part_transform", [](JitRenderer& jit, size_t overlay, std::array<size_t, 3> placement, size_t part, Transform transform){
            float values[12];
            extract_transform(transform, values);
            jit->set_part_transform(overlay, hdr::OverlayPlacementData{placement[0], placement[1], placement[2]}, part, values);
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("part"), nb::arg("transform"));

#if HYPERDRONE_RENDER_CORE_HAS_CUDA
    m.attr("HAS_CUDA") = true;

    hyperdrone::register_cuda_staging(m);

    jit_renderer_class
        .def("set_cameras_device", [](JitRenderer& jit, nb::ndarray<const float, nb::c_contig, nb::device::cuda> cameras, uintptr_t stream){
            int64_t dims[3] = {0, 0, 0};
            for(size_t dimension = 0; dimension < cameras.ndim() && dimension < 3; dimension++){
                dims[dimension] = (int64_t)cameras.shape(dimension);
            }
            jit.check_cameras_dims(cameras.ndim(), dims);
            jit->set_cameras_device(cameras.data(), stream);
        }, nb::arg("cameras"), nb::arg("stream") = 0)
        // benchmark fast path: index a pre-uploaded set without any per-call DLPack traffic
        .def("set_cameras_from_cuda_buffer", [](JitRenderer& jit, hyperdrone::CudaBuffer& buffer, size_t index, uintptr_t stream){
            const hdr::Config c = jit->config();
            if(buffer.slice_elements() != (size_t)c.num_cameras * 12){
                throw std::invalid_argument("hyperdrone: CudaBuffer slices must hold num_cameras * 12 floats");
            }
            jit->set_cameras_device(buffer.slice_pointer(index), stream);
        }, nb::arg("buffer"), nb::arg("index"), nb::arg("stream") = 0);
#else
    m.attr("HAS_CUDA") = false;
#endif
}
