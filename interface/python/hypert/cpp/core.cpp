#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_common.h>

#include "iface.h"

#include <dlfcn.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <map>

namespace nb = nanobind;
namespace rlt = rl_tools;
namespace rrt = rl_tools::rendering::raytracing;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
static DEVICE g_device;

using Vec3 = std::array<float, 3>;
using Vec4 = std::array<float, 4>;

using FloatArray = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;
using Transform = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;

static void extract_transform(const Transform& transform, float out[12]){
    if(transform.ndim() == 1 && transform.shape(0) == 12){
        std::memcpy(out, transform.data(), 12 * sizeof(float));
    }
    else if(transform.ndim() == 2 && transform.shape(0) == 3 && transform.shape(1) == 4){
        std::memcpy(out, transform.data(), 12 * sizeof(float));
    }
    else {
        throw std::invalid_argument("hypert: transform must have shape (12,) or (3, 4)");
    }
}

static nb::ndarray<nb::numpy, float> make_owned_array(const float* values, std::initializer_list<size_t> shape){
    size_t count = 1;
    for(size_t dim : shape){
        count *= dim;
    }
    float* buffer = new float[count];
    std::memcpy(buffer, values, count * sizeof(float));
    nb::capsule owner(buffer, [](void* pointer) noexcept { delete[] (float*)pointer; });
    return nb::ndarray<nb::numpy, float>(buffer, shape, owner);
}

template <typename TARGET>
static bool load_dispatch(TARGET& target, const std::string& path, int shading, bool rgb){
    switch(shading){
        case 0: return rgb ? rlt::load<rrt::Low, true>(g_device, target, path) : rlt::load<rrt::Low, false>(g_device, target, path);
        case 1: return rgb ? rlt::load<rrt::Medium, true>(g_device, target, path) : rlt::load<rrt::Medium, false>(g_device, target, path);
        case 2: return rgb ? rlt::load<rrt::High, true>(g_device, target, path) : rlt::load<rrt::High, false>(g_device, target, path);
        case 3: return rgb ? rlt::load<rrt::VeryHigh, true>(g_device, target, path) : rlt::load<rrt::VeryHigh, false>(g_device, target, path);
    }
    throw std::invalid_argument("hypert: shading must be in [0, 3]");
}

static rrt::Mesh make_mesh(FloatArray vertices, nb::ndarray<const int32_t, nb::c_contig, nb::device::cpu> indices,
                           Vec3 color, nb::object normals, nb::object tex_coords,
                           float metallic, float roughness, Vec3 emissive){
    if(vertices.ndim() != 2 || vertices.shape(1) != 3){
        throw std::invalid_argument("hypert: vertices must have shape (V, 3) and dtype float32");
    }
    if(indices.ndim() != 2 || indices.shape(1) != 3){
        throw std::invalid_argument("hypert: indices must have shape (F, 3) and dtype int32");
    }
    rrt::Mesh mesh;
    const size_t num_vertices = vertices.shape(0);
    const size_t num_faces = indices.shape(0);
    mesh.vertices.assign(vertices.data(), vertices.data() + num_vertices * 3);
    mesh.indices.assign(indices.data(), indices.data() + num_faces * 3);
    mesh.color[0] = color[0]; mesh.color[1] = color[1]; mesh.color[2] = color[2];
    mesh.metallic = metallic;
    mesh.roughness = roughness;
    mesh.emissive[0] = emissive[0]; mesh.emissive[1] = emissive[1]; mesh.emissive[2] = emissive[2];
    if(!normals.is_none()){
        auto normals_array = nb::cast<FloatArray>(normals);
        if(normals_array.ndim() != 2 || normals_array.shape(0) != num_vertices || normals_array.shape(1) != 3){
            throw std::invalid_argument("hypert: normals must have shape (V, 3) and dtype float32");
        }
        mesh.normals.assign(normals_array.data(), normals_array.data() + num_vertices * 3);
    }
    else {
        // area-weighted vertex normals so meshes shade correctly under normal/PBR tiers
        mesh.normals.assign(num_vertices * 3, 0.0f);
        for(size_t face = 0; face < num_faces; face++){
            const int32_t i0 = mesh.indices[face * 3 + 0], i1 = mesh.indices[face * 3 + 1], i2 = mesh.indices[face * 3 + 2];
            const float* v0 = &mesh.vertices[(size_t)i0 * 3];
            const float* v1 = &mesh.vertices[(size_t)i1 * 3];
            const float* v2 = &mesh.vertices[(size_t)i2 * 3];
            const float e1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
            const float e2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
            const float n[3] = {
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0]
            };
            for(int32_t index : {i0, i1, i2}){
                mesh.normals[(size_t)index * 3 + 0] += n[0];
                mesh.normals[(size_t)index * 3 + 1] += n[1];
                mesh.normals[(size_t)index * 3 + 2] += n[2];
            }
        }
        for(size_t vertex = 0; vertex < num_vertices; vertex++){
            float* n = &mesh.normals[vertex * 3];
            const float length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if(length > 0){
                n[0] /= length; n[1] /= length; n[2] /= length;
            }
        }
    }
    if(!tex_coords.is_none()){
        auto uv_array = nb::cast<FloatArray>(tex_coords);
        if(uv_array.ndim() != 2 || uv_array.shape(0) != num_vertices || uv_array.shape(1) != 2){
            throw std::invalid_argument("hypert: tex_coords must have shape (V, 2) and dtype float32");
        }
        mesh.tex_coords.assign(uv_array.data(), uv_array.data() + num_vertices * 2);
    }
    return mesh;
}

// JIT renderer library handle: dlopened once per path and kept for the process lifetime so
// repeated Renderer construction is cheap and CUDA/OWL teardown ordering stays trivial
struct JitLibrary {
    hypert::Renderer* (*create)();
    void (*destroy)(hypert::Renderer*);
    const char* (*config_string)();
};

static JitLibrary& get_jit_library(const std::string& path){
    static std::map<std::string, JitLibrary> libraries;
    auto existing = libraries.find(path);
    if(existing != libraries.end()){
        return existing->second;
    }
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if(handle == nullptr){
        throw std::runtime_error(std::string("hypert: failed to load renderer library: ") + dlerror());
    }
    JitLibrary library;
    library.create = (hypert::Renderer* (*)())dlsym(handle, "hypert_create");
    library.destroy = (void (*)(hypert::Renderer*))dlsym(handle, "hypert_destroy");
    library.config_string = (const char* (*)())dlsym(handle, "hypert_config_string");
    if(library.create == nullptr || library.destroy == nullptr || library.config_string == nullptr){
        throw std::runtime_error("hypert: renderer library is missing the hypert_create/hypert_destroy/hypert_config_string symbols: " + path);
    }
    return libraries.emplace(path, library).first->second;
}

struct JitRenderer {
    hypert::Renderer* renderer = nullptr;
    void (*destroy)(hypert::Renderer*) = nullptr;

    JitRenderer(const std::string& path, const std::string& expected_config){
        JitLibrary& library = get_jit_library(path);
        const std::string actual_config = library.config_string();
        if(actual_config != expected_config){
            throw std::runtime_error("hypert: renderer library config mismatch (expected \"" + expected_config + "\", library reports \"" + actual_config + "\"): " + path);
        }
        destroy = library.destroy;
        renderer = library.create();
    }
    ~JitRenderer(){
        if(renderer != nullptr){
            destroy(renderer);
        }
    }
    JitRenderer(const JitRenderer&) = delete;
    JitRenderer& operator=(const JitRenderer&) = delete;

    hypert::Config config() const { return renderer->config(); }

    size_t pixel_count() const {
        const hypert::Config c = renderer->config();
        return (size_t)c.num_cameras * c.height * c.width;
    }
    void check_image_shape(size_t ndim, const size_t* shape) const {
        const hypert::Config c = renderer->config();
        if(ndim != 3 || shape[0] != c.num_cameras || shape[1] != c.height || shape[2] != c.width){
            throw std::invalid_argument("hypert: output array must have shape (num_cameras, height, width)");
        }
    }
    void check_cameras_shape(const FloatArray& cameras) const {
        const hypert::Config c = renderer->config();
        const bool flat = cameras.ndim() == 2 && cameras.shape(0) == c.num_cameras && cameras.shape(1) == 12;
        const bool structured = cameras.ndim() == 3 && cameras.shape(0) == c.num_cameras && cameras.shape(1) == 4 && cameras.shape(2) == 3;
        if(!flat && !structured){
            throw std::invalid_argument("hypert: cameras must have shape (num_cameras, 12) or (num_cameras, 4, 3) and dtype float32");
        }
    }
};

NB_MODULE(hypert_core, m){
    m.doc() = "hypert core: scene assembly and JIT renderer loader for the RLtools raytracer";

    nb::class_<rrt::SceneLight>(m, "SceneLight")
        .def_static("directional", [](Vec3 direction, Vec3 color){
            rrt::SceneLight light{};
            light.type = 0;
            for(int i = 0; i < 3; i++){ light.direction[i] = direction[i]; light.color[i] = color[i]; }
            return light;
        }, nb::arg("direction"), nb::arg("color"))
        .def_static("point", [](Vec3 position, Vec3 color, Vec3 attenuation){
            rrt::SceneLight light{};
            light.type = 1;
            for(int i = 0; i < 3; i++){ light.position[i] = position[i]; light.color[i] = color[i]; }
            light.attenuation_constant = attenuation[0];
            light.attenuation_linear = attenuation[1];
            light.attenuation_quadratic = attenuation[2];
            return light;
        }, nb::arg("position"), nb::arg("color"), nb::arg("attenuation") = Vec3{1.0f, 0.0f, 0.0f})
        .def_static("spot", [](Vec3 position, Vec3 direction, Vec3 color, float cos_inner_cone, float cos_outer_cone, Vec3 attenuation){
            rrt::SceneLight light{};
            light.type = 2;
            for(int i = 0; i < 3; i++){ light.position[i] = position[i]; light.direction[i] = direction[i]; light.color[i] = color[i]; }
            light.cos_inner_cone = cos_inner_cone;
            light.cos_outer_cone = cos_outer_cone;
            light.attenuation_constant = attenuation[0];
            light.attenuation_linear = attenuation[1];
            light.attenuation_quadratic = attenuation[2];
            return light;
        }, nb::arg("position"), nb::arg("direction"), nb::arg("color"), nb::arg("cos_inner_cone"), nb::arg("cos_outer_cone"), nb::arg("attenuation") = Vec3{1.0f, 0.0f, 0.0f})
        .def_prop_ro("type", [](const rrt::SceneLight& light){ return light.type; });

    nb::class_<rrt::Mesh>(m, "Mesh")
        .def("__init__", [](rrt::Mesh* mesh, FloatArray vertices, nb::ndarray<const int32_t, nb::c_contig, nb::device::cpu> indices,
                            Vec3 color, nb::object normals, nb::object tex_coords, float metallic, float roughness, Vec3 emissive){
            new (mesh) rrt::Mesh(make_mesh(vertices, indices, color, normals, tex_coords, metallic, roughness, emissive));
        }, nb::arg("vertices"), nb::arg("indices"), nb::arg("color") = Vec3{0.8f, 0.8f, 0.8f},
           nb::arg("normals") = nb::none(), nb::arg("tex_coords") = nb::none(),
           nb::arg("metallic") = 0.0f, nb::arg("roughness") = 1.0f, nb::arg("emissive") = Vec3{0.0f, 0.0f, 0.0f})
        .def_prop_ro("num_vertices", [](const rrt::Mesh& mesh){ return mesh.vertices.size() / 3; })
        .def_prop_ro("num_faces", [](const rrt::Mesh& mesh){ return mesh.indices.size() / 3; });

    nb::class_<rrt::Object>(m, "Object")
        .def("__init__", [](rrt::Object* object, const std::string& name, uint32_t segmentation_class){
            new (object) rrt::Object();
            object->name = name;
            object->segmentation_class = segmentation_class;
        }, nb::arg("name") = "", nb::arg("segmentation_class") = 0)
        .def_rw("name", &rrt::Object::name)
        .def_rw("segmentation_class", &rrt::Object::segmentation_class)
        .def("add_mesh", [](rrt::Object& object, const rrt::Mesh& mesh){ object.meshes.push_back(mesh); })
        .def("add_light", [](rrt::Object& object, const rrt::SceneLight& light){ object.lights.push_back(light); })
        .def_prop_ro("num_meshes", [](const rrt::Object& object){ return object.meshes.size(); });

    nb::class_<rrt::ObjectAssembly>(m, "ObjectAssembly")
        .def(nb::init<>())
        .def_prop_ro("num_objects", [](const rrt::ObjectAssembly& assembly){ return assembly.objects.size(); })
        .def_prop_ro("num_parts", [](const rrt::ObjectAssembly& assembly){ return assembly.parts.size(); })
        .def("object_name", [](const rrt::ObjectAssembly& assembly, size_t index){ return assembly.objects.at(index).name; })
        .def("object_names", [](const rrt::ObjectAssembly& assembly){
            std::vector<std::string> names;
            for(const auto& object : assembly.objects){ names.push_back(object.name); }
            return names;
        })
        .def("segmentation_class", [](const rrt::ObjectAssembly& assembly, size_t index){ return assembly.objects.at(index).segmentation_class; })
        .def("set_segmentation_class", [](rrt::ObjectAssembly& assembly, size_t index, uint32_t segmentation_class){
            assembly.objects.at(index).segmentation_class = segmentation_class;
        })
        .def("part", [](const rrt::ObjectAssembly& assembly, size_t index){
            const auto& part = assembly.parts.at(index);
            return nb::make_tuple(part.object, make_owned_array(part.transform, {3, 4}));
        });

    nb::class_<rrt::AssetPool>(m, "AssetPool")
        .def(nb::init<>())
        .def("add_assembly", [](rrt::AssetPool& pool, const rrt::ObjectAssembly& assembly){ return rlt::add(g_device, pool, assembly).index; })
        .def("add_object", [](rrt::AssetPool& pool, const rrt::Object& object){ return rlt::add(g_device, pool, object).index; })
        .def("add_mesh", [](rrt::AssetPool& pool, const rrt::Mesh& mesh){ return rlt::add(g_device, pool, mesh).index; })
        .def_prop_ro("num_assets", [](const rrt::AssetPool& pool){ return pool.assemblies.size(); });

    nb::class_<rrt::Scene>(m, "Scene")
        .def(nb::init<>())
        .def("load", [](rrt::Scene& scene, const std::string& path, int shading, bool rgb){
            bool success;
            {
                nb::gil_scoped_release release;
                success = load_dispatch(scene, path, shading, rgb);
            }
            if(!success){
                throw std::runtime_error("hypert: failed to load scene from " + path);
            }
        }, nb::arg("path"), nb::arg("shading") = 2, nb::arg("rgb") = true)
        .def("add_object", [](rrt::Scene& scene, const rrt::Object& object, nb::object transform){
            if(transform.is_none()){
                return rlt::add(g_device, scene, object);
            }
            float values[12];
            extract_transform(nb::cast<Transform>(transform), values);
            return rlt::add(g_device, scene, object, values);
        }, nb::arg("object"), nb::arg("transform") = nb::none())
        .def("add_mesh", [](rrt::Scene& scene, const rrt::Mesh& mesh){ return rlt::add(g_device, scene, mesh); })
        .def("add_assembly", [](rrt::Scene& scene, const rrt::ObjectAssembly& assembly, nb::object transform){
            rrt::Placement placement;
            if(transform.is_none()){
                placement = rlt::add(g_device, scene, assembly);
            }
            else {
                float values[12];
                extract_transform(nb::cast<Transform>(transform), values);
                placement = rlt::add(g_device, scene, assembly, values);
            }
            return nb::make_tuple(placement.first_instance, placement.num_instances);
        }, nb::arg("assembly"), nb::arg("transform") = nb::none())
        .def("add_light", [](rrt::Scene& scene, const rrt::SceneLight& light){ scene.lights.push_back(light); })
        .def_prop_ro("num_objects", [](const rrt::Scene& scene){ return scene.objects.size(); })
        .def_prop_ro("num_instances", [](const rrt::Scene& scene){ return scene.instances.size(); })
        .def_prop_ro("num_lights", [](const rrt::Scene& scene){ return scene.lights.size(); })
        .def("object_name", [](const rrt::Scene& scene, size_t index){ return scene.objects.at(index).name; })
        .def("object_segmentation_class", [](const rrt::Scene& scene, size_t index){ return scene.objects.at(index).segmentation_class; })
        .def("set_object_segmentation_class", [](rrt::Scene& scene, size_t index, uint32_t segmentation_class){
            scene.objects.at(index).segmentation_class = segmentation_class;
        })
        .def("instance_object", [](const rrt::Scene& scene, size_t index){ return scene.instances.at(index).object; })
        .def("instance_objects", [](const rrt::Scene& scene){
            std::vector<size_t> objects;
            for(const auto& instance : scene.instances){ objects.push_back(instance.object); }
            return objects;
        })
        .def("instance_transform", [](const rrt::Scene& scene, size_t index){
            return make_owned_array(scene.instances.at(index).transform, {3, 4});
        });

    m.def("load_object", [](const std::string& path, int shading, bool rgb){
        rrt::Object object;
        bool success;
        {
            nb::gil_scoped_release release;
            success = load_dispatch(object, path, shading, rgb);
        }
        if(!success){
            throw std::runtime_error("hypert: failed to load object from " + path);
        }
        return object;
    }, nb::arg("path"), nb::arg("shading") = 2, nb::arg("rgb") = true);

    m.def("load_assembly", [](const std::string& path, int shading, bool rgb){
        rrt::ObjectAssembly assembly;
        bool success;
        {
            nb::gil_scoped_release release;
            success = load_dispatch(assembly, path, shading, rgb);
        }
        if(!success){
            throw std::runtime_error("hypert: failed to load assembly from " + path);
        }
        return assembly;
    }, nb::arg("path"), nb::arg("shading") = 2, nb::arg("rgb") = true);

    m.def("make_camera", [](Vec3 position, Vec3 look_at, Vec3 up, float fov, float aspect){
        rrt::Camera<float> camera = rlt::make_camera_data(position.data(), look_at.data(), up.data(), fov, aspect);
        return make_owned_array(camera.pos, {4, 3});
    }, nb::arg("position"), nb::arg("look_at"), nb::arg("up"), nb::arg("fov"), nb::arg("aspect"));

    m.def("make_transform", [](Vec3 position, Vec4 orientation_wxyz){
        float values[12];
        rlt::make_transform(position.data(), orientation_wxyz.data(), values);
        return make_owned_array(values, {3, 4});
    }, nb::arg("position"), nb::arg("orientation_wxyz") = Vec4{1.0f, 0.0f, 0.0f, 0.0f});

    m.def("compose_transforms", [](Transform a, Transform b){
        float a_values[12], b_values[12], out[12];
        extract_transform(a, a_values);
        extract_transform(b, b_values);
        rlt::compose_transforms(a_values, b_values, out);
        return make_owned_array(out, {3, 4});
    }, nb::arg("a"), nb::arg("b"));

    nb::class_<JitRenderer>(m, "JitRenderer")
        .def(nb::init<const std::string&, const std::string&>(), nb::arg("library_path"), nb::arg("expected_config"))
        .def_prop_ro("backend", [](const JitRenderer& jit){ return std::string(jit.renderer->backend()); })
        .def("init", [](JitRenderer& jit, const rrt::Scene& scene, const rrt::AssetPool* pool){
            nb::gil_scoped_release release;
            jit.renderer->init(&scene, pool);
        }, nb::arg("scene"), nb::arg("asset_pool").none() = nb::none(), nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>())
        .def("update", [](JitRenderer& jit){
            nb::gil_scoped_release release;
            jit.renderer->update();
        })
        .def("synchronize", [](JitRenderer& jit){
            nb::gil_scoped_release release;
            jit.renderer->synchronize();
        })
        .def("set_cameras", [](JitRenderer& jit, FloatArray cameras){
            jit.check_cameras_shape(cameras);
            jit.renderer->set_cameras(cameras.data());
        }, nb::arg("cameras"))
        .def("set_motion_blur_cameras", [](JitRenderer& jit, FloatArray cameras_open, FloatArray cameras_close){
            jit.check_cameras_shape(cameras_open);
            jit.check_cameras_shape(cameras_close);
            jit.renderer->set_motion_blur_cameras(cameras_open.data(), cameras_close.data());
        }, nb::arg("cameras_open"), nb::arg("cameras_close"))
        .def("generate_cameras", [](JitRenderer& jit, Vec3 center, float radius, Vec3 up, float fov){
            jit.renderer->generate_cameras(center.data(), radius, up.data(), fov);
        }, nb::arg("center"), nb::arg("radius"), nb::arg("up"), nb::arg("fov"))
        .def("generate_probe_directions", [](JitRenderer& jit){ jit.renderer->generate_probe_directions(); })
        .def("render", [](JitRenderer& jit, int target, int phase){
            nb::gil_scoped_release release;
            jit.renderer->render((hypert::RenderTarget)target, (hypert::RenderPhase)phase);
        }, nb::arg("target"), nb::arg("phase"))
        .def("read_frame_buffer", [](JitRenderer& jit, nb::ndarray<uint32_t, nb::c_contig, nb::device::cpu> out){
            size_t shape[3] = {(size_t)out.shape(0), out.ndim() > 1 ? (size_t)out.shape(1) : 0, out.ndim() > 2 ? (size_t)out.shape(2) : 0};
            jit.check_image_shape(out.ndim(), shape);
            nb::gil_scoped_release release;
            jit.renderer->read_frame_buffer(out.data());
        }, nb::arg("out"))
        .def("read_depth_buffer", [](JitRenderer& jit, nb::ndarray<float, nb::c_contig, nb::device::cpu> out){
            size_t shape[3] = {(size_t)out.shape(0), out.ndim() > 1 ? (size_t)out.shape(1) : 0, out.ndim() > 2 ? (size_t)out.shape(2) : 0};
            jit.check_image_shape(out.ndim(), shape);
            nb::gil_scoped_release release;
            jit.renderer->read_depth_buffer(out.data());
        }, nb::arg("out"))
        .def("read_segmentation_buffer", [](JitRenderer& jit, nb::ndarray<uint32_t, nb::c_contig, nb::device::cpu> out){
            size_t shape[3] = {(size_t)out.shape(0), out.ndim() > 1 ? (size_t)out.shape(1) : 0, out.ndim() > 2 ? (size_t)out.shape(2) : 0};
            jit.check_image_shape(out.ndim(), shape);
            nb::gil_scoped_release release;
            jit.renderer->read_segmentation_buffer(out.data());
        }, nb::arg("out"))
        .def("read_collision_results", [](JitRenderer& jit, nb::ndarray<float, nb::c_contig, nb::device::cpu> distances, nb::ndarray<int32_t, nb::c_contig, nb::device::cpu> hits){
            const hypert::Config config = jit.renderer->config();
            const size_t expected = (size_t)config.num_cameras * config.num_probes;
            if(distances.size() != expected || hits.size() != expected){
                throw std::invalid_argument("hypert: collision output arrays must have num_cameras * num_probes elements");
            }
            nb::gil_scoped_release release;
            jit.renderer->read_collision_results(distances.data(), hits.data());
        }, nb::arg("distances"), nb::arg("hits"))
        .def("framebuffer_device_ptr", [](JitRenderer& jit){ return (uintptr_t)jit.renderer->framebuffer_device_ptr(); })
        .def("depthbuffer_device_ptr", [](JitRenderer& jit){ return (uintptr_t)jit.renderer->depthbuffer_device_ptr(); })
        .def("frame_view", [](JitRenderer& jit, bool refresh){
            const hypert::Config c = jit.renderer->config();
            uint32_t* pointer;
            {
                nb::gil_scoped_release release;
                pointer = jit.renderer->frame_buffer_host(refresh);
            }
            return nb::ndarray<nb::numpy, uint32_t>(pointer, {c.num_cameras, c.height, c.width}, nb::find(&jit));
        }, nb::arg("refresh") = true)
        .def("depth_view", [](JitRenderer& jit, bool refresh){
            const hypert::Config c = jit.renderer->config();
            float* pointer;
            {
                nb::gil_scoped_release release;
                pointer = jit.renderer->depth_buffer_host(refresh);
            }
            return nb::ndarray<nb::numpy, float>(pointer, {c.num_cameras, c.height, c.width}, nb::find(&jit));
        }, nb::arg("refresh") = true)
        .def("segmentation_view", [](JitRenderer& jit, bool refresh){
            const hypert::Config c = jit.renderer->config();
            uint32_t* pointer;
            {
                nb::gil_scoped_release release;
                pointer = jit.renderer->segmentation_buffer_host(refresh);
            }
            return nb::ndarray<nb::numpy, uint32_t>(pointer, {c.num_cameras, c.height, c.width}, nb::find(&jit));
        }, nb::arg("refresh") = true)
        .def("frame_dlpack", [](JitRenderer& jit){
            const hypert::Config c = jit.renderer->config();
            uint32_t* pointer = jit.renderer->frame_buffer_live();
            return nb::ndarray<>(pointer, {c.num_cameras, c.height, c.width}, nb::find(&jit), {},
                                 nb::dtype<uint32_t>(), jit.renderer->buffer_device_type(), 0);
        })
        .def("depth_dlpack", [](JitRenderer& jit){
            const hypert::Config c = jit.renderer->config();
            float* pointer = jit.renderer->depth_buffer_live();
            return nb::ndarray<>(pointer, {c.num_cameras, c.height, c.width}, nb::find(&jit), {},
                                 nb::dtype<float>(), jit.renderer->buffer_device_type(), 0);
        })
        .def("buffer_device_type", [](JitRenderer& jit){ return jit.renderer->buffer_device_type(); })
        .def("save", [](JitRenderer& jit, int target, const std::string& path){
            nb::gil_scoped_release release;
            jit.renderer->save((hypert::SaveTarget)target, path.c_str());
        }, nb::arg("target"), nb::arg("path"))
        .def("scene_bounds", [](JitRenderer& jit){
            float center[3], half_extent[3], camera_radius;
            jit.renderer->scene_bounds(center, half_extent, camera_radius);
            return nb::make_tuple(make_owned_array(center, {3}), make_owned_array(half_extent, {3}), camera_radius);
        })
        .def("can_attach", [](JitRenderer& jit, size_t camera, size_t overlay){ return jit.renderer->can_attach(camera, overlay); })
        .def("attach", [](JitRenderer& jit, size_t camera, size_t overlay){ jit.renderer->attach(camera, overlay); })
        .def("detach", [](JitRenderer& jit, size_t camera, size_t overlay){ jit.renderer->detach(camera, overlay); })
        .def("can_spawn", [](JitRenderer& jit, size_t overlay, size_t asset){ return jit.renderer->can_spawn(overlay, asset); })
        .def("spawn", [](JitRenderer& jit, size_t overlay, size_t asset, Transform transform){
            float values[12];
            extract_transform(transform, values);
            hypert::OverlayPlacementData placement = jit.renderer->spawn(overlay, asset, values);
            return nb::make_tuple(placement.first_slot, placement.num_parts, placement.first_part);
        }, nb::arg("overlay"), nb::arg("asset"), nb::arg("transform"))
        .def("despawn", [](JitRenderer& jit, size_t overlay, std::array<size_t, 3> placement){
            jit.renderer->despawn(overlay, hypert::OverlayPlacementData{placement[0], placement[1], placement[2]});
        }, nb::arg("overlay"), nb::arg("placement"))
        .def("set_transform", [](JitRenderer& jit, size_t overlay, std::array<size_t, 3> placement, Transform transform){
            float values[12];
            extract_transform(transform, values);
            jit.renderer->set_transform(overlay, hypert::OverlayPlacementData{placement[0], placement[1], placement[2]}, values);
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("transform"))
        .def("set_part_transform", [](JitRenderer& jit, size_t overlay, std::array<size_t, 3> placement, size_t part, Transform transform){
            float values[12];
            extract_transform(transform, values);
            jit.renderer->set_part_transform(overlay, hypert::OverlayPlacementData{placement[0], placement[1], placement[2]}, part, values);
        }, nb::arg("overlay"), nb::arg("placement"), nb::arg("part"), nb::arg("transform"));
}
