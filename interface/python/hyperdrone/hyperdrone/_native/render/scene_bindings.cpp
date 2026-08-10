#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_common.h>

#include "bindings.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace nb = nanobind;
namespace rlt = rl_tools;
namespace rrt = rl_tools::rendering::raytracing;

using hyperdrone::FloatArray;
using hyperdrone::Transform;
using hyperdrone::Vec3;
using hyperdrone::Vec4;
using hyperdrone::extract_transform;
using hyperdrone::make_owned_array;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
static DEVICE g_device;

template <typename TARGET>
static bool load_dispatch(TARGET& target, const std::string& path, int fidelity, bool rgb){
    switch(fidelity){
        case 0: return rgb ? rlt::load<rrt::Low, true>(g_device, target, path) : rlt::load<rrt::Low, false>(g_device, target, path);
        case 1: return rgb ? rlt::load<rrt::Medium, true>(g_device, target, path) : rlt::load<rrt::Medium, false>(g_device, target, path);
        case 2: return rgb ? rlt::load<rrt::High, true>(g_device, target, path) : rlt::load<rrt::High, false>(g_device, target, path);
        case 3: return rgb ? rlt::load<rrt::VeryHigh, true>(g_device, target, path) : rlt::load<rrt::VeryHigh, false>(g_device, target, path);
    }
    throw std::invalid_argument("hyperdrone: fidelity must be in [0, 3]");
}

static rrt::Mesh make_mesh(FloatArray vertices, nb::ndarray<const int32_t, nb::c_contig, nb::device::cpu> indices,
                           Vec3 color, nb::object normals, nb::object tex_coords,
                           float metallic, float roughness, Vec3 emissive){
    if(vertices.ndim() != 2 || vertices.shape(1) != 3){
        throw std::invalid_argument("hyperdrone: vertices must have shape (V, 3) and dtype float32");
    }
    if(indices.ndim() != 2 || indices.shape(1) != 3){
        throw std::invalid_argument("hyperdrone: indices must have shape (F, 3) and dtype int32");
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
            throw std::invalid_argument("hyperdrone: normals must have shape (V, 3) and dtype float32");
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
            throw std::invalid_argument("hyperdrone: tex_coords must have shape (V, 2) and dtype float32");
        }
        mesh.tex_coords.assign(uv_array.data(), uv_array.data() + num_vertices * 2);
    }
    return mesh;
}

void register_scene_bindings(nb::module_& m){
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
        .def("load", [](rrt::Scene& scene, const std::string& path, int fidelity, bool rgb){
            bool success;
            {
                nb::gil_scoped_release release;
                success = load_dispatch(scene, path, fidelity, rgb);
            }
            if(!success){
                throw std::runtime_error("hyperdrone: failed to load scene from " + path);
            }
        }, nb::arg("path"), nb::arg("fidelity") = 2, nb::arg("rgb") = true)
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

    m.def("load_object", [](const std::string& path, int fidelity, bool rgb){
        rrt::Object object;
        bool success;
        {
            nb::gil_scoped_release release;
            success = load_dispatch(object, path, fidelity, rgb);
        }
        if(!success){
            throw std::runtime_error("hyperdrone: failed to load object from " + path);
        }
        return object;
    }, nb::arg("path"), nb::arg("fidelity") = 2, nb::arg("rgb") = true);

    m.def("load_assembly", [](const std::string& path, int fidelity, bool rgb){
        rrt::ObjectAssembly assembly;
        bool success;
        {
            nb::gil_scoped_release release;
            success = load_dispatch(assembly, path, fidelity, rgb);
        }
        if(!success){
            throw std::runtime_error("hyperdrone: failed to load assembly from " + path);
        }
        return assembly;
    }, nb::arg("path"), nb::arg("fidelity") = 2, nb::arg("rgb") = true);

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
}
