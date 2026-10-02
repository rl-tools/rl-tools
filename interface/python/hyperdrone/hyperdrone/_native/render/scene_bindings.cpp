#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_common.h>
#include <rl_tools/rendering/datasets/glb/operations_cpu.h>

#include "bindings.h"

#include <cmath>
#include <cstdint>
#include <cstring>
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
using rlt::segmentation_taxonomy;
using rlt::assign_segmentation_class;
using rlt::remap_segmentation_classes;

// the Python-facing "Scene" is a Bundle: the composed scene plus its loader/computed
// metadata (bounds, max ray length), which the renderer consumes at init
using SceneBundle = rlt::rendering::Bundle<float>;

template <typename TARGET, typename... MODES>
static bool load_dispatch(TARGET& target, const std::string& path, int fidelity, bool rgb, const MODES&... modes){
    using rlt::load;
    switch(fidelity){
        case 0: return rgb ? load<rrt::Low, true>(g_device, target, path, modes...) : load<rrt::Low, false>(g_device, target, path, modes...);
        case 1: return rgb ? load<rrt::Medium, true>(g_device, target, path, modes...) : load<rrt::Medium, false>(g_device, target, path, modes...);
        case 2: return rgb ? load<rrt::High, true>(g_device, target, path, modes...) : load<rrt::High, false>(g_device, target, path, modes...);
        case 3: return rgb ? load<rrt::VeryHigh, true>(g_device, target, path, modes...) : load<rrt::VeryHigh, false>(g_device, target, path, modes...);
    }
    throw std::invalid_argument("hyperdrone: fidelity must be in [0, 3]");
}

template <typename TARGET>
static std::string taxonomy_name(const TARGET& target){
    const auto taxonomy = segmentation_taxonomy(g_device, target);
    return taxonomy ? taxonomy->name : std::string{};
}

template <typename TARGET>
static std::map<uint32_t, std::string> class_names(const TARGET& target){
    const auto taxonomy = segmentation_taxonomy(g_device, target);
    return taxonomy ? taxonomy->classes : std::map<uint32_t, std::string>{};
}

static std::shared_ptr<const rlt::rendering::SegmentationTaxonomy> make_taxonomy(const std::string& name, const std::map<uint32_t, std::string>& classes){
    if(name.empty() && classes.empty()){
        return nullptr;
    }
    return std::make_shared<const rlt::rendering::SegmentationTaxonomy>(rlt::rendering::SegmentationTaxonomy{name, classes});
}

template <typename TARGET>
static void remap_labels(TARGET& target, const std::map<uint32_t, uint32_t>& mapping, const std::string& taxonomy, const std::map<uint32_t, std::string>& classes){
    remap_segmentation_classes(g_device, target, mapping, make_taxonomy(taxonomy, classes));
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
            rrt::Object value;
            value.name = name;
            assign_segmentation_class(g_device, value, segmentation_class);
            new (object) rrt::Object(std::move(value));
        }, nb::arg("name") = "", nb::arg("segmentation_class") = 0)
        .def_rw("name", &rrt::Object::name)
        .def_prop_rw("segmentation_class", [](const rrt::Object& object){ return object.segmentation_class; },
                     [](rrt::Object& object, uint32_t id){ assign_segmentation_class(g_device, object, id); })
        .def("assign_segmentation_class", [](rrt::Object& object, uint32_t id, const std::string& taxonomy, const std::map<uint32_t, std::string>& classes){
            assign_segmentation_class(g_device, object, id, make_taxonomy(taxonomy, classes));
        }, nb::arg("id"), nb::arg("taxonomy") = "", nb::arg("classes") = std::map<uint32_t, std::string>{},
           "Assign one class and its taxonomy before init; omitted taxonomy selects unnamed numeric classes.")
        .def("remap_segmentation_classes", &remap_labels<rrt::Object>, nb::arg("mapping"), nb::arg("taxonomy") = "", nb::arg("classes") = std::map<uint32_t, std::string>{})
        .def_prop_ro("segmentation_class_valid", [](const rrt::Object& object){ return object.segmentation_class_valid; })
        .def_prop_ro("segmentation_taxonomy", [](const rrt::Object& object){ return object.segmentation_taxonomy ? object.segmentation_taxonomy->name : std::string{}; })
        .def_prop_ro("segmentation_classes", [](const rrt::Object& object){ return object.segmentation_taxonomy ? object.segmentation_taxonomy->classes : std::map<uint32_t, std::string>{}; })
        .def("add_mesh", [](rrt::Object& object, const rrt::Mesh& mesh){ object.meshes.push_back(mesh); })
        .def("add_light", [](rrt::Object& object, const rrt::SceneLight& light){ object.lights.push_back(light); })
        .def_prop_ro("num_meshes", [](const rrt::Object& object){ return object.meshes.size(); });

    nb::class_<rrt::ObjectAssembly>(m, "ObjectAssembly")
        .def(nb::init<>())
        .def_prop_ro("num_objects", [](const rrt::ObjectAssembly& assembly){ return assembly.objects.size(); })
        .def_prop_ro("segmentation_taxonomy", &taxonomy_name<rrt::ObjectAssembly>)
        .def_prop_ro("segmentation_classes", &class_names<rrt::ObjectAssembly>)
        .def_prop_ro("num_parts", [](const rrt::ObjectAssembly& assembly){ return assembly.parts.size(); })
        .def("object_name", [](const rrt::ObjectAssembly& assembly, size_t index){ return assembly.objects.at(index).name; })
        .def("object_names", [](const rrt::ObjectAssembly& assembly){
            std::vector<std::string> names;
            for(const auto& object : assembly.objects){ names.push_back(object.name); }
            return names;
        })
        .def("segmentation_class", [](const rrt::ObjectAssembly& assembly, size_t index){ return assembly.objects.at(index).segmentation_class; })
        .def("set_segmentation_class", [](rrt::ObjectAssembly& assembly, size_t index, uint32_t segmentation_class){
            assign_segmentation_class(g_device, assembly, index, segmentation_class);
        })
        .def("remap_segmentation_classes", &remap_labels<rrt::ObjectAssembly>, nb::arg("mapping"), nb::arg("taxonomy") = "", nb::arg("classes") = std::map<uint32_t, std::string>{})
        .def("part", [](const rrt::ObjectAssembly& assembly, size_t index){
            const auto& part = assembly.parts.at(index);
            return nb::make_tuple(part.object, make_owned_array(part.transform, {3, 4}));
        });

    nb::class_<rrt::AssetPool>(m, "AssetPool")
        .def(nb::init<>())
        .def("add_assembly", [](rrt::AssetPool& pool, const rrt::ObjectAssembly& assembly){ return rlt::add(g_device, pool, assembly).index; })
        .def("add_object", [](rrt::AssetPool& pool, const rrt::Object& object){ return rlt::add(g_device, pool, object).index; })
        .def("add_mesh", [](rrt::AssetPool& pool, const rrt::Mesh& mesh){ return rlt::add(g_device, pool, mesh).index; })
        .def_prop_ro("num_assets", [](const rrt::AssetPool& pool){ return pool.assemblies.size(); })
        .def_prop_ro("segmentation_taxonomy", &taxonomy_name<rrt::AssetPool>)
        .def_prop_ro("segmentation_classes", &class_names<rrt::AssetPool>)
        .def("object_segmentation_class", [](const rrt::AssetPool& pool, size_t asset, size_t object){
            return pool.assemblies.at(asset).objects.at(object).segmentation_class;
        })
        .def("set_object_segmentation_class", [](rrt::AssetPool& pool, size_t asset, size_t object, uint32_t id){
            assign_segmentation_class(g_device, pool, asset, object, id);
        })
        .def("remap_segmentation_classes", &remap_labels<rrt::AssetPool>, nb::arg("mapping"), nb::arg("taxonomy") = "", nb::arg("classes") = std::map<uint32_t, std::string>{})
        .def("remap_asset_segmentation_classes", [](rrt::AssetPool& pool, size_t asset, const std::map<uint32_t, uint32_t>& mapping, const std::string& taxonomy, const std::map<uint32_t, std::string>& classes){
            remap_labels(pool.assemblies.at(asset), mapping, taxonomy, classes);
        }, nb::arg("asset"), nb::arg("mapping"), nb::arg("taxonomy") = "", nb::arg("classes") = std::map<uint32_t, std::string>{});

    nb::class_<SceneBundle>(m, "Scene")
        .def(nb::init<>())
        .def("load", [](SceneBundle& bundle, const std::string& path, int fidelity, bool rgb, bool preserve_instances){
            bool success;
            {
                nb::gil_scoped_release release;
                success = preserve_instances
                    ? load_dispatch(bundle, path, fidelity, rgb, rlt::Mode<rlt::mode::rendering::SceneRoots<>>{})
                    : load_dispatch(bundle, path, fidelity, rgb);
            }
            if(!success){
                throw std::runtime_error("hyperdrone: failed to load scene from " + path);
            }
        }, nb::arg("path"), nb::arg("fidelity") = 2, nb::arg("rgb") = true, nb::arg("preserve_instances") = false)
        .def("add_object", [](SceneBundle& bundle, const rrt::Object& object, nb::object transform){
            if(transform.is_none()){
                return rlt::add(g_device, bundle.scene, object);
            }
            float values[12];
            extract_transform(nb::cast<Transform>(transform), values);
            return rlt::add(g_device, bundle.scene, object, values);
        }, nb::arg("object"), nb::arg("transform") = nb::none())
        .def("add_mesh", [](SceneBundle& bundle, const rrt::Mesh& mesh){ return rlt::add(g_device, bundle.scene, mesh); })
        .def("add_assembly", [](SceneBundle& bundle, const rrt::ObjectAssembly& assembly, nb::object transform){
            rrt::Placement placement;
            if(transform.is_none()){
                placement = rlt::add(g_device, bundle.scene, assembly);
            }
            else {
                float values[12];
                extract_transform(nb::cast<Transform>(transform), values);
                placement = rlt::add(g_device, bundle.scene, assembly, values);
            }
            return nb::make_tuple(placement.first_instance, placement.num_instances);
        }, nb::arg("assembly"), nb::arg("transform") = nb::none())
        .def("add_light", [](SceneBundle& bundle, const rrt::SceneLight& light){ bundle.scene.lights.push_back(light); })
        .def("set_environment", [](SceneBundle& bundle, Vec3 ambient, Vec3 background){
            const auto valid_component = [](float value){
                // Bit inspection keeps NaN/Inf validation effective under -ffast-math.
                uint32_t bits;
                static_assert(sizeof(bits) == sizeof(value));
                std::memcpy(&bits, &value, sizeof(bits));
                volatile uint32_t representation = bits;
                return (representation & 0x7f800000u) != 0x7f800000u && value >= 0.0f;
            };
            for(int component = 0; component < 3; component++){
                if(!valid_component(ambient[component]) || !valid_component(background[component])){
                    throw std::invalid_argument("hyperdrone: environment colors must be finite and nonnegative");
                }
            }
            auto& environment = bundle.scene.environment;
            environment.mode = rrt::Scene::Environment::Mode::SOLID;
            for(int component = 0; component < 3; component++){
                environment.ambient[component] = ambient[component];
                environment.horizon[component] = background[component];
            }
        }, nb::arg("ambient") = Vec3{0.1f, 0.1f, 0.1f}, nb::arg("background") = Vec3{0.0f, 0.0f, 0.0f},
           "Set linear RGB ambient illumination (high/veryhigh) and a solid background before Renderer.init().")
        .def_prop_ro("num_objects", [](const SceneBundle& bundle){ return bundle.scene.objects.size(); })
        .def_prop_ro("num_instances", [](const SceneBundle& bundle){ return bundle.scene.instances.size(); })
        .def_prop_ro("num_lights", [](const SceneBundle& bundle){ return bundle.scene.lights.size(); })
        .def_prop_ro("segmentation_taxonomy", [](const SceneBundle& bundle){ return taxonomy_name(bundle.scene); })
        .def_prop_ro("segmentation_classes", [](const SceneBundle& bundle){ return class_names(bundle.scene); })
        .def("object_name", [](const SceneBundle& bundle, size_t index){ return bundle.scene.objects.at(index).name; })
        .def("object_segmentation_class", [](const SceneBundle& bundle, size_t index){ return bundle.scene.objects.at(index).segmentation_class; })
        .def("set_object_segmentation_class", [](SceneBundle& bundle, size_t index, uint32_t segmentation_class){
            assign_segmentation_class(g_device, bundle.scene, index, segmentation_class);
        })
        .def("remap_segmentation_classes", [](SceneBundle& bundle, const std::map<uint32_t, uint32_t>& mapping, const std::string& taxonomy, const std::map<uint32_t, std::string>& classes){
            remap_labels(bundle.scene, mapping, taxonomy, classes);
        }, nb::arg("mapping"), nb::arg("taxonomy") = "", nb::arg("classes") = std::map<uint32_t, std::string>{})
        .def("instance_object", [](const SceneBundle& bundle, size_t index){ return bundle.scene.instances.at(index).object; })
        .def("instance_objects", [](const SceneBundle& bundle){
            std::vector<size_t> objects;
            for(const auto& instance : bundle.scene.instances){ objects.push_back(instance.object); }
            return objects;
        })
        .def("instance_transform", [](const SceneBundle& bundle, size_t index){
            return make_owned_array(bundle.scene.instances.at(index).transform, {3, 4});
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
