#include "environment.h"
#include "bindings.h"
#include "../common/output_bindings.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace HYPERDRONE_MODULE_NAME {
    using HostMask = nb::ndarray<const bool, nb::c_contig, nb::device::cpu>;
    using Mask = nb::ndarray<const bool, nb::c_contig>;
    using Actions = nb::ndarray<const float, nb::c_contig>;
    using FloatOutput = nb::ndarray<float, nb::c_contig>;
    using FlagsOutput = nb::ndarray<bool, nb::c_contig>;

    template <typename ARRAY>
    void check_size(const ARRAY& array, size_t expected){
        if(array.size() != expected){
            throw std::invalid_argument("hyperdrone: buffer size does not match the environment specification");
        }
    }

    template <typename ARRAY>
    bool resident(const Environment& self, const ARRAY& array, size_t expected){
        check_size(array, expected);
        const bool cuda = array.device_type() == nb::device::cuda::value;
        if(!cuda && array.device_type() != nb::device::cpu::value){
            throw std::invalid_argument("hyperdrone: arrays must live in host or CUDA device memory");
        }
        if(cuda && !CUDA_WORLD){
            throw std::invalid_argument("hyperdrone: this environment's World runs on the host (device='cpu'); pass host arrays");
        }
        if(cuda && array.device_id() != self.device_id()){
            throw std::invalid_argument("hyperdrone: CUDA arrays must live on the environment's GPU (device_id)");
        }
        return cuda == CUDA_WORLD;
    }

    nb::dict layout_dict(const ObservationLayout& layout){
        nb::list shape;
        for(TI dimension : layout.shape){
            shape.append(dimension);
        }
        nb::dict blocks;
        for(const ObservationBlock& block : layout.blocks){
            blocks[block.name.c_str()] = nb::make_tuple(block.offset, block.size);
        }
        nb::dict value;
        value["shape"] = nb::tuple(shape);
        value["axis"] = layout.axis;
        value["blocks"] = blocks;
        return value;
    }

    template <void (Environment::*VERB)(float*, bool), TI DIM>
    void bind_output(nb::class_<Environment>& environment_class, const char* name){
        environment_class.def(name, [](Environment& self, FloatOutput out){
            const bool in_place = resident(self, out, TOTAL * DIM);
            nb::gil_scoped_release release;
            (self.*VERB)(out.data(), in_place);
        }, nb::arg("out").noconvert());
    }

    template <typename WORLD_TYPE>
    void bind_imu(nb::module_& m, nb::class_<Environment>& environment_class){
        if constexpr (ObservationIMUDim<WORLD_TYPE>::VALUE > 0){
            m.attr("observation_layout_imu") = layout_dict(observation_layout_imu());
            bind_output<&Environment::observe_imu, ObservationIMUDim<WORLD_TYPE>::VALUE>(environment_class, "observe_imu");
        }
    }
}

NB_MODULE(HYPERDRONE_MODULE_NAME, m){
    using namespace HYPERDRONE_MODULE_NAME;
    m.def("_set_verbosity", &hyperdrone::set_verbosity);

    m.doc() = "JIT-specialized RLtools hyperdrone environment";
    hyperdrone::bind_output_blocks(m);
    m.attr("observation_layout") = layout_dict(observation_layout());
    m.attr("observation_layout_privileged") = layout_dict(observation_layout_privileged());
    m.attr("CUDA_WORLD") = CUDA_WORLD;
    m.attr("render_backend") = rlt::rendering::raytracing::backends::name<typename WORLD::RENDERER::BACKEND>();
    m.attr("synchronized_resets") = SynchronizedResets<WORLD>::VALUE;
    nb::dict dimensions;
    dimensions["num_environments"] = NUM_ENVIRONMENTS;
    dimensions["instances_per_environment"] = WORLD::INSTANCES;
    dimensions["total_instances"] = TOTAL;
    dimensions["num_agents"] = WORLD::N_AGENTS;
    dimensions["num_camera_agents"] = WORLD::N_CAMERA_AGENTS;
    dimensions["num_cameras_per_agent"] = WORLD::N_CAMERAS_PER_AGENT;
    dimensions["num_views"] = WORLD::N_VIEWS;
    dimensions["width"] = WORLD::SPEC::CAM_WIDTH;
    dimensions["height"] = WORLD::SPEC::CAM_HEIGHT;
    dimensions["has_image_observation"] = WORLD::HAS_IMAGE_OBSERVATION;
    dimensions["num_observation_views"] = WORLD::N_OBSERVATION_VIEWS;
    dimensions["image_channels"] = WORLD::IMAGE_CHANNELS;
    dimensions["observation_dim"] = WORLD::OBSERVATION_DIM;
    dimensions["observation_dim_privileged"] = WORLD::OBSERVATION_DIM_PRIVILEGED;
    dimensions["action_dim"] = WORLD::ACTION_DIM;
    dimensions["episode_step_limit"] = WORLD::EPISODE_STEP_LIMIT;
    dimensions["observation_dim_imu"] = OBSERVATION_DIM_IMU;
    dimensions["frame_stride"] = FrameStride<WORLD>::VALUE;
    dimensions["max_drone_types"] = WORLD::SPEC::MAX_DRONES;
    dimensions["self_visible"] = WORLD::SPEC::SELF_VISIBLE;
    m.attr("dimensions") = dimensions;
    auto environment_class = nb::class_<Environment>(m, "Environment");
    environment_class
        .def(nb::init<>())
        .def("init", &Environment::init, nb::arg("scenes"), nb::arg("drone_assets"), nb::arg("drone_parameters"), nb::arg("assets"), nb::arg("seed"), nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("segmentation_taxonomy", &Environment::segmentation_taxonomy)
        .def_prop_ro("segmentation_classes", &Environment::segmentation_classes)
        .def("parameters", &Environment::parameters_json, nb::call_guard<nb::gil_scoped_release>())
        .def("drones", &Environment::drones_json, nb::call_guard<nb::gil_scoped_release>())
        .def("instance_parameters", &Environment::instance_parameters_json, nb::call_guard<nb::gil_scoped_release>())
        .def("spawn_points", &Environment::spawn_points_json, nb::call_guard<nb::gil_scoped_release>())
        .def("set_instance_parameters", [](Environment& self, HostMask mask, const std::string& json){
            check_size(mask, TOTAL);
            nb::gil_scoped_release release;
            self.set_instance_parameters(mask.data(), json);
        }, nb::arg("mask").noconvert(), nb::arg("json"))
        .def("model", &Environment::model, nb::arg("name"), nb::call_guard<nb::gil_scoped_release>())
        .def("rotate_scene", &Environment::rotate_scene, nb::call_guard<nb::gil_scoped_release>())
        .def("reset", [](Environment& self, std::optional<Mask> mask, const std::string& spawn_points){
            const bool in_place = !mask || resident(self, *mask, TOTAL);
            const bool* pointer = mask ? mask->data() : nullptr;
            nb::gil_scoped_release release;
            self.reset(pointer, in_place, spawn_points);
        }, nb::arg("mask").none().noconvert(), nb::arg("spawn_points") = "")
        .def("step", [](Environment& self, Actions actions){
            const bool in_place = resident(self, actions, TOTAL * WORLD::ACTION_DIM);
            nb::gil_scoped_release release;
            self.step(actions.data(), in_place);
        }, nb::arg("actions").noconvert())
        .def("terminated", [](Environment& self, FlagsOutput out){
            const bool in_place = resident(self, out, TOTAL);
            nb::gil_scoped_release release;
            self.terminated(out.data(), in_place);
        }, nb::arg("out").noconvert())
        .def_prop_rw("stream", [](const Environment& self){ return self.stream(); }, [](Environment& self, uintptr_t stream){
            if(!CUDA_WORLD){
                throw std::invalid_argument("hyperdrone: this environment's World runs on the host and has no stream");
            }
            nb::gil_scoped_release release;
            self.set_stream(stream);
        })
        .def_prop_ro("device_id", &Environment::device_id)
        .def("spy_open", &Environment::spy_open, nb::call_guard<nb::gil_scoped_release>())
        .def("spy_update", &Environment::spy_update, nb::call_guard<nb::gil_scoped_release>())
        .def("spy_close", &Environment::spy_close, nb::call_guard<nb::gil_scoped_release>())
        .def("synchronize", &Environment::synchronize, nb::call_guard<nb::gil_scoped_release>())
        .def("render", &Environment::render, nb::call_guard<nb::gil_scoped_release>())
        .def("probe", &Environment::probe, nb::call_guard<nb::gil_scoped_release>())
        .def("output_metadata", [](Environment& self){ return hyperdrone::output_metadata_dicts(self.output_metadata()); })
        .def("output_read", [](Environment& self, size_t index, nb::ndarray<nb::c_contig> out, const std::string& field, uintptr_t stream){
            const auto& metadata = self.output_metadata().at(index);
            hyperdrone::validate_output_array(hyperdrone::output_layout(metadata, field), out);
            if(out.device_type() == 2 && (metadata.device_type != 2 || out.device_id() != metadata.device_id)){
                throw std::invalid_argument("hyperdrone: CUDA output destination must use the renderer's device");
            }
            nb::gil_scoped_release release;
            self.output_read(index, out.data(), out.device_type(), field, stream);
        }, nb::arg("index"), nb::arg("out").noconvert(), nb::arg("field") = "", nb::arg("stream") = 0)
        .def("output_block", [](Environment& self, size_t index, size_t member, const std::string& field){
            OutputBlock block;
            {
                nb::gil_scoped_release release;
                block = self.output_block(index, member, field);
            }
            auto* owner = new std::shared_ptr<void>(std::move(block.owner));
            nb::capsule capsule(owner, [](void* pointer) noexcept { delete static_cast<std::shared_ptr<void>*>(pointer); });
            return hyperdrone::NativeOutputBlock{std::move(capsule), block.pointer, std::move(block.layout), block.device_type, block.device_id, std::move(block.order_consumer)};
        }, nb::arg("index"), nb::arg("block") = 0, nb::arg("field") = "");
    bind_output<&Environment::observe, WORLD::OBSERVATION_DIM>(environment_class, "observe");
    bind_output<&Environment::observe_privileged, WORLD::OBSERVATION_DIM_PRIVILEGED>(environment_class, "observe_privileged");
    bind_output<&Environment::rewards, 1>(environment_class, "rewards");
    bind_imu<WORLD>(m, environment_class);
}
