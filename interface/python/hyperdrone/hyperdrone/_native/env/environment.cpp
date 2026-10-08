#include "environment.h"
#include "bindings.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace HYPERDRONE_MODULE_NAME {
    using FloatOutput = nb::ndarray<float, nb::c_contig, nb::device::cpu>;
    using Mask = nb::ndarray<const bool, nb::c_contig, nb::device::cpu>;
    using Flags = nb::ndarray<bool, nb::c_contig, nb::device::cpu>;

    template <typename ARRAY>
    void check_size(const ARRAY& array, size_t expected){
        if(array.size() != expected){
            throw std::invalid_argument("hyperdrone: buffer size does not match the environment specification");
        }
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

    template <typename WORLD_TYPE>
    void bind_imu(nb::module_& m, nb::class_<Environment>& environment_class){
        if constexpr (ObservationIMUDim<WORLD_TYPE>::VALUE > 0){
            m.attr("observation_layout_imu") = layout_dict(observation_layout_imu());
            environment_class.def("observe_imu", [](Environment& self, FloatOutput out){
                check_size(out, TOTAL * ObservationIMUDim<WORLD_TYPE>::VALUE);
                nb::gil_scoped_release release;
                observe_imu<WORLD_TYPE>(self, out.data());
            }, nb::arg("out").noconvert());
        }
    }
}

NB_MODULE(HYPERDRONE_MODULE_NAME, m){
    using namespace HYPERDRONE_MODULE_NAME;

    m.doc() = "JIT-specialized RLtools hyperdrone environment";
    m.attr("observation_layout") = layout_dict(observation_layout());
    m.attr("observation_layout_privileged") = layout_dict(observation_layout_privileged());
    nb::dict dimensions;
    dimensions["num_environments"] = NUM_ENVIRONMENTS;
    dimensions["instances_per_environment"] = WORLD::INSTANCES;
    dimensions["total_instances"] = TOTAL;
    dimensions["n_agents"] = WORLD::N_AGENTS;
    dimensions["cam_width"] = WORLD::SPEC::CAM_WIDTH;
    dimensions["cam_height"] = WORLD::SPEC::CAM_HEIGHT;
    dimensions["image_channels"] = WORLD::IMAGE_CHANNELS;
    dimensions["observation_dim"] = WORLD::OBSERVATION_DIM;
    dimensions["observation_dim_privileged"] = WORLD::OBSERVATION_DIM_PRIVILEGED;
    dimensions["action_dim"] = WORLD::ACTION_DIM;
    dimensions["episode_step_limit"] = WORLD::EPISODE_STEP_LIMIT;
    dimensions["observation_dim_imu"] = OBSERVATION_DIM_IMU;
    dimensions["frame_stride"] = FrameStride<WORLD>::VALUE;
    dimensions["max_drones"] = WORLD::SPEC::MAX_DRONES;
    dimensions["self_visible"] = WORLD::SPEC::SELF_VISIBLE;
    m.attr("dimensions") = dimensions;

    auto environment_class = nb::class_<Environment>(m, "Environment");
    environment_class
        .def(nb::init<>())
        .def("init", &Environment::init, nb::arg("scenes"), nb::arg("drone_assets"), nb::arg("drone_parameters"), nb::arg("gate_asset"), nb::arg("seed"), nb::call_guard<nb::gil_scoped_release>())
        .def("parameters", &Environment::parameters_json, nb::call_guard<nb::gil_scoped_release>())
        .def("drones", &Environment::drones_json, nb::call_guard<nb::gil_scoped_release>())
        .def("instance_parameters", &Environment::instance_parameters_json, nb::call_guard<nb::gil_scoped_release>())
        .def("set_instance_parameters", [](Environment& self, Mask mask, const std::string& json){
            check_size(mask, TOTAL);
            nb::gil_scoped_release release;
            self.set_instance_parameters(mask.data(), json);
        }, nb::arg("mask").noconvert(), nb::arg("json"))
        .def("model", &Environment::model, nb::arg("name"), nb::call_guard<nb::gil_scoped_release>())
        .def("reset", [](Environment& self, Mask mask){
            check_size(mask, TOTAL);
            nb::gil_scoped_release release;
            self.reset(mask.data());
        }, nb::arg("mask").noconvert())
        .def("render", [](Environment& self, Mask mask){
            check_size(mask, TOTAL);
            nb::gil_scoped_release release;
            self.render(mask.data());
        }, nb::arg("mask").noconvert())
        .def("observe", [](Environment& self, FloatOutput out){
            check_size(out, TOTAL * WORLD::OBSERVATION_DIM);
            nb::gil_scoped_release release;
            self.observe(out.data());
        }, nb::arg("out").noconvert())
        .def("observe_privileged", [](Environment& self, FloatOutput out){
            check_size(out, TOTAL * WORLD::OBSERVATION_DIM_PRIVILEGED);
            nb::gil_scoped_release release;
            self.observe_privileged(out.data());
        }, nb::arg("out").noconvert())
        .def("step", [](Environment& self, hyperdrone::FloatArray actions){
            check_size(actions, TOTAL * WORLD::ACTION_DIM);
            nb::gil_scoped_release release;
            self.step(actions.data());
        }, nb::arg("actions").noconvert())
        .def("rewards", [](Environment& self, FloatOutput out){
            check_size(out, TOTAL);
            nb::gil_scoped_release release;
            self.read_rewards(out.data());
        }, nb::arg("out").noconvert())
        .def("terminated", [](Environment& self, Flags out){
            check_size(out, TOTAL);
            nb::gil_scoped_release release;
            self.read_terminated(out.data());
        }, nb::arg("out").noconvert())
        .def("rotate_scene", &Environment::rotate_scene, nb::call_guard<nb::gil_scoped_release>());
    bind_imu<WORLD>(m, environment_class);
}
