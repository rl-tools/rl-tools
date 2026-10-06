#include "environment.h"
#include "bindings.h"
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace HYPERDRONE_MODULE_NAME {
    std::string canonical_config(){
        std::string value =
            std::string("HYPERDRONE_ENV_CAM_HEIGHT=") + std::to_string(HYPERDRONE_ENV_CAM_HEIGHT)
            + " HYPERDRONE_ENV_CAM_WIDTH=" + std::to_string(HYPERDRONE_ENV_CAM_WIDTH)
            + " HYPERDRONE_ENV_HISTORY_LENGTH=" + std::to_string(HYPERDRONE_ENV_HISTORY_LENGTH)
            + " HYPERDRONE_ENV_INSTANCES=" + std::to_string(HYPERDRONE_ENV_INSTANCES)
            + " HYPERDRONE_ENV_NUM_ENVIRONMENTS=" + std::to_string(HYPERDRONE_ENV_NUM_ENVIRONMENTS)
            + " HYPERDRONE_ENV_N_AGENTS=" + std::to_string(HYPERDRONE_ENV_N_AGENTS)
            + " HYPERDRONE_ENV_PRESET=" + std::to_string(HYPERDRONE_ENV_PRESET)
            + " HYPERDRONE_ENV_SHADING=" + std::to_string(HYPERDRONE_ENV_SHADING);
#ifdef HYPERDRONE_ENV_SPEC_HEADER
        value += " HYPERDRONE_ENV_SPEC_HEADER=" HYPERDRONE_ENV_STRINGIFY(HYPERDRONE_ENV_SPEC_HEADER);
#endif
        value += " HYPERDRONE_ENV_TASK=" + std::to_string(HYPERDRONE_ENV_TASK);
#ifdef HYPERDRONE_ENV_SPEC_HEADER
        value += " spec_hash=" HYPERDRONE_ENV_STRINGIFY(HYPERDRONE_ENV_SPEC_HASH);
#endif
        return value;
    }

    template <typename ARRAY>
    void check_size(const ARRAY& array, size_t expected){
        if(array.size() != expected){
            throw std::invalid_argument("hyperdrone: buffer size does not match the environment specification");
        }
    }
}

NB_MODULE(HYPERDRONE_MODULE_NAME, m){
    using namespace HYPERDRONE_MODULE_NAME;
    using FloatOutput = nb::ndarray<float, nb::c_contig, nb::device::cpu>;
    using Mask = nb::ndarray<const uint8_t, nb::c_contig, nb::device::cpu>;
    using Flags = nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu>;

    m.doc() = "JIT-specialized RLtools hyperdrone environment";
    m.attr("config") = canonical_config();
    m.attr("config_string") = config_string();
    m.attr("observation_layout") = observation_layout_value(false);
    m.attr("observation_layout_privileged") = observation_layout_value(true);
    m.attr("observation_layout_imu") = observation_layout_imu();
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
    m.attr("dimensions") = dimensions;

    nb::class_<Environment>(m, "Environment")
        .def(nb::init<>())
        .def("init", &Environment::init, nb::arg("scenes"), nb::arg("drone_asset"), nb::arg("gate_asset"), nb::arg("seed"), nb::call_guard<nb::gil_scoped_release>())
        .def("parameters", &Environment::parameters_json, nb::call_guard<nb::gil_scoped_release>())
        .def("set_parameters", &Environment::set_parameters, nb::arg("json"), nb::call_guard<nb::gil_scoped_release>())
        .def("instance_parameters", &Environment::instance_parameters_json, nb::call_guard<nb::gil_scoped_release>())
        .def("set_instance_parameters", [](Environment& self, Mask mask, const std::string& json){
            check_size(mask, TOTAL);
            nb::gil_scoped_release release;
            return self.set_instance_parameters(mask.data(), json);
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
        .def("observe_imu", [](Environment& self, FloatOutput out){
            check_size(out, TOTAL * OBSERVATION_DIM_IMU);
            nb::gil_scoped_release release;
            self.observe_imu(out.data());
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
}
