#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "iface.h"
#include "jit_host.h"

#include <cstring>
#include <stdexcept>
#include <string>

#if defined(HYPERDRONE_DYNAMICS_CUDA)
#include "cuda_staging.h"
#include <cuda_runtime.h>
#endif

namespace nb = nanobind;
namespace hdd = hyperdrone::dynamics;

struct JitSim {
    hyperdrone::JitInstance<hdd::Sim> instance;

    JitSim(const std::string& path, const std::string& expected_config)
        : instance(path, "hyperdrone_dynamics", HYPERDRONE_DYNAMICS_IFACE_VERSION, expected_config) {}

    hdd::Sim* operator->() const { return instance.instance; }

    size_t component_dim(int component) const {
        if(component < 0 || component > 4){
            throw std::invalid_argument("hyperdrone: unknown state component");
        }
        return component == 4 ? instance.instance->config().action_dim
                              : (size_t)hdd::state_component_dims[component];
    }

    void copy_out(const float* source, float* destination, size_t count) const {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
        instance.instance->synchronize();
        cudaMemcpy(destination, source, count * sizeof(float), cudaMemcpyDeviceToHost);
#else
        std::memcpy(destination, source, count * sizeof(float));
#endif
    }
};

using HostFloatArray = nb::ndarray<const float, nb::c_contig, nb::device::cpu>;

NB_MODULE(hyperdrone_dynamics_core, m){
    m.doc() = "hyperdrone dynamics core: JIT loader for the vectorized L2F multirotor simulator";

    auto jit_sim_class = nb::class_<JitSim>(m, "JitSim");
    jit_sim_class
        .def(nb::init<const std::string&, const std::string&>(), nb::arg("library_path"), nb::arg("expected_config"))
        .def_prop_ro("num_drones", [](const JitSim& jit){ return jit->config().num_drones; })
        .def_prop_ro("action_dim", [](const JitSim& jit){ return jit->config().action_dim; })
        .def_prop_ro("observation_dim", [](const JitSim& jit){ return jit->config().observation_dim; })
        .def_prop_ro("domain_randomization", [](const JitSim& jit){ return jit->config().domain_randomization; })
        .def_prop_ro("device_name", [](const JitSim& jit){ return std::string(jit->device_name()); })
        .def_prop_ro("buffer_device_type", [](const JitSim& jit){ return jit->buffer_device_type(); })
        .def_prop_ro("stream", [](const JitSim& jit){ return (uintptr_t)jit->stream(); })
        .def("synchronize", [](JitSim& jit){
            nb::gil_scoped_release release;
            jit->synchronize();
        })
        .def("set_model", [](JitSim& jit, const std::string& name){ return jit->set_model(name.c_str()); }, nb::arg("name"))
        .def("set_dt", [](JitSim& jit, float dt){ jit->set_dt(dt); }, nb::arg("dt"))
        .def_prop_ro("dt", [](const JitSim& jit){ return jit->dt(); })
        .def("reset", [](JitSim& jit, uint64_t seed, bool sample_parameters, bool sample_states){
            nb::gil_scoped_release release;
            jit->reset(seed, sample_parameters, sample_states);
        }, nb::arg("seed"), nb::arg("sample_parameters"), nb::arg("sample_states"))
        .def("step", [](JitSim& jit, HostFloatArray actions){
            const hdd::Config config = jit->config();
            if(actions.ndim() != 2 || actions.shape(0) != config.num_drones || actions.shape(1) != config.action_dim){
                throw std::invalid_argument("hyperdrone: actions must have shape (num_drones, action_dim) and dtype float32");
            }
            nb::gil_scoped_release release;
            jit->step(actions.data(), 1, 0);
        }, nb::arg("actions"))
        .def("state_dlpack", [](JitSim& jit, int component){
            const hdd::Config config = jit->config();
            const size_t dim = jit.component_dim(component);
            return nb::ndarray<>((void*)jit->state_buffer(component), {config.num_drones, dim}, nb::find(&jit), {},
                                 nb::dtype<float>(), jit->buffer_device_type(), 0);
        }, nb::arg("component"))
        .def("read_state", [](JitSim& jit, int component, nb::ndarray<float, nb::c_contig, nb::device::cpu> out){
            const hdd::Config config = jit->config();
            const size_t dim = jit.component_dim(component);
            if(out.size() != config.num_drones * dim){
                throw std::invalid_argument("hyperdrone: output array has the wrong size for this state component");
            }
            nb::gil_scoped_release release;
            jit.copy_out(jit->state_buffer(component), out.data(), config.num_drones * dim);
        }, nb::arg("component"), nb::arg("out"))
        .def("set_state", [](JitSim& jit, int component, HostFloatArray values){
            const hdd::Config config = jit->config();
            const size_t dim = jit.component_dim(component);
            if(values.ndim() != 2 || values.shape(0) != config.num_drones || values.shape(1) != dim){
                throw std::invalid_argument("hyperdrone: state component values must have shape (num_drones, dim) and dtype float32");
            }
            nb::gil_scoped_release release;
            jit->set_state_component(component, values.data());
        }, nb::arg("component"), nb::arg("values"))
        .def("update_observations", [](JitSim& jit){
            nb::gil_scoped_release release;
            jit->update_observations();
        })
        .def("observations_dlpack", [](JitSim& jit){
            const hdd::Config config = jit->config();
            return nb::ndarray<>((void*)jit->observation_buffer(), {config.num_drones, config.observation_dim}, nb::find(&jit), {},
                                 nb::dtype<float>(), jit->buffer_device_type(), 0);
        })
        .def("read_observations", [](JitSim& jit, nb::ndarray<float, nb::c_contig, nb::device::cpu> out){
            const hdd::Config config = jit->config();
            if(out.size() != (size_t)config.num_drones * config.observation_dim){
                throw std::invalid_argument("hyperdrone: observations output must have num_drones * observation_dim elements");
            }
            nb::gil_scoped_release release;
            jit.copy_out(jit->observation_buffer(), out.data(), (size_t)config.num_drones * config.observation_dim);
        }, nb::arg("out"))
        .def("update_camera_bases", [](JitSim& jit, HostFloatArray mount, float fov, float aspect){
            if(mount.size() != 12){
                throw std::invalid_argument("hyperdrone: mount transform must have 12 elements (3x4 row-major)");
            }
            nb::gil_scoped_release release;
            jit->update_camera_bases(mount.data(), fov, aspect);
        }, nb::arg("mount"), nb::arg("fov"), nb::arg("aspect"))
        .def("camera_bases_dlpack", [](JitSim& jit){
            const hdd::Config config = jit->config();
            return nb::ndarray<>((void*)jit->camera_bases_buffer(), {config.num_drones, (size_t)12}, nb::find(&jit), {},
                                 nb::dtype<float>(), jit->buffer_device_type(), 0);
        })
        .def("read_camera_bases", [](JitSim& jit, nb::ndarray<float, nb::c_contig, nb::device::cpu> out){
            const hdd::Config config = jit->config();
            if(out.size() != (size_t)config.num_drones * 12){
                throw std::invalid_argument("hyperdrone: camera bases output must have num_drones * 12 elements");
            }
            nb::gil_scoped_release release;
            jit.copy_out(jit->camera_bases_buffer(), out.data(), (size_t)config.num_drones * 12);
        }, nb::arg("out"))
        .def("read_parameter", [](JitSim& jit, const std::string& name, nb::ndarray<float, nb::c_contig, nb::device::cpu> out){
            if(out.size() != jit->config().num_drones){
                throw std::invalid_argument("hyperdrone: parameter output must have num_drones elements");
            }
            if(!jit->read_parameter(name.c_str(), out.data())){
                throw std::invalid_argument("hyperdrone: unknown parameter " + name);
            }
        }, nb::arg("name"), nb::arg("out"))
        .def("write_parameter", [](JitSim& jit, const std::string& name, HostFloatArray values){
            if(values.size() != jit->config().num_drones){
                throw std::invalid_argument("hyperdrone: parameter values must have num_drones elements");
            }
            if(!jit->write_parameter(name.c_str(), values.data())){
                throw std::invalid_argument("hyperdrone: unknown parameter " + name);
            }
        }, nb::arg("name"), nb::arg("values"));

#if defined(HYPERDRONE_DYNAMICS_CUDA)
    m.attr("HAS_CUDA") = true;

    hyperdrone::register_cuda_staging(m);

    jit_sim_class
        .def("step_device", [](JitSim& jit, nb::ndarray<const float, nb::c_contig, nb::device::cuda> actions, uintptr_t stream){
            const hdd::Config config = jit->config();
            if(actions.ndim() != 2 || actions.shape(0) != config.num_drones || actions.shape(1) != config.action_dim){
                throw std::invalid_argument("hyperdrone: actions must have shape (num_drones, action_dim) and dtype float32");
            }
            jit->step(actions.data(), 2, stream);
        }, nb::arg("actions"), nb::arg("stream") = 0);
#else
    m.attr("HAS_CUDA") = false;
#endif
}
