// the environment's body/prop_* drone rig (rl/environments/hyperdrone/rig) as a producer of the
// renderer's generic RigModel: the render core registers the type, this module instantiates it
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include "../common/verbosity.h"
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/rig/operations_cpu.h>

#include "../render/rig_bindings.h"

#include <optional>
#include <stdexcept>

namespace nb = nanobind;
namespace rlt = rl_tools;

NB_MODULE(hyperdrone_env_core, m){
    using namespace hyperdrone::rig_bindings;
    m.def("_set_verbosity", &hyperdrone::set_verbosity);
    // 63 props plus the base link fill the RigModel capacity exactly, so the rotorcraft is a RigModel
    using ROTORCRAFT = rlt::rl::environments::hyperdrone::rig::Rotorcraft<size_t, RIG_SPEC::MAX_LINKS - 1, RIG_SPEC::MAX_PARTS>;
    static_assert(std::is_same_v<typename ROTORCRAFT::SPEC, RIG_SPEC>);

    m.doc() = "hyperdrone env core: the environment's drone rig for the renderer";
    m.def("rotorcraft_rig", [](const rlt::rendering::ObjectAssembly& assembly, std::optional<hyperdrone::FloatArray> rotor_positions, std::optional<hyperdrone::FloatArray> rotor_torque_directions){
        if(rotor_positions.has_value() != rotor_torque_directions.has_value()){
            throw std::invalid_argument("hyperdrone: rotor_positions and rotor_torque_directions come together");
        }
        ROTORCRAFT rotorcraft;
        if(!rotor_positions.has_value()){
            if(!rlt::init(device(), rotorcraft, assembly)){
                throw std::invalid_argument("hyperdrone: the assembly does not follow the body/prop_* convention: part 0 is \"body\", prop_* parts carry geometry, at most 63 props");
            }
        }
        else{
            const auto& positions = *rotor_positions;
            const auto& torques = *rotor_torque_directions;
            if(positions.ndim() != 2 || positions.shape(1) != 3 || torques.ndim() != 2 || torques.shape(1) != 3 || positions.shape(0) != torques.shape(0)){
                throw std::invalid_argument("hyperdrone: rotor_positions and rotor_torque_directions must both have shape (num_rotors, 3) and dtype float32");
            }
            const bool matched = rlt::init(device(), rotorcraft, assembly,
                                           reinterpret_cast<const float(*)[3]>(positions.data()),
                                           reinterpret_cast<const float(*)[3]>(torques.data()),
                                           positions.shape(0));
            if(!matched){
                throw std::invalid_argument("hyperdrone: the assembly does not follow the body/prop_* convention, or its prop hubs do not pair one-to-one with the rotors (nearest in the XY plane) with nonzero torque directions");
            }
        }
        return RigModel(static_cast<const rlt::rendering::rig::Model<RIG_SPEC>&>(rotorcraft));
    }, nb::arg("assembly"), nb::arg("rotor_positions").none() = nb::none(), nb::arg("rotor_torque_directions").none() = nb::none());
}
