// body/prop_* rigging of hyperlux assemblies: articulate writes the joints, rotorcraft pairs
// the props with a dynamics model's rotors; the assembly type is hyperlux's, bound by its core
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "verbosity.h"
#include "bindings.h"
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/hyperlux/operations_cpu_common.h>
#include <rl_tools/rl/environments/hyperdrone/rotorcraft/operations_cpu.h>

#include <optional>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
namespace rlt = rl_tools;

namespace {
    using DEVICE = rlt::devices::DEVICE_FACTORY<hyperlux::CPUSpecification>;
    DEVICE g_device;
}

NB_MODULE(hyperdrone_rotorcraft_core, m){
    m.def("_set_verbosity", &hyperlux::set_verbosity);
    m.def("articulate", [](rlt::hyperlux::ObjectAssembly& assembly){
        if(!rlt::articulate(g_device, assembly)){
            throw std::invalid_argument("hyperdrone: the assembly does not follow the body/prop_* convention: part 0 is \"body\", prop_* parts carry geometry");
        }
    }, nb::arg("assembly"));
    m.def("rotorcraft", [](const rlt::hyperlux::ObjectAssembly& assembly, std::optional<hyperlux::FloatArray> rotor_positions, std::optional<hyperlux::FloatArray> rotor_torque_directions){
        if(rotor_positions.has_value() != rotor_torque_directions.has_value()){
            throw std::invalid_argument("hyperdrone: rotor_positions and rotor_torque_directions come together");
        }
        rlt::rl::environments::hyperdrone::Rotorcraft rotorcraft;
        if(!rotor_positions.has_value()){
            if(!rlt::init(g_device, rotorcraft, assembly)){
                throw std::invalid_argument("hyperdrone: the assembly carries no rotorcraft joints; call hyperdrone.articulate(assembly) first");
            }
        }
        else{
            const auto& positions = *rotor_positions;
            const auto& torques = *rotor_torque_directions;
            if(positions.ndim() != 2 || positions.shape(1) != 3 || torques.ndim() != 2 || torques.shape(1) != 3 || positions.shape(0) != torques.shape(0)){
                throw std::invalid_argument("hyperdrone: rotor_positions and rotor_torque_directions must both have shape (num_rotors, 3) and dtype float32");
            }
            if(!rlt::init(g_device, rotorcraft, assembly, reinterpret_cast<const float(*)[3]>(positions.data()), reinterpret_cast<const float(*)[3]>(torques.data()), positions.shape(0))){
                throw std::invalid_argument("hyperdrone: the assembly carries no rotorcraft joints (call hyperdrone.articulate(assembly) first), or its prop hubs do not pair one-to-one with the rotors (nearest in the XY plane) with nonzero torque directions");
            }
        }
        return nb::make_tuple(rotorcraft.wheelbase, rotorcraft.scale, rotorcraft.rotor_position_deviation, std::vector<size_t>(rotorcraft.prop_rotor, rotorcraft.prop_rotor + rotorcraft.num_props), std::vector<float>(rotorcraft.prop_spin, rotorcraft.prop_spin + rotorcraft.num_props));
    }, nb::arg("assembly"), nb::arg("rotor_positions").none(), nb::arg("rotor_torque_directions").none());
}
