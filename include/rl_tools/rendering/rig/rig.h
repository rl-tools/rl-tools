#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RIG_RIG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RIG_RIG_H

#include "../../rl_tools.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::rig {
    enum class Joint { FIXED, REVOLUTE, PRISMATIC, SPHERICAL };

    // Rigid-link tree over 3x4 row-major [R|t] transforms, links declared parent before child:
    // world_link = world_parent ∘ origin ∘ joint(coordinates), world_geometry = world_link ∘ offset.
    // Fixed capacity keeps the model a plain value, so a kernel parameter, a copied payload and a
    // host object are one type. Revolute and prismatic joints take one coordinate (radians,
    // metres) about/along the unit axis in the joint frame, spherical joints a wxyz quaternion.
    template <typename T_T, typename T_TI, T_TI T_MAX_LINKS, T_TI T_MAX_PARTS>
    struct Specification {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI MAX_LINKS = T_MAX_LINKS;
        static constexpr TI MAX_PARTS = T_MAX_PARTS;
        static constexpr TI MAX_COORDINATES = 4 * MAX_LINKS;
        static constexpr TI BASE = ~(TI)0; // parent of a root link
    };

    template <typename T_SPEC>
    struct Link {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        TI parent = SPEC::BASE;
        Joint joint = Joint::FIXED;
        T axis[3] = {0, 0, 1};
        T origin[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; // parent link frame -> joint frame
        TI coordinate = 0;                                   // first coordinate, assigned by init
    };

    template <typename T_SPEC>
    struct Part {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        TI link = 0;
        T offset[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}; // link frame -> geometry frame
    };

    template <typename T_SPEC>
    struct Model {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        Link<SPEC> links[SPEC::MAX_LINKS];
        Part<SPEC> parts[SPEC::MAX_PARTS];
        TI num_links = 0;
        TI num_parts = 0;
        TI num_coordinates = 0;
    };

    template <typename T_SPEC>
    struct Buffer {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        T link_poses[SPEC::MAX_LINKS][12];
        T coordinates[SPEC::MAX_COORDINATES];
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
