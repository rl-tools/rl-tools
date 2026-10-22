#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_PRIMITIVES_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_PRIMITIVES_OPERATIONS_CPU_H

#include "../scene.h"

#include <stdexcept>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE>
    hyperlux::Object make_box(DEVICE&, const float size[3], const float color[3]){
        for(unsigned axis = 0; axis < 3; axis++){
            if(!hyperlux::is_finite(size[axis]) || size[axis] <= 0){
                throw std::invalid_argument("box size must contain three finite positive dimensions");
            }
            if(!hyperlux::is_finite(color[axis])){
                throw std::invalid_argument("box color must contain three finite components");
            }
        }
        hyperlux::Object object;
        object.meshes.emplace_back();
        auto& mesh = object.meshes.back();
        mesh.vertices.resize(24 * 3);
        mesh.normals.resize(24 * 3, 0);
        mesh.indices.resize(12 * 3);
        for(unsigned axis = 0; axis < 3; axis++){
            mesh.color[axis] = color[axis];
        }
        constexpr int CORNERS[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        constexpr int TRIANGLES[2][6] = {{0, 2, 1, 0, 3, 2}, {0, 1, 2, 0, 2, 3}};
        for(unsigned face = 0; face < 6; face++){
            const unsigned axis = face / 2;
            const unsigned u = (axis + 1) % 3;
            const unsigned v = (axis + 2) % 3;
            const float sign = face % 2 == 0 ? -1.0f : 1.0f;
            for(unsigned corner = 0; corner < 4; corner++){
                const unsigned vertex = (face * 4 + corner) * 3;
                mesh.vertices[vertex + axis] = sign * size[axis] * 0.5f;
                mesh.vertices[vertex + u] = CORNERS[corner][0] * size[u] * 0.5f;
                mesh.vertices[vertex + v] = CORNERS[corner][1] * size[v] * 0.5f;
                mesh.normals[vertex + axis] = sign;
            }
            for(unsigned index = 0; index < 6; index++){
                mesh.indices[face * 6 + index] = static_cast<int>(face * 4) + TRIANGLES[face % 2][index];
            }
        }
        return object;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
