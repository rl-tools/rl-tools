#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_COLLISION_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_VISUAL_INERTIAL_LOCALIZATION_COLLISION_OPERATIONS_CPU_H

#include "collision.h"
#include "../../../../../hyperlux/scene.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rl::environments::hyperdrone::tasks::visual_inertial_localization::collision::detail {
        using hyperlux::backends::generic::Vec3;
        using hyperlux::backends::generic::cross;
        using hyperlux::backends::generic::dot;
        using hyperlux::backends::generic::to_vec3;

        template <typename T>
        bool finite(const T& value) {
            static_assert(std::is_integral_v<T> || std::is_same_v<T, float> || std::is_same_v<T, double>);
            // Bit inspection remains valid when callers build the simulator with fast-math.
            if constexpr (std::is_integral_v<T>) {
                return true;
            } else if constexpr (std::is_same_v<T, float>) {
                std::uint32_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
            } else {
                std::uint64_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
            }
        }

        template <typename T>
        T absolute(T value) {
            return value < (T)0 ? -value : value;
        }

        template <typename T>
        T clamp(T value) {
            return std::max((T)0, std::min((T)1, value));
        }

        template <typename T>
        T point_segment_distance_squared(Vec3<T> point, Vec3<T> from, Vec3<T> to) {
            const auto edge = to - from;
            const T length_squared = dot(edge, edge);
            const T fraction = length_squared > (T)0 ? clamp(dot(point - from, edge) / length_squared) : (T)0;
            const auto residual = point - (from + fraction * edge);
            return dot(residual, residual);
        }

        template <typename T>
        T segment_distance_squared(Vec3<T> a, Vec3<T> b, Vec3<T> c, Vec3<T> d) {
            const auto u = b - a;
            const auto v = d - c;
            const auto w = a - c;
            const T uu = dot(u, u), vv = dot(v, v), uv = dot(u, v);
            const T uw = dot(u, w), vw = dot(v, w);
            if (uu == (T)0) {
                return point_segment_distance_squared(a, c, d);
            }
            if (vv == (T)0) {
                return point_segment_distance_squared(c, a, b);
            }
            const T denominator = uu * vv - uv * uv;
            T s = denominator > (T)0 ? clamp((uv * vw - uw * vv) / denominator) : (T)0;
            T t = (uv * s + vw) / vv;
            if (t < (T)0) {
                t = 0;
                s = clamp(-uw / uu);
            } else if (t > (T)1) {
                t = 1;
                s = clamp((uv - uw) / uu);
            }
            const auto residual = w + s * u - t * v;
            return dot(residual, residual);
        }

        template <typename T>
        bool inside_triangle(Vec3<T> point, Vec3<T> a, Vec3<T> b, Vec3<T> c, Vec3<T> normal) {
            const T tolerance = (T)64 * std::numeric_limits<T>::epsilon() * dot(normal, normal);
            return dot(cross(b - a, point - a), normal) >= -tolerance &&
                   dot(cross(c - b, point - b), normal) >= -tolerance &&
                   dot(cross(a - c, point - c), normal) >= -tolerance;
        }

        template <typename T>
        T point_triangle_distance_squared(Vec3<T> point, Vec3<T> a, Vec3<T> b, Vec3<T> c, Vec3<T> normal) {
            T result = std::min({point_segment_distance_squared(point, a, b), point_segment_distance_squared(point, b, c), point_segment_distance_squared(point, c, a)});
            const T normal_squared = dot(normal, normal);
            if (normal_squared > (T)0) {
                const T distance = dot(point - a, normal);
                const auto projection = point - (distance / normal_squared) * normal;
                if (inside_triangle(projection, a, b, c, normal)) {
                    result = std::min(result, distance * distance / normal_squared);
                }
            }
            return result;
        }

        template <typename T>
        bool intersects(const Triangle<T>& triangle, const T from[3], const T to[3], T radius) {
            T offsets[4][3];
            T scale = radius;
            for (int axis = 0; axis < 3; axis++) {
                offsets[0][axis] = triangle.vertices[1][axis] - triangle.vertices[0][axis];
                offsets[1][axis] = triangle.vertices[2][axis] - triangle.vertices[0][axis];
                offsets[2][axis] = from[axis] - triangle.vertices[0][axis];
                offsets[3][axis] = to[axis] - triangle.vertices[0][axis];
                for (int point = 0; point < 4; point++) {
                    if (!finite(offsets[point][axis])) {
                        return true;
                    }
                    scale = std::max(scale, absolute(offsets[point][axis]));
                }
            }
            if (scale == (T)0) {
                return true;
            }
            for (auto& point : offsets) {
                for (T& value : point) {
                    value /= scale;
                }
            }
            const Vec3<T> a{0, 0, 0};
            const auto b = to_vec3(offsets[0]), c = to_vec3(offsets[1]);
            const auto p = to_vec3(offsets[2]), q = to_vec3(offsets[3]);
            const auto normal = cross(b, c);
            const T d0 = dot(p, normal), d1 = dot(q, normal);
            if (((d0 <= (T)0 && d1 >= (T)0) || (d0 >= (T)0 && d1 <= (T)0)) && d0 != d1) {
                const auto point = p + (d0 / (d0 - d1)) * (q - p);
                if (inside_triangle(point, a, b, c, normal)) {
                    return true;
                }
            }
            const T distance_squared = std::min({
                point_triangle_distance_squared(p, a, b, c, normal),
                point_triangle_distance_squared(q, a, b, c, normal),
                segment_distance_squared(p, q, a, b),
                segment_distance_squared(p, q, b, c),
                segment_distance_squared(p, q, c, a)
            });
            const T threshold = radius / scale + (T)64 * std::numeric_limits<T>::epsilon();
            return !finite(distance_squared) || distance_squared <= threshold * threshold;
        }

        template <typename T, typename TI>
        bool overlaps(const hyperlux::backends::generic::BVHNode<T, TI>& node, const T from[3], const T to[3], T radius) {
            T first = 0, last = 1;
            for (int axis = 0; axis < 3; axis++) {
                const T magnitude = std::max({(T)1, absolute(from[axis]), absolute(to[axis]), absolute(node.bounds_min[axis]), absolute(node.bounds_max[axis])});
                const T padding = radius + (T)64 * std::numeric_limits<T>::epsilon() * magnitude;
                const T low = node.bounds_min[axis] - padding, high = node.bounds_max[axis] + padding;
                const T direction = to[axis] - from[axis];
                if (!finite(direction) || !finite(low) || !finite(high)) {
                    return true;
                }
                if (direction == (T)0) {
                    if (from[axis] < low || from[axis] > high) {
                        return false;
                    }
                } else {
                    T near = (low - from[axis]) / direction, far = (high - from[axis]) / direction;
                    if (!finite(near) || !finite(far)) {
                        continue;
                    }
                    if (near > far) {
                        std::swap(near, far);
                    }
                    first = std::max(first, near);
                    last = std::min(last, far);
                    if (first > last) {
                        return false;
                    }
                }
            }
            return true;
        }
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE&, rl::environments::hyperdrone::tasks::visual_inertial_localization::collision::Scene<SPEC>& scene) {
        decltype(scene.triangles){}.swap(scene.triangles);
        decltype(scene.nodes){}.swap(scene.nodes);
        decltype(scene.primitives){}.swap(scene.primitives);
        scene.valid = false;
    }

    template <typename DEVICE, typename SPEC>
    bool init(DEVICE& device, rl::environments::hyperdrone::tasks::visual_inertial_localization::collision::Scene<SPEC>& target, const hyperlux::Scene& source) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        namespace collision = rl::environments::hyperdrone::tasks::visual_inertial_localization::collision;
        free(device, target);
        for (const auto& instance : source.instances) {
            if (instance.object >= source.objects.size()) {
                return false;
            }
            for (const float& value : instance.transform) {
                if (!collision::detail::finite(value)) {
                    return false;
                }
            }
            for (const auto& mesh : source.objects[instance.object].meshes) {
                if (mesh.indices.size() % 3 != 0 || mesh.vertices.size() % 3 != 0) {
                    return false;
                }
                for (const float& value : mesh.vertices) {
                    if (!collision::detail::finite(value)) {
                        return false;
                    }
                }
                for (std::size_t index = 0; index < mesh.indices.size(); index += 3) {
                    collision::Triangle<T> triangle;
                    for (int vertex = 0; vertex < 3; vertex++) {
                        const int local = mesh.indices[index + vertex];
                        if (local < 0 || static_cast<std::size_t>(local) >= mesh.vertices.size() / 3) {
                            return false;
                        }
                        const float* position = mesh.vertices.data() + 3 * static_cast<std::size_t>(local);
                        for (int axis = 0; axis < 3; axis++) {
                            T coordinate = (T)instance.transform[axis * 4 + 3];
                            for (int component = 0; component < 3; component++) {
                                coordinate += (T)instance.transform[axis * 4 + component] * (T)position[component];
                            }
                            if (!collision::detail::finite(coordinate)) {
                                return false;
                            }
                            triangle.vertices[vertex][axis] = coordinate;
                        }
                    }
                    target.triangles.push_back(triangle);
                }
            }
        }
        const std::size_t count = target.triangles.size();
        if (count > (std::size_t)std::numeric_limits<TI>::max() / 2) {
            return false;
        }
        target.nodes.resize(count * 2);
        target.primitives.resize(count);
        std::vector<TI> temporary(count);
        std::vector<T> low(count * 3), high(count * 3), center(count * 3);
        for (std::size_t index = 0; index < count; index++) {
            target.primitives[index] = (TI)index;
            for (int axis = 0; axis < 3; axis++) {
                const auto& triangle = target.triangles[index];
                low[index * 3 + axis] = std::min({triangle.vertices[0][axis], triangle.vertices[1][axis], triangle.vertices[2][axis]});
                high[index * 3 + axis] = std::max({triangle.vertices[0][axis], triangle.vertices[1][axis], triangle.vertices[2][axis]});
                center[index * 3 + axis] = low[index * 3 + axis] / (T)2 + high[index * 3 + axis] / (T)2;
            }
        }
        using hyperlux::backends::generic::build_bvh_nodes;
        const TI node_count = build_bvh_nodes(target.nodes.data(), target.primitives.data(), temporary.data(), low.data(), high.data(), center.data(), (TI)count);
        target.nodes.resize(node_count);
        target.valid = true;
        return true;
    }

    template <typename DEVICE, typename SPEC, typename POINT_T, typename RADIUS_T>
    bool collision_free(DEVICE&, const rl::environments::hyperdrone::tasks::visual_inertial_localization::collision::Scene<SPEC>& scene, const POINT_T from[3], const POINT_T to[3], const RADIUS_T& radius) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        namespace detail = rl::environments::hyperdrone::tasks::visual_inertial_localization::collision::detail;
        if (!scene.valid || !detail::finite(radius) || !detail::finite((T)radius) || radius < (RADIUS_T)0) {
            return false;
        }
        T start[3], finish[3];
        for (int axis = 0; axis < 3; axis++) {
            if (!detail::finite(from[axis]) || !detail::finite(to[axis])) {
                return false;
            }
            start[axis] = (T)from[axis];
            finish[axis] = (T)to[axis];
            if (!detail::finite(start[axis]) || !detail::finite(finish[axis])) {
                return false;
            }
        }
        if (scene.nodes.empty()) {
            return true;
        }
        constexpr TI STACK_SIZE = hyperlux::backends::generic::constants::BUILD_STACK_SIZE<TI>;
        TI stack[STACK_SIZE], count = 0;
        stack[count++] = 0;
        while (count > 0) {
            const auto& node = scene.nodes[stack[--count]];
            if (!detail::overlaps(node, start, finish, (T)radius)) {
                continue;
            }
            if (node.count > 0) {
                for (TI index = 0; index < node.count; index++) {
                    if (detail::intersects(scene.triangles[scene.primitives[node.left_or_first + index]], start, finish, (T)radius)) {
                        return false;
                    }
                }
            } else {
                if (count + 2 > STACK_SIZE) {
                    return false;
                }
                stack[count++] = node.left_or_first + 1;
                stack[count++] = node.left_or_first;
            }
        }
        return true;
    }

    namespace rl::environments::hyperdrone::tasks::visual_inertial_localization::collision::detail {
        template <typename DEVICE, typename SPEC, std::size_t N>
        bool bezier_clear(DEVICE& device, const Scene<SPEC>& scene, const typename SPEC::T (&control)[N][3], typename SPEC::T radius, typename SPEC::T tolerance, unsigned depth) {
            using T = typename SPEC::T;
            const auto from = to_vec3(control[0]), to = to_vec3(control[N - 1]);
            T deviation_squared = 0;
            for (std::size_t point = 1; point + 1 < N; point++) {
                const T distance = point_segment_distance_squared(to_vec3(control[point]), from, to);
                if (!finite(distance)) {
                    return false;
                }
                deviation_squared = std::max(deviation_squared, distance);
            }
            const T deviation = math::sqrt(device.math, deviation_squared);
            if (collision_free(device, scene, control[0], control[N - 1], radius + deviation)) {
                return true;
            }
            if (deviation <= tolerance || depth == 16) {
                return false;
            }
            T left[N][3], right[N][3], intermediate[N][3];
            for (std::size_t point = 0; point < N; point++) {
                for (int axis = 0; axis < 3; axis++) {
                    intermediate[point][axis] = control[point][axis];
                }
            }
            for (int axis = 0; axis < 3; axis++) {
                left[0][axis] = control[0][axis];
                right[N - 1][axis] = control[N - 1][axis];
            }
            for (std::size_t level = 1; level < N; level++) {
                for (std::size_t point = 0; point < N - level; point++) {
                    for (int axis = 0; axis < 3; axis++) {
                        intermediate[point][axis] = intermediate[point][axis] / (T)2 + intermediate[point + 1][axis] / (T)2;
                    }
                }
                for (int axis = 0; axis < 3; axis++) {
                    left[level][axis] = intermediate[0][axis];
                    right[N - 1 - level][axis] = intermediate[N - 1 - level][axis];
                }
            }
            return bezier_clear(device, scene, left, radius, tolerance, depth + 1) &&
                   bezier_clear(device, scene, right, radius, tolerance, depth + 1);
        }
    }

    template <typename DEVICE, typename SPEC, typename POINT_T, std::size_t N, typename RADIUS_T>
    bool bezier_collision_free(DEVICE& device, const rl::environments::hyperdrone::tasks::visual_inertial_localization::collision::Scene<SPEC>& scene, const POINT_T (&control)[N][3], const RADIUS_T& radius, typename SPEC::T tolerance = (typename SPEC::T)0.01) {
        static_assert(N >= 2);
        using T = typename SPEC::T;
        namespace detail = rl::environments::hyperdrone::tasks::visual_inertial_localization::collision::detail;
        if (!scene.valid || !detail::finite(radius) || !detail::finite((T)radius) || radius < (RADIUS_T)0 || !detail::finite(tolerance) || tolerance < (T)0) {
            return false;
        }
        T converted[N][3];
        for (std::size_t point = 0; point < N; point++) {
            for (int axis = 0; axis < 3; axis++) {
                if (!detail::finite(control[point][axis])) {
                    return false;
                }
                converted[point][axis] = (T)control[point][axis];
                if (!detail::finite(converted[point][axis])) {
                    return false;
                }
            }
        }
        return detail::bezier_clear(device, scene, converted, (T)radius, tolerance, 0);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
