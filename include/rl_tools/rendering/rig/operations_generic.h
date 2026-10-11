#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RIG_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RIG_OPERATIONS_GENERIC_H

#include "rig.h"
#include "../transforms.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::rig::detail {
        // Read exponent bits through a reference: fast-math may assume a floating-point value parameter is finite before inspecting its bits.
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT bool is_finite(const T& value){
            static_assert(sizeof(T) == 4 || sizeof(T) == 8, "is_finite expects an IEEE-754 single or double");
            using BITS = utils::typing::conditional_t<sizeof(T) == 4, unsigned int, unsigned long long>;
            static_assert(sizeof(BITS) == sizeof(T));
            BITS bits;
            auto* destination = reinterpret_cast<unsigned char*>(&bits);
            const auto* source = reinterpret_cast<const unsigned char*>(&value);
            for(unsigned byte = 0; byte < sizeof(T); byte++){
                destination[byte] = source[byte];
            }
            constexpr BITS EXPONENT = sizeof(T) == 4 ? (BITS)0x7f800000u : (BITS)0x7ff0000000000000ull;
            return (bits & EXPONENT) != EXPONENT;
        }
        template <typename T>
        RL_TOOLS_FUNCTION_PLACEMENT bool is_finite(const T* values, unsigned count){
            for(unsigned i = 0; i < count; i++){
                if(!is_finite(values[i])){
                    return false;
                }
            }
            return true;
        }
    }
    namespace rendering::rig {
        template <typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void joint_transform(DEVICE& device, const Link<SPEC>& link, const typename SPEC::T* coordinates, typename SPEC::T out[12]){
            using T = typename SPEC::T;
            for(unsigned element = 0; element < 12; element++){
                out[element] = element % 5 == 0 ? (T)1 : (T)0;
            }
            if(link.joint == Joint::FIXED){
                return;
            }
            const T* q = coordinates + link.coordinate;
            if(link.joint == Joint::PRISMATIC){
                for(unsigned axis = 0; axis < 3; axis++){
                    out[axis * 4 + 3] = link.axis[axis] * q[0];
                }
            }
            else if(link.joint == Joint::SPHERICAL){
                const T norm = math::sqrt(device.math, q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
                const T w = q[0] / norm, x = q[1] / norm, y = q[2] / norm, z = q[3] / norm;
                out[0] = 1 - 2*(y*y + z*z); out[1] = 2*(x*y - w*z);     out[2]  = 2*(x*z + w*y);
                out[4] = 2*(x*y + w*z);     out[5] = 1 - 2*(x*x + z*z); out[6]  = 2*(y*z - w*x);
                out[8] = 2*(x*z - w*y);     out[9] = 2*(y*z + w*x);     out[10] = 1 - 2*(x*x + y*y);
            }
            else{
                const T c = math::cos(device.math, q[0]);
                const T s = math::sin(device.math, q[0]);
                const T v = (T)1 - c;
                const T x = link.axis[0], y = link.axis[1], z = link.axis[2];
                out[0] = c + x*x*v;   out[1] = x*y*v - z*s; out[2]  = x*z*v + y*s;
                out[4] = y*x*v + z*s; out[5] = c + y*y*v;   out[6]  = y*z*v - x*s;
                out[8] = z*x*v - y*s; out[9] = z*y*v + x*s; out[10] = c + z*z*v;
            }
        }
    }

    // poses: num_parts geometry-to-world transforms, 12 values each, in part order
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate(DEVICE& device, const rendering::rig::Model<SPEC>& model, const typename SPEC::T base[12], const typename SPEC::T* coordinates, rendering::rig::Buffer<SPEC>& buffer, typename SPEC::T* poses, typename SPEC::T placement_scale = 1){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        T placement[12];
        for(unsigned row = 0; row < 3; row++){
            for(unsigned column = 0; column < 3; column++){
                placement[row * 4 + column] = base[row * 4 + column] * placement_scale;
            }
            placement[row * 4 + 3] = base[row * 4 + 3];
        }
        for(TI link_i = 0; link_i < model.num_links; link_i++){
            const auto& link = model.links[link_i];
            const T* parent = link.parent == SPEC::BASE ? placement : buffer.link_poses[link.parent];
            T joint[12], frame[12];
            rendering::rig::joint_transform(device, link, coordinates, joint);
            rendering::compose_transforms(parent, link.origin, frame);
            rendering::compose_transforms(frame, joint, buffer.link_poses[link_i]);
        }
        for(TI part_i = 0; part_i < model.num_parts; part_i++){
            const auto& part = model.parts[part_i];
            rendering::compose_transforms(buffer.link_poses[part.link], part.offset, poses + part_i * 12);
        }
    }

    // coordinates between two states at t: scalar joints lerp (a revolute joint keeps every
    // revolution between the states), spherical joints slerp along the shorter arc
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void interpolate(DEVICE& device, const rendering::rig::Model<SPEC>& model, const typename SPEC::T* open, const typename SPEC::T* close, typename SPEC::T t, typename SPEC::T* out){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        for(TI coordinate = 0; coordinate < model.num_coordinates; coordinate++){
            out[coordinate] = open[coordinate] + t * (close[coordinate] - open[coordinate]);
        }
        for(TI link_i = 0; link_i < model.num_links; link_i++){
            const auto& link = model.links[link_i];
            if(link.joint != rendering::rig::Joint::SPHERICAL){
                continue;
            }
            const T* a = open + link.coordinate;
            const T* b = close + link.coordinate;
            const T norm_a = math::sqrt(device.math, a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]);
            const T norm_b = math::sqrt(device.math, b[0]*b[0] + b[1]*b[1] + b[2]*b[2] + b[3]*b[3]);
            T cosine = (a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]) / (norm_a * norm_b);
            const T sign = cosine < 0 ? (T)-1 : (T)1;
            cosine *= sign;
            T weight_a = (T)1 - t;
            T weight_b = t;
            if(cosine < (T)0.9995){
                const T angle = math::acos(device.math, cosine);
                const T sine = math::sin(device.math, angle);
                weight_a = math::sin(device.math, ((T)1 - t) * angle) / sine;
                weight_b = math::sin(device.math, t * angle) / sine;
            }
            for(unsigned component = 0; component < 4; component++){
                out[link.coordinate + component] = weight_a * a[component] / norm_a + weight_b * sign * b[component] / norm_b;
            }
        }
    }

    // num_samples poses per part at the shutter times (sample + 0.5) / num_samples, sample s at
    // samples + s * sample_stride (12 values per part), plus the poses at t = 1 in close; the
    // base follows the constant-twist screw between base_open and base_close, the coordinates
    // interpolate before the kinematics; placement scale is applied after rigid-base interpolation
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void sample(DEVICE& device, const rendering::rig::Model<SPEC>& model, const typename SPEC::T base_open[12], const typename SPEC::T base_close[12], const typename SPEC::T* coordinates_open, const typename SPEC::T* coordinates_close, typename SPEC::TI num_samples, rendering::rig::Buffer<SPEC>& buffer, typename SPEC::T* samples, typename SPEC::TI sample_stride, typename SPEC::T* close, typename SPEC::T placement_scale = 1){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<T, float>, "rig::sample interpolates the base with the float transform helpers");
        for(TI sample_i = 0; sample_i < num_samples; sample_i++){
            const T t = ((T)sample_i + (T)0.5) / (T)num_samples;
            T base[12];
            rendering::screw_interpolate(base_open, base_close, t, base);
            interpolate(device, model, coordinates_open, coordinates_close, t, buffer.coordinates);
            evaluate(device, model, base, buffer.coordinates, buffer, samples + sample_i * sample_stride, placement_scale);
        }
        evaluate(device, model, base_close, coordinates_close, buffer, close, placement_scale);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
