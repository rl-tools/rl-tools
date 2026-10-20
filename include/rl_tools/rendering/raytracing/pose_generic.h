#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_POSE_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_POSE_GENERIC_H

#include "../../rl_tools.h"
#include "../transforms.h"
#include "pose.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rendering::raytracing::detail{
        template <typename DEVICE>
        RL_TOOLS_FUNCTION_PLACEMENT void joint_transform(DEVICE& device, const rendering::Joint& joint, float q, float out[12]){
            for(unsigned element = 0; element < 12; element++){
                out[element] = element % 5 == 0 ? 1.0f : 0.0f;
            }
            const float x = joint.axis[0], y = joint.axis[1], z = joint.axis[2];
            if(joint.type == rendering::Joint::Type::PRISMATIC){
                out[3] = x * q;
                out[7] = y * q;
                out[11] = z * q;
                return;
            }
            const float c = math::cos(device.math, q);
            const float s = math::sin(device.math, q);
            const float v = 1.0f - c;
            out[0] = c + x*x*v;   out[1] = x*y*v - z*s; out[2]  = x*z*v + y*s;
            out[4] = y*x*v + z*s; out[5] = c + y*y*v;   out[6]  = y*z*v - x*s;
            out[8] = z*x*v - y*s; out[9] = z*y*v + x*s; out[10] = c + z*z*v;
            for(unsigned row = 0; row < 3; row++){
                out[row * 4 + 3] = joint.pivot[row] - (out[row * 4] * joint.pivot[0] + out[row * 4 + 1] * joint.pivot[1] + out[row * 4 + 2] * joint.pivot[2]);
            }
        }

        // one state's rows (part i at rows + 12 i): reference ∘ origin ∘ joint(q), the reference being
        // the scaled base or the row already written for the part's articulated ancestor, so rigid
        // parts compose base ∘ rest exactly. With interpolate, q lerps between the coordinate sets
        // at t, keeping every revolution in between
        template <typename DEVICE>
        RL_TOOLS_FUNCTION_PLACEMENT void pose_rows(DEVICE& device, const rendering::raytracing::AssetPart* parts, size_t num_parts, const float base[12], float scale, const float* coordinates_open, const float* coordinates_close, bool interpolate, float t, float* rows){
            float scaled[12];
            for(unsigned row = 0; row < 3; row++){
                for(unsigned column = 0; column < 3; column++){
                    scaled[row * 4 + column] = base[row * 4 + column] * scale;
                }
                scaled[row * 4 + 3] = base[row * 4 + 3];
            }
            for(size_t part_i = 0; part_i < num_parts; part_i++){
                const auto& part = parts[part_i];
                const float* reference = part.reference == rendering::raytracing::AssetPart::NO_REFERENCE ? scaled : rows + part.reference * 12;
                float* row = rows + part_i * 12;
                if(part.joint.type == rendering::Joint::Type::FIXED){
                    rendering::compose_transforms(reference, part.origin, row);
                    continue;
                }
                const size_t coordinate = part.joint.coordinate;
                const float close = coordinates_close != nullptr ? coordinates_close[coordinate] : 0.0f;
                float q = close;
                if(interpolate){
                    const float open = coordinates_open != nullptr ? coordinates_open[coordinate] : 0.0f;
                    q = open + t * (close - open);
                }
                float motion[12], frame[12];
                joint_transform(device, part.joint, q, motion);
                rendering::compose_transforms(reference, part.origin, frame);
                rendering::compose_transforms(frame, motion, row);
            }
        }
    }

    RL_TOOLS_FUNCTION_PLACEMENT inline rendering::raytracing::Pose make_pose(const float base[12], float scale = 1){
        rendering::raytracing::Pose pose;
        for(unsigned element = 0; element < 12; element++){
            pose.base[element] = base[element];
        }
        pose.scale = scale;
        return pose;
    }

    // the device verb behind the host set_pose: one placement's rows (and, under dynamic motion
    // blur, the same rows in every motion sample) written into a pose sink. coordinates are the
    // asset's joint values in coordinate order, borrowed for the call; nullptr is the rest state
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void set_pose(DEVICE& device, const rendering::raytracing::PoseSink<SPEC>& sink, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const rendering::raytracing::Pose& pose, const float* coordinates = nullptr){
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        const size_t row = overlay.index * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot;
        float* close = sink.transforms + row * 12;
        rendering::raytracing::detail::pose_rows(device, sink.parts + placement.first_part, placement.num_parts, pose.base, pose.scale, nullptr, coordinates, false, 1.0f, close);
        if constexpr (SPEC::ENABLE_DYNAMIC_MOTION_BLUR){
            for(size_t sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
                float* samples = sink.transforms_motion + (sample * SLOTS + row) * 12;
                for(size_t element = 0; element < placement.num_parts * 12; element++){
                    samples[element] = close[element];
                }
            }
        }
    }

    // shutter-open/close states: the close rows are the steady state segmentation, probes and the
    // next frame see; sample s at (s + 0.5) / MOTION_BLUR_SAMPLES follows the constant-twist screw
    // between the bases with scale and coordinates interpolated
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void set_pose(DEVICE& device, const rendering::raytracing::PoseSink<SPEC>& sink, rendering::raytracing::OverlayIndex overlay, const rendering::raytracing::OverlayPlacement& placement, const rendering::raytracing::Pose& open, const rendering::raytracing::Pose& close, const float* coordinates_open = nullptr, const float* coordinates_close = nullptr){
        static_assert(SPEC::ENABLE_DYNAMIC_MOTION_BLUR, "set_pose with shutter-open/close states requires a dynamic-motion-blur renderer specification");
        constexpr size_t SLOTS = (size_t)SPEC::NUM_OVERLAYS * SPEC::MAX_OVERLAY_INSTANCES;
        const size_t row = overlay.index * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot;
        const rendering::raytracing::AssetPart* parts = sink.parts + placement.first_part;
        rendering::raytracing::detail::pose_rows(device, parts, placement.num_parts, close.base, close.scale, nullptr, coordinates_close, false, 1.0f, sink.transforms + row * 12);
        for(size_t sample = 0; sample < SPEC::MOTION_BLUR_SAMPLES; sample++){
            const float t = ((float)sample + 0.5f) / (float)SPEC::MOTION_BLUR_SAMPLES;
            float base[12];
            rendering::screw_interpolate(open.base, close.base, t, base);
            const float scale = open.scale + t * (close.scale - open.scale);
            rendering::raytracing::detail::pose_rows(device, parts, placement.num_parts, base, scale, coordinates_open, coordinates_close, true, t, sink.transforms_motion + (sample * SLOTS + row) * 12);
        }
    }

    // flow: the previous instant of one overlay becomes its current instant (an episode reset: no
    // motion of its own in the next frame); the device overload of advance_flow
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void advance_flow(DEVICE& device, const rendering::raytracing::PoseSink<SPEC>& sink, rendering::raytracing::OverlayIndex overlay){
        static_assert(SPEC::HAS_FLOW_OVERLAYS, "advance_flow of an overlay requires a flow renderer specification with overlays");
        const size_t first = overlay.index * SPEC::MAX_OVERLAY_INSTANCES * 12;
        for(size_t element = 0; element < SPEC::MAX_OVERLAY_INSTANCES * 12; element++){
            sink.transforms_previous[first + element] = sink.transforms[first + element];
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
