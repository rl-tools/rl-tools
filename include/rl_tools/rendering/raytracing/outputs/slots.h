#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_OUTPUTS_SLOTS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUTS_SLOTS_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::raytracing::outputs {
    // slot order is shader ABI: device.metal, device.comp and device.wgsl index fc_output_mask bits and output_offsets by these values
    enum class Slot: unsigned int {
        RGB_RGBA8, RGB_FLOAT32, DEPTH_FLOAT32, DEPTH_RGBA8,
        SEGMENTATION_UINT32, SEGMENTATION_RGBA8, SEMANTIC_UINT32, SEMANTIC_RGBA8,
        NORMALS_FLOAT32, NORMALS_RGBA8, FLOW_FLOAT32, FLOW_RGBA8,
        FLOW_STATUS_UINT32, FLOW_STATUS_RGBA8, FLOW_FORWARD_FLOAT32, FLOW_FORWARD_RGBA8,
        FLOW_FORWARD_STATUS_UINT32, FLOW_FORWARD_STATUS_RGBA8, PROBES, COUNT
    };

    constexpr unsigned int bit(Slot slot) { return 1u << (unsigned int)slot; }
    // per-camera work gates over a camera's output-slot mask
    constexpr unsigned int RGB_SLOTS = bit(Slot::RGB_RGBA8) | bit(Slot::RGB_FLOAT32);
    constexpr unsigned int DEPTH_SLOTS = bit(Slot::DEPTH_FLOAT32) | bit(Slot::DEPTH_RGBA8);
    constexpr unsigned int SEGMENTATION_SLOTS = bit(Slot::SEGMENTATION_UINT32) | bit(Slot::SEGMENTATION_RGBA8) | bit(Slot::SEMANTIC_UINT32) | bit(Slot::SEMANTIC_RGBA8);
    constexpr unsigned int NORMALS_SLOTS = bit(Slot::NORMALS_FLOAT32) | bit(Slot::NORMALS_RGBA8);
    constexpr unsigned int SURFACE_SLOTS = DEPTH_SLOTS | SEGMENTATION_SLOTS | NORMALS_SLOTS;
    constexpr unsigned int FLOW_BACKWARD_VISIBILITY_SLOTS = bit(Slot::FLOW_RGBA8) | bit(Slot::FLOW_STATUS_UINT32) | bit(Slot::FLOW_STATUS_RGBA8);
    constexpr unsigned int FLOW_BACKWARD_SLOTS = bit(Slot::FLOW_FLOAT32) | FLOW_BACKWARD_VISIBILITY_SLOTS;
    constexpr unsigned int FLOW_FORWARD_VISIBILITY_SLOTS = bit(Slot::FLOW_FORWARD_RGBA8) | bit(Slot::FLOW_FORWARD_STATUS_UINT32) | bit(Slot::FLOW_FORWARD_STATUS_RGBA8);
    constexpr unsigned int FLOW_FORWARD_SLOTS = bit(Slot::FLOW_FORWARD_FLOAT32) | FLOW_FORWARD_VISIBILITY_SLOTS;
    // camera table: per camera its output-slot mask, then its row in each output's tensor (NO_ROW
    // where unselected); built once from SPEC::OUTPUTS by fill_camera_table
    constexpr unsigned int CAMERA_TABLE_STRIDE = 1 + (unsigned int)Slot::COUNT;
    constexpr unsigned int NO_ROW = 0xffffffffu;
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
