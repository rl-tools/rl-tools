#pragma once

#include "model.h"

namespace rl_tools {

    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model, GROUP& group) {
        set_attribute(device, group, "type", "cross_conv_siamese");
        write_attributes(device, group);
        auto g_early_a = create_group(device, group, "early_encoder_a");
        save(device, model.early_encoder_a, g_early_a);
        auto g_early_b = create_group(device, group, "early_encoder_b");
        save(device, model.early_encoder_b, g_early_b);
        auto g_mid_a = create_group(device, group, "mid_conv_a");
        save(device, model.mid_conv_a, g_mid_a);
        auto g_mid_b = create_group(device, group, "mid_conv_b");
        save(device, model.mid_conv_b, g_mid_b);
        auto g_late_a = create_group(device, group, "late_encoder_a");
        save(device, model.late_encoder_a, g_late_a);
        auto g_late_b = create_group(device, group, "late_encoder_b");
        save(device, model.late_encoder_b, g_late_b);
        auto g_head = create_group(device, group, "head");
        save(device, model.head, g_head);
    }

    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rendering::raytracing::yaw_prediction::ModuleForward<SPEC>& model, GROUP& group) {
        auto g_early_a = get_group(device, group, "early_encoder_a");
        bool success = load(device, model.early_encoder_a, g_early_a);
        auto g_early_b = get_group(device, group, "early_encoder_b");
        success &= load(device, model.early_encoder_b, g_early_b);
        auto g_mid_a = get_group(device, group, "mid_conv_a");
        success &= load(device, model.mid_conv_a, g_mid_a);
        auto g_mid_b = get_group(device, group, "mid_conv_b");
        success &= load(device, model.mid_conv_b, g_mid_b);
        auto g_late_a = get_group(device, group, "late_encoder_a");
        success &= load(device, model.late_encoder_a, g_late_a);
        auto g_late_b = get_group(device, group, "late_encoder_b");
        success &= load(device, model.late_encoder_b, g_late_b);
        auto g_head = get_group(device, group, "head");
        success &= load(device, model.head, g_head);
        return success;
    }
}
