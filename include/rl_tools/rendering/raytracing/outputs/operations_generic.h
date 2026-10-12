#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_OUTPUTS_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUTS_OPERATIONS_GENERIC_H
#include "outputs.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename SPEC, typename OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr OUTPUT output_type(const rendering::raytracing::outputs::Buffer<SPEC, OUTPUT>&) { return {}; }
    template <typename DEVICE, typename TI, typename FUNCTION>
    RL_TOOLS_FUNCTION_PLACEMENT void for_each_output(DEVICE&, utils::Tuple<TI>&, FUNCTION&&) {}
    template <typename DEVICE, typename TI, typename HEAD, typename... TAIL, typename FUNCTION>
    RL_TOOLS_FUNCTION_PLACEMENT void for_each_output(DEVICE& device, utils::Tuple<TI, HEAD, TAIL...>& values, FUNCTION&& function) {
        function(values.content);
        for_each_output(device, static_cast<utils::Tuple<TI, TAIL...>&>(values), function);
    }
    template <typename DEVICE, typename SPEC, typename LIST, typename FUNCTION>
    RL_TOOLS_FUNCTION_PLACEMENT void for_each_output(DEVICE& device, rendering::raytracing::outputs::Buffers<SPEC, LIST>& values, FUNCTION&& function) {
        for_each_output(device, values.content, function);
    }
    template <typename DEVICE, typename TI, typename SPEC, typename HEAD, typename... TAIL, typename OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT auto& output(DEVICE& device, utils::Tuple<TI, rendering::raytracing::outputs::Buffer<SPEC, HEAD>, TAIL...>& values, OUTPUT descriptor) {
        if constexpr (utils::typing::is_same_v<HEAD, OUTPUT>) {
            return values.content.tensor;
        }
        else {
            return output(device, static_cast<utils::Tuple<TI, TAIL...>&>(values), descriptor);
        }
    }
    template <typename DEVICE, typename SPEC, typename LIST, typename OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT auto& output(DEVICE& device, rendering::raytracing::outputs::Buffers<SPEC, LIST>& values, OUTPUT descriptor) {
        static_assert(rendering::raytracing::outputs::contains<LIST, OUTPUT>::value, "requested output is not configured");
        return output(device, values.content, descriptor);
    }
    template <typename DEVICE, typename TI, typename SPEC, typename HEAD, typename... TAIL, typename OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT const auto& output(DEVICE& device, const utils::Tuple<TI, rendering::raytracing::outputs::Buffer<SPEC, HEAD>, TAIL...>& values, OUTPUT descriptor) {
        if constexpr (utils::typing::is_same_v<HEAD, OUTPUT>) {
            return values.content.tensor;
        }
        else {
            return output(device, static_cast<const utils::Tuple<TI, TAIL...>&>(values), descriptor);
        }
    }
    template <typename DEVICE, typename SPEC, typename LIST, typename OUTPUT>
    RL_TOOLS_FUNCTION_PLACEMENT const auto& output(DEVICE& device, const rendering::raytracing::outputs::Buffers<SPEC, LIST>& values, OUTPUT descriptor) {
        static_assert(rendering::raytracing::outputs::contains<LIST, OUTPUT>::value, "requested output is not configured");
        return output(device, values.content, descriptor);
    }
    template <typename DEVICE, typename SPEC, typename LIST>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rendering::raytracing::outputs::Buffers<SPEC, LIST>& values) {
        for_each_output(device, values, [&](auto& value) { malloc(device, value.tensor); });
    }
    template <typename DEVICE, typename SPEC, typename LIST>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rendering::raytracing::outputs::Buffers<SPEC, LIST>& values) {
        for_each_output(device, values, [&](auto& value) { free(device, value.tensor); });
    }
    namespace rendering::raytracing::outputs {
        template <Source T_SOURCE>
        struct Sample {
            static constexpr Source SOURCE = T_SOURCE;
            float values[3] = {};
            unsigned int instance = 0xffffffffu;
            unsigned int semantic = 0xffffffffu;
            unsigned int status = rendering::raytracing::FlowStatus::MISS;
            float depth;
        };
    }
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT void hue_channels(DEVICE& device, float hue, float channels[3]) {
        const float descending = 1.f - math::abs(device.math, (hue - 2.f * math::floor(device.math, hue * 0.5f)) - 1.f);
        const float sectors[6][3] = {{1, descending, 0}, {descending, 1, 0}, {0, 1, descending}, {0, descending, 1}, {descending, 0, 1}, {1, 0, descending}};
        const int sector = (int)hue % 6;
        for(unsigned int channel = 0; channel < 3; channel++) {
            channels[channel] = sectors[sector >= 0 && sector < 5 ? sector : 5][channel];
        }
    }
    // golden-ratio hue hash, one distinct color per id; miss sentinel black
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT unsigned int segmentation_color(DEVICE& device, unsigned int id) {
        if(id == 0xffffffffu) {
            return 0xff000000u;
        }
        const float scaled = (float)id * 0.61803398875f;
        float channels[3];
        hue_channels(device, (scaled - math::floor(device.math, scaled)) * 6.f, channels);
        return 0xff000000u | (unsigned int)(channels[0] * 255.f) | ((unsigned int)(channels[1] * 255.f) << 8) | ((unsigned int)(channels[2] * 255.f) << 16);
    }
    // pinned encoding mirrored by the shaders: two f32 roundings, then a double half-up bias. The
    // volatile intermediates defeat fast-math reassociation; NaN components clamp to -1 like fmaxf
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT unsigned int normal_color(DEVICE&, const float normal[3]) {
        unsigned int packed = 0xff000000u;
        for(unsigned int channel = 0; channel < 3; channel++) {
            const float value = normal[channel];
            const float clamped = value > 1.f ? 1.f : (value > -1.f ? value : -1.f);
            volatile float unit = clamped * 0.5f + 0.5f;
            volatile float scaled = unit * 255.f;
            packed |= (unsigned int)((double)scaled + 0.5) << (8 * channel);
        }
        return packed;
    }
    // VALID white, MISS black, BEHIND magenta, OUT_OF_FRAME blue, OCCLUDED orange, unknown red
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT unsigned int flow_status_color(DEVICE&, unsigned int status) {
        using STATUS = rendering::raytracing::FlowStatus;
        switch(status) {
            case STATUS::VALID: {
                return 0xffffffffu;
            }
            case STATUS::MISS: {
                return 0xff000000u;
            }
            case STATUS::BEHIND: {
                return 0xffff00ffu;
            }
            case STATUS::OUT_OF_FRAME: {
                return 0xffff4020u;
            }
            case STATUS::OCCLUDED: {
                return 0xff0090ffu;
            }
            default: {
                return 0xff0000ffu;
            }
        }
    }
    // direction to hue, magnitude to saturation against magnitude_scale in pixels (review writers
    // pass the per-image maximum, live viewers a fixed scale)
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT unsigned int flow_color(DEVICE& device, float u, float v, float magnitude_scale) {
        const float magnitude = math::sqrt(device.math, u * u + v * v);
        const float saturation = magnitude_scale > 0 ? math::min(device.math, magnitude / magnitude_scale, 1.f) : 0.f;
        float channels[3];
        hue_channels(device, (math::atan2(device.math, v, u) / 6.28318530717958647692f + 0.5f) * 6.f, channels);
        unsigned int packed = 0xff000000u;
        for(unsigned int channel = 0; channel < 3; channel++) {
            packed |= (unsigned int)((1.f - saturation * (1.f - channels[channel])) * 255.f) << (8 * channel);
        }
        return packed;
    }
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT unsigned int depth_color(DEVICE& device, float depth, float scale) {
        const unsigned int value = (unsigned int)(math::clamp(device.math, depth / scale, 0.f, 1.f) * 255.f);
        return 0xff000000u | value | (value << 8) | (value << 16);
    }
    template <typename DEVICE, typename SPEC, typename FORMAT>
    RL_TOOLS_FUNCTION_PLACEMENT void write_output(DEVICE&, rendering::raytracing::outputs::Buffer<SPEC, rendering::raytracing::outputs::RGB<FORMAT>>& buffer, typename SPEC::TI pixel, const rendering::raytracing::outputs::Sample<rendering::raytracing::outputs::Source::RGB>& sample) {
        if constexpr (utils::typing::is_same_v<FORMAT, rendering::raytracing::outputs::format::RGBA8>) {
            unsigned int packed = 0xff000000u;
            for(unsigned int channel = 0; channel < 3; channel++) {
                const int value = (int)(sample.values[channel] * 256.f);
                packed |= (unsigned int)(value < 0 ? 0 : (value > 255 ? 255 : value)) << (channel * 8);
            }
            data(buffer.tensor)[pixel] = packed;
        }
        else {
            for(unsigned int channel = 0; channel < 3; channel++) {
                data(buffer.tensor)[pixel * 3 + channel] = sample.values[channel];
            }
        }
    }
    template <typename DEVICE, typename SPEC, typename FORMAT, typename PARAMETERS>
    RL_TOOLS_FUNCTION_PLACEMENT void write_output(DEVICE& device, rendering::raytracing::outputs::Buffer<SPEC, rendering::raytracing::outputs::Depth<FORMAT, PARAMETERS>>& buffer, typename SPEC::TI pixel, const rendering::raytracing::outputs::Sample<rendering::raytracing::outputs::Source::Surface>& sample) {
        if constexpr (utils::typing::is_same_v<FORMAT, rendering::raytracing::outputs::format::RGBA8>) {
            data(buffer.tensor)[pixel] = depth_color(device, sample.depth, PARAMETERS::SCALE);
        }
        else {
            data(buffer.tensor)[pixel] = sample.depth;
        }
    }
    template <typename DEVICE, typename SPEC, typename FORMAT, typename LABELS>
    RL_TOOLS_FUNCTION_PLACEMENT void write_output(DEVICE& device, rendering::raytracing::outputs::Buffer<SPEC, rendering::raytracing::outputs::Segmentation<FORMAT, LABELS>>& buffer, typename SPEC::TI pixel, const rendering::raytracing::outputs::Sample<rendering::raytracing::outputs::Source::Surface>& sample) {
        const unsigned int id = utils::typing::is_same_v<LABELS, rendering::raytracing::outputs::labels::Semantic> ? sample.semantic : sample.instance;
        if constexpr (utils::typing::is_same_v<FORMAT, rendering::raytracing::outputs::format::RGBA8>) {
            data(buffer.tensor)[pixel] = segmentation_color(device, id);
        }
        else {
            data(buffer.tensor)[pixel] = id;
        }
    }
    template <typename DEVICE, typename SPEC, typename FORMAT>
    RL_TOOLS_FUNCTION_PLACEMENT void write_output(DEVICE& device, rendering::raytracing::outputs::Buffer<SPEC, rendering::raytracing::outputs::Normals<FORMAT>>& buffer, typename SPEC::TI pixel, const rendering::raytracing::outputs::Sample<rendering::raytracing::outputs::Source::Surface>& sample) {
        if constexpr (utils::typing::is_same_v<FORMAT, rendering::raytracing::outputs::format::RGBA8>) {
            data(buffer.tensor)[pixel] = normal_color(device, sample.values);
        }
        else {
            for(unsigned int channel = 0; channel < 3; channel++) {
                data(buffer.tensor)[pixel * 3 + channel] = sample.values[channel];
            }
        }
    }
    template <typename DEVICE, typename SPEC, typename DIRECTION, typename FORMAT, typename PARAMETERS, rendering::raytracing::outputs::Source SOURCE>
    RL_TOOLS_FUNCTION_PLACEMENT void write_output(DEVICE& device, rendering::raytracing::outputs::Buffer<SPEC, rendering::raytracing::outputs::Flow<DIRECTION, FORMAT, PARAMETERS>>& buffer, typename SPEC::TI pixel, const rendering::raytracing::outputs::Sample<SOURCE>& sample) {
        if constexpr (utils::typing::is_same_v<FORMAT, rendering::raytracing::outputs::format::RGBA8>) {
            data(buffer.tensor)[pixel] = sample.status == rendering::raytracing::FlowStatus::VALID ? flow_color(device, sample.values[0], sample.values[1], PARAMETERS::SCALE) : flow_status_color(device, sample.status);
        }
        else {
            data(buffer.tensor)[pixel * 2] = sample.values[0];
            data(buffer.tensor)[pixel * 2 + 1] = sample.values[1];
        }
    }
    template <typename DEVICE, typename SPEC, typename DIRECTION, typename FORMAT, rendering::raytracing::outputs::Source SOURCE>
    RL_TOOLS_FUNCTION_PLACEMENT void write_output(DEVICE& device, rendering::raytracing::outputs::Buffer<SPEC, rendering::raytracing::outputs::FlowStatus<DIRECTION, FORMAT>>& buffer, typename SPEC::TI pixel, const rendering::raytracing::outputs::Sample<SOURCE>& sample) {
        if constexpr (utils::typing::is_same_v<FORMAT, rendering::raytracing::outputs::format::RGBA8>) {
            data(buffer.tensor)[pixel] = flow_status_color(device, sample.status);
        }
        else {
            data(buffer.tensor)[pixel] = sample.status;
        }
    }
    template <typename DEVICE, typename SPEC, typename LIST, typename SAMPLE>
    RL_TOOLS_FUNCTION_PLACEMENT void write_outputs(DEVICE& device, rendering::raytracing::outputs::Buffers<SPEC, LIST>& buffers, typename SPEC::TI pixel, const SAMPLE& sample) {
        for_each_output(device, buffers, [&](auto& buffer) {
            using OUTPUT = decltype(output_type(buffer));
            if constexpr (rendering::raytracing::outputs::Traits<OUTPUT>::SOURCE == SAMPLE::SOURCE) {
                write_output(device, buffer, pixel, sample);
            }
        });
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
