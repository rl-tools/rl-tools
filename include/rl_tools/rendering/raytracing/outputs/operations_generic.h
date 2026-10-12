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
    template <bool DIRECT = false, typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT unsigned int output_hue(DEVICE& device, float hue, float saturation) {
        const float descending = 1.f - math::abs(device.math, (hue - 2.f * math::floor(device.math, hue * 0.5f)) - 1.f);
        float red = 0, green = 0, blue = 0;
        switch((int)hue % 6) {
            case 0: red = 1; green = descending; break;
            case 1: red = descending; green = 1; break;
            case 2: green = 1; blue = descending; break;
            case 3: green = descending; blue = 1; break;
            case 4: red = descending; blue = 1; break;
            default: red = 1; blue = descending; break;
        }
        const auto channel = [&](float value) {
            if constexpr (DIRECT) {
                return (unsigned int)(value * 255.f);
            }
            else {
                return (unsigned int)((1.f - saturation * (1.f - value)) * 255.f);
            }
        };
        return 0xff000000u | channel(red) | (channel(green) << 8) | (channel(blue) << 16);
    }
    template <typename DEVICE>
    RL_TOOLS_FUNCTION_PLACEMENT unsigned int output_status_color(DEVICE&, unsigned int status) {
        using STATUS = rendering::raytracing::FlowStatus;
        switch(status) {
            case STATUS::VALID: return 0xffffffffu;
            case STATUS::MISS: return 0xff000000u;
            case STATUS::BEHIND: return 0xffff00ffu;
            case STATUS::OUT_OF_FRAME: return 0xffff4020u;
            case STATUS::OCCLUDED: return 0xff0090ffu;
            default: return 0xff0000ffu;
        }
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
    RL_TOOLS_FUNCTION_PLACEMENT void write_output(DEVICE&, rendering::raytracing::outputs::Buffer<SPEC, rendering::raytracing::outputs::Depth<FORMAT, PARAMETERS>>& buffer, typename SPEC::TI pixel, const rendering::raytracing::outputs::Sample<rendering::raytracing::outputs::Source::Surface>& sample) {
        if constexpr (utils::typing::is_same_v<FORMAT, rendering::raytracing::outputs::format::RGBA8>) {
            const float fraction = sample.depth / PARAMETERS::SCALE;
            const unsigned int value = (unsigned int)((fraction < 0 ? 0 : (fraction > 1 ? 1 : fraction)) * 255.f);
            data(buffer.tensor)[pixel] = 0xff000000u | value | (value << 8) | (value << 16);
        }
        else {
            data(buffer.tensor)[pixel] = sample.depth;
        }
    }
    template <typename DEVICE, typename SPEC, typename FORMAT, typename LABELS>
    RL_TOOLS_FUNCTION_PLACEMENT void write_output(DEVICE& device, rendering::raytracing::outputs::Buffer<SPEC, rendering::raytracing::outputs::Segmentation<FORMAT, LABELS>>& buffer, typename SPEC::TI pixel, const rendering::raytracing::outputs::Sample<rendering::raytracing::outputs::Source::Surface>& sample) {
        const unsigned int id = utils::typing::is_same_v<LABELS, rendering::raytracing::outputs::labels::Semantic> ? sample.semantic : sample.instance;
        if constexpr (utils::typing::is_same_v<FORMAT, rendering::raytracing::outputs::format::RGBA8>) {
            const float scaled = (float)id * 0.61803398875f;
            data(buffer.tensor)[pixel] = id == 0xffffffffu ? 0xff000000u : output_hue<true>(device, (scaled - math::floor(device.math, scaled)) * 6.f, 1.f);
        }
        else {
            data(buffer.tensor)[pixel] = id;
        }
    }
    template <typename DEVICE, typename SPEC, typename FORMAT>
    RL_TOOLS_FUNCTION_PLACEMENT void write_output(DEVICE&, rendering::raytracing::outputs::Buffer<SPEC, rendering::raytracing::outputs::Normals<FORMAT>>& buffer, typename SPEC::TI pixel, const rendering::raytracing::outputs::Sample<rendering::raytracing::outputs::Source::Surface>& sample) {
        if constexpr (utils::typing::is_same_v<FORMAT, rendering::raytracing::outputs::format::RGBA8>) {
            unsigned int packed = 0xff000000u;
            for(unsigned int channel = 0; channel < 3; channel++) {
                const float value = sample.values[channel];
                const float clamped = value < -1.f ? -1.f : (value > 1.f ? 1.f : value);
                volatile float unit = clamped * 0.5f + 0.5f;
                volatile float scaled = unit * 255.f;
                packed |= (unsigned int)((double)scaled + 0.5) << (8 * channel);
            }
            data(buffer.tensor)[pixel] = packed;
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
            unsigned int color = output_status_color(device, sample.status);
            if(sample.status == rendering::raytracing::FlowStatus::VALID) {
                const float u = sample.values[0], v = sample.values[1];
                const float fraction = PARAMETERS::SCALE > 0 ? math::sqrt(device.math, u*u + v*v) / PARAMETERS::SCALE : 0.f;
                const float hue = (math::atan2(device.math, v, u) / 6.28318530717958647692f + 0.5f) * 6.f;
                color = output_hue(device, hue, fraction > 1.f ? 1.f : fraction);
            }
            data(buffer.tensor)[pixel] = color;
        }
        else {
            data(buffer.tensor)[pixel * 2] = sample.values[0];
            data(buffer.tensor)[pixel * 2 + 1] = sample.values[1];
        }
    }
    template <typename DEVICE, typename SPEC, typename DIRECTION, typename FORMAT, rendering::raytracing::outputs::Source SOURCE>
    RL_TOOLS_FUNCTION_PLACEMENT void write_output(DEVICE& device, rendering::raytracing::outputs::Buffer<SPEC, rendering::raytracing::outputs::FlowStatus<DIRECTION, FORMAT>>& buffer, typename SPEC::TI pixel, const rendering::raytracing::outputs::Sample<SOURCE>& sample) {
        if constexpr (utils::typing::is_same_v<FORMAT, rendering::raytracing::outputs::format::RGBA8>) {
            data(buffer.tensor)[pixel] = output_status_color(device, sample.status);
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
