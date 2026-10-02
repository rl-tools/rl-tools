#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_OPERATIONS_GENERIC_POSITION_HOLD_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TASKS_TARGET_FRAME_OPERATIONS_GENERIC_POSITION_HOLD_H

#include "position_hold.h"
#include "../../../../../nn_models/sequential/operations_generic.h"
#include "../../../../../nn_models/parallel/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename MODEL, typename TYPE_POLICY, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, MODEL& actor, const rl::environments::hyperdrone::tasks::target_frame::position_hold::Specification<TYPE_POLICY, TI>&){
        using SPEC = rl::environments::hyperdrone::tasks::target_frame::position_hold::Specification<TYPE_POLICY, TI>;
        using T = typename SPEC::T;
        constexpr TI ACTION_DIM = SPEC::ACTION_DIM;
        using nn_models::sequential::layer;
        {
            auto& image_branch = get<0>(actor.pipelines);
            auto& grey = layer<0>(image_branch);
            auto& cost = layer<1>(image_branch);
            auto& pool = layer<2>(image_branch);
            set_all(device, layer<5>(image_branch).biases.parameters, 0);
            set_all(device, layer<6>(image_branch).biases.parameters, 0);
            set_all(device, layer<7>(image_branch).biases.parameters, 0);
            set_all(device, layer<7>(image_branch).weights.parameters, 0);
            auto& pair_difference = layer<9>(image_branch);
            set_all(device, pair_difference.weights.parameters, 0);
            set_all(device, pair_difference.biases.parameters, 0);
            for(TI action = 0; action < ACTION_DIM; action++){
                set(device, pair_difference.weights.parameters, static_cast<T>(0.5), action, action);
                set(device, pair_difference.weights.parameters, static_cast<T>(-0.5), action, action + ACTION_DIM);
            }
            auto& known = layer<1>(get<1>(actor.pipelines));
            using STATE_WEIGHTS_SHAPE = typename decltype(known.weights.parameters)::SPEC::SHAPE;
            static_assert(get<0>(STATE_WEIGHTS_SHAPE{}) == 4 && get<1>(STATE_WEIGHTS_SHAPE{}) == 65);
            set_all(device, known.weights.parameters, 0);
            set_all(device, known.biases.parameters, 0);
            const T gravity = SPEC::GRAVITY;
            set(device, known.weights.parameters, static_cast<T>(1), 0, 6);
            set(device, known.weights.parameters, static_cast<T>(-0.02) * gravity, 0, 40);
            set(device, known.weights.parameters, static_cast<T>(-0.04), 0, 64);
            set(device, known.weights.parameters, static_cast<T>(-1.5), 1, 1);
            set(device, known.weights.parameters, static_cast<T>(3) / gravity, 1, 63);
            set(device, known.weights.parameters, static_cast<T>(1.5), 2, 0);
            set(device, known.weights.parameters, static_cast<T>(-3) / gravity, 2, 62);
            set_all(device, actor.head.input_layer.weights.parameters, 0);
            set_all(device, actor.head.input_layer.biases.parameters, 0);
            set_all(device, actor.head.output_layer.weights.parameters, 0);
            set_all(device, actor.head.output_layer.biases.parameters, 0);
            for(TI action = 0; action < ACTION_DIM; action++){
                set(device, actor.head.input_layer.weights.parameters, static_cast<T>(1), action, action);
                set(device, actor.head.input_layer.weights.parameters, static_cast<T>(1), action, ACTION_DIM + action);
                set(device, actor.head.output_layer.weights.parameters, static_cast<T>(1), action, action);
            }
            set_all(device, grey.weights.parameters, 0);
            set_all(device, grey.biases.parameters, 0);
            set_all(device, cost.weights.parameters, 0);
            set_all(device, cost.biases.parameters, 0);
            set_all(device, pool.weights.parameters, 0);
            set_all(device, pool.biases.parameters, 0);
            auto& normalization = layer<3>(image_branch);
            auto& difference = layer<4>(image_branch);
            set_all(device, normalization.weights.parameters, 0);
            set_all(device, normalization.biases.parameters, 0);
            set_all(device, difference.weights.parameters, 0);
            set_all(device, difference.biases.parameters, 0);
            const T scales[5] = {0.5, 2, 8, 32, 128};
            for(TI feature = 0; feature < 25; feature++){
                for(TI level = 0; level < 5; level++){
                    for(TI direction = 0; direction < 2; direction++){
                        const TI channel = (feature * 5 + level) * 2 + direction;
                        set(device, normalization.weights.parameters, -scales[level], channel, 0, 0, feature * 2 + direction);
                        set(device, difference.weights.parameters, static_cast<T>(1), feature * 2 + direction, 0, 0, channel);
                    }
                }
            }
            const TI frame_channels[2] = {0, 3};
            const T luminance[3] = {0.299, 0.587, 0.114};
            for(TI frame = 0; frame < 2; frame++){
                for(TI y = 0; y < 2; y++){
                    for(TI x = 0; x < 2; x++){
                        for(TI channel = 0; channel < 3; channel++){
                            set(device, grey.weights.parameters, luminance[channel] / static_cast<T>(4), frame, y, x, frame_channels[frame] + channel);
                        }
                    }
                }
            }
            for(TI frame = 0; frame < 1; frame++){
                for(TI y = 0; y < 5; y++){
                    for(TI x = 0; x < 5; x++){
                        const TI feature = frame * 25 + y * 5 + x;
                        for(TI direction = 0; direction < 2; direction++){
                            const TI current = direction == 0 ? 0 : frame + 1;
                            const TI reference = direction == 0 ? frame + 1 : 0;
                            for(TI sign = 0; sign < 2; sign++){
                                const TI output = feature * 4 + direction * 2 + sign;
                                const T scale = sign == 0 ? static_cast<T>(10) : static_cast<T>(-10);
                                set(device, cost.weights.parameters, scale, output, 2, 2, current);
                                set(device, cost.weights.parameters, -scale, output, y, x, reference);
                                for(TI py = 0; py < 2; py++){
                                    for(TI px = 0; px < 2; px++){
                                        const T interval_scale = 1;
                                        set(device, pool.weights.parameters, static_cast<T>(1) / 4 * interval_scale, feature * 2 + direction, py, px, output);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        {
            auto& state_normalization = layer<0>(get<1>(actor.pipelines));
            const T gravity = SPEC::GRAVITY;
            constexpr TI acceleration_offset = SPEC::BASE_OBSERVATION_DIM - 3 * SPEC::ACTION_HISTORY_LENGTH;
            for(TI item = 0; item < SPEC::ACTION_HISTORY_LENGTH; item++){
                for(TI axis = 0; axis < 3; axis++){
                    set(device, state_normalization.mean.parameters, axis == 2 ? gravity : static_cast<T>(0), acceleration_offset + 3 * item + axis);
                    set(device, state_normalization.precision.parameters, static_cast<T>(1) / gravity, acceleration_offset + 3 * item + axis);
                }
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
