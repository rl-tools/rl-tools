#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_MIXED_IMITATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_MIXED_IMITATION_H

#include "mixed_imitation.h"
#include "../../../containers/matrix/matrix.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_imitation_sample(DEVICE&, const SPEC&, typename SPEC::TI row){
        return row % SPEC::GROUP_SIZE < SPEC::IMITATION_PER_GROUP;
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT bool use_teacher_action(DEVICE& device, const SPEC& spec, typename SPEC::TI row, typename SPEC::TI optimizer_updates){
        return is_imitation_sample(device, spec, row) && optimizer_updates < SPEC::PARAMETERS::TEACHER_FORCING_UPDATES;
    }

    // The incoming derivative contains the RL subset mean, scaled for accumulation. IL rows
    // replace it with half-MSE derivatives; the chosen weight is held constant in backward.
    template <typename DEVICE, typename SPEC, typename PREDICTIONS_SPEC, typename TARGETS_SPEC, typename GRADIENT_SPEC, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void mix_policy_output_gradients(DEVICE& device, const SPEC& spec, const Matrix<PREDICTIONS_SPEC>& predictions, const Matrix<TARGETS_SPEC>& targets, Matrix<GRADIENT_SPEC>& gradient, rl::algorithms::ppo::mixed_imitation::Metrics<SPEC>& metrics, typename SPEC::T accumulation_scale, Mode<MODE>){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using PARAMETERS = typename SPEC::PARAMETERS;
        static_assert(SPEC::IMITATION_BATCH_SIZE > 0, "Mixed learning requires IL and RL in every minibatch");
        static_assert(PREDICTIONS_SPEC::ROWS == SPEC::BATCH_SIZE && TARGETS_SPEC::ROWS == SPEC::BATCH_SIZE && GRADIENT_SPEC::ROWS == SPEC::BATCH_SIZE);
        static_assert(PREDICTIONS_SPEC::COLS == TARGETS_SPEC::COLS && PREDICTIONS_SPEC::COLS == GRADIENT_SPEC::COLS);
        static_assert(mode::is<MODE, rl::algorithms::ppo::mixed_imitation::FixedWeight> || mode::is<MODE, rl::algorithms::ppo::mixed_imitation::FixedNormRatio>);
        constexpr TI ACTION_DIM = PREDICTIONS_SPEC::COLS;
        constexpr T IL_ELEMENTS = static_cast<T>(SPEC::IMITATION_BATCH_SIZE * ACTION_DIM);
        metrics = {};
        T rl_squared_norm = 0;
        T il_squared_norm = 0;
        for(TI row = 0; row < SPEC::BATCH_SIZE; row++){
            const bool imitation = is_imitation_sample(device, spec, row);
            for(TI action = 0; action < ACTION_DIM; action++){
                if(imitation){
                    T error = static_cast<T>(get(predictions, row, action)) - static_cast<T>(get(targets, row, action));
                    T derivative = error * accumulation_scale / IL_ELEMENTS;
                    set(gradient, row, action, derivative);
                    metrics.imitation_mse += error * error / IL_ELEMENTS;
                    il_squared_norm += derivative * derivative;
                }
                else{
                    T derivative = static_cast<T>(get(gradient, row, action));
                    rl_squared_norm += derivative * derivative;
                }
            }
        }
        metrics.rl_output_gradient_norm = math::sqrt(device.math, rl_squared_norm);
        metrics.imitation_output_gradient_norm = math::sqrt(device.math, il_squared_norm);
        metrics.imitation_weight = PARAMETERS::IMITATION_WEIGHT;
        if constexpr(mode::is<MODE, rl::algorithms::ppo::mixed_imitation::FixedNormRatio>){
            if(metrics.rl_output_gradient_norm > PARAMETERS::NORM_EPSILON && metrics.imitation_output_gradient_norm > PARAMETERS::NORM_EPSILON){
                T weight = PARAMETERS::OUTPUT_GRADIENT_NORM_RATIO * metrics.rl_output_gradient_norm / metrics.imitation_output_gradient_norm;
                metrics.weight_clamped = weight > PARAMETERS::MAX_IMITATION_WEIGHT;
                metrics.imitation_weight = math::min(device.math, weight, PARAMETERS::MAX_IMITATION_WEIGHT);
            }
            else{
                metrics.balancing_fallback = true;
            }
        }
        metrics.weighted_imitation_output_gradient_norm = metrics.imitation_weight * metrics.imitation_output_gradient_norm;
        metrics.ratio_valid = metrics.rl_output_gradient_norm > PARAMETERS::NORM_EPSILON;
        if(metrics.ratio_valid){
            metrics.output_gradient_norm_ratio = metrics.weighted_imitation_output_gradient_norm / metrics.rl_output_gradient_norm;
        }
        for(TI row = 0; row < SPEC::BATCH_SIZE; row++){
            if(is_imitation_sample(device, spec, row)){
                for(TI action = 0; action < ACTION_DIM; action++){
                    multiply(gradient, row, action, metrics.imitation_weight);
                }
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
