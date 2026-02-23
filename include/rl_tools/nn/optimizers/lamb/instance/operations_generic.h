#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_LAMB_INSTANCE_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_LAMB_INSTANCE_OPERATIONS_GENERIC_H

#include "../lamb.h"
#include "../../adam/instance/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename PARAMETER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::parameters::Adam::Instance<PARAMETER_SPEC>& parameter, nn::optimizers::Lamb<SPEC>& optimizer){
        _reset_optimizer_state(device, parameter, static_cast<nn::optimizers::Adam<SPEC>&>(optimizer));
    }

    template<typename DEVICE, typename SPEC, typename PARAMETER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void gradient_descent(DEVICE& device, nn::parameters::Adam::Instance<PARAMETER_SPEC>& parameter, nn::optimizers::Lamb<SPEC>& optimizer){
        using TI = typename DEVICE::index_t;
        using T_OPTIMIZER = typename PARAMETER_SPEC::TYPE_POLICY::template GET<numeric_types::categories::OptimizerState>;
        using T_PARAMETER = typename decltype(parameter.parameters)::T;
        const auto& optimizer_parameters = get(device, optimizer.parameters, 0);
        const auto& lamb_params = get(device, optimizer.lamb_parameters, 0);
        auto parameters = matrix_view(device, parameter.parameters);
        auto gradient_first_order_moment = matrix_view(device, parameter.gradient_first_order_moment);
        auto gradient_second_order_moment = matrix_view(device, parameter.gradient_second_order_moment);
        constexpr TI ROWS = decltype(parameters)::ROWS;
        constexpr TI COLS = decltype(parameters)::COLS;

        T_OPTIMIZER weight_norm_sq = 0;
        T_OPTIMIZER update_norm_sq = 0;
        for(TI row_i = 0; row_i < ROWS; row_i++){
            for(TI col_i = 0; col_i < COLS; col_i++){
                T_OPTIMIZER w = get(parameters, row_i, col_i);
                weight_norm_sq += w * w;

                T_OPTIMIZER pre_sqrt_term = get(gradient_second_order_moment, row_i, col_i) * get(device, optimizer.second_order_moment_bias_correction, 0);
                pre_sqrt_term = math::max(device.math, pre_sqrt_term, (T_OPTIMIZER)optimizer_parameters.epsilon_sqrt);
                T_OPTIMIZER r = get(device, optimizer.first_order_moment_bias_correction, 0) * get(gradient_first_order_moment, row_i, col_i) / (math::sqrt(device.math, pre_sqrt_term) + optimizer_parameters.epsilon);
                if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::CATEGORY_TAG, nn::parameters::categories::Weights>){
                    if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::GROUP_TAG, nn::parameters::groups::Normal> && SPEC::ENABLE_WEIGHT_DECAY){
                        r += w * optimizer_parameters.weight_decay / 2;
                    }
                    if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::GROUP_TAG, nn::parameters::groups::Input> && SPEC::ENABLE_WEIGHT_DECAY){
                        r += w * optimizer_parameters.weight_decay_input / 2;
                    }
                    if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::GROUP_TAG, nn::parameters::groups::Output> && SPEC::ENABLE_WEIGHT_DECAY){
                        r += w * optimizer_parameters.weight_decay_output / 2;
                    }
                }
                update_norm_sq += r * r;
            }
        }

        T_OPTIMIZER weight_norm = math::sqrt(device.math, weight_norm_sq);
        T_OPTIMIZER update_norm = math::sqrt(device.math, update_norm_sq);

        T_OPTIMIZER trust_ratio = 1;
        if(weight_norm > 0 && update_norm > 0){
            trust_ratio = weight_norm / update_norm;
            trust_ratio = math::min(device.math, trust_ratio, lamb_params.upper_bound_trust_ratio);
            trust_ratio = math::max(device.math, trust_ratio, lamb_params.lower_bound_trust_ratio);
        }

        for(TI row_i = 0; row_i < ROWS; row_i++){
            for(TI col_i = 0; col_i < COLS; col_i++){
                T_OPTIMIZER pre_sqrt_term = get(gradient_second_order_moment, row_i, col_i) * get(device, optimizer.second_order_moment_bias_correction, 0);
                pre_sqrt_term = math::max(device.math, pre_sqrt_term, (T_OPTIMIZER)optimizer_parameters.epsilon_sqrt);
                T_OPTIMIZER r = get(device, optimizer.first_order_moment_bias_correction, 0) * get(gradient_first_order_moment, row_i, col_i) / (math::sqrt(device.math, pre_sqrt_term) + optimizer_parameters.epsilon);
                if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::CATEGORY_TAG, nn::parameters::categories::Biases> && SPEC::ENABLE_BIAS_LR_FACTOR){
                    r *= optimizer_parameters.bias_lr_factor;
                }
                if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::CATEGORY_TAG, nn::parameters::categories::Weights>){
                    if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::GROUP_TAG, nn::parameters::groups::Normal> && SPEC::ENABLE_WEIGHT_DECAY){
                        r += get(parameters, row_i, col_i) * optimizer_parameters.weight_decay / 2;
                    }
                    if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::GROUP_TAG, nn::parameters::groups::Input> && SPEC::ENABLE_WEIGHT_DECAY){
                        r += get(parameters, row_i, col_i) * optimizer_parameters.weight_decay_input / 2;
                    }
                    if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::GROUP_TAG, nn::parameters::groups::Output> && SPEC::ENABLE_WEIGHT_DECAY){
                        r += get(parameters, row_i, col_i) * optimizer_parameters.weight_decay_output / 2;
                    }
                }
                T_OPTIMIZER value = get(parameters, row_i, col_i);
                value -= optimizer_parameters.alpha * trust_ratio * r;
                set(parameters, row_i, col_i, (T_PARAMETER)value);
            }
        }
    }

    template<typename DEVICE, typename SPEC, typename LAMB_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::parameters::Adam::Instance<SPEC>& parameter, nn::optimizers::Lamb<LAMB_SPEC>& optimizer){
        using PARAMETERS = typename LAMB_SPEC::DEFAULT_PARAMETERS;
        const auto& optimizer_parameters = get(device, optimizer.parameters, 0);
        utils::polyak::update(device, parameter.gradient, parameter.gradient_first_order_moment, optimizer_parameters.beta_1, PARAMETERS::ENABLE_GRADIENT_CLIPPING, PARAMETERS::GRADIENT_CLIP_VALUE);
        utils::polyak::update_squared(device, parameter.gradient, parameter.gradient_second_order_moment, optimizer_parameters.beta_2, PARAMETERS::ENABLE_GRADIENT_CLIPPING, PARAMETERS::GRADIENT_CLIP_VALUE);
        gradient_descent(device, parameter, optimizer);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
