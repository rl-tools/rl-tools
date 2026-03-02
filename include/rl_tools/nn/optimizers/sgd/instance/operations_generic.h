#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_SGD_INSTANCE_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_SGD_INSTANCE_OPERATIONS_GENERIC_H

#include "../sgd.h"
#include "../../../../nn/parameters/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, nn::parameters::SGD::Instance<SPEC>& p){
        malloc(device, (nn::parameters::Gradient::Instance<SPEC>&) p);
        malloc(device, p.velocity);
        if constexpr(nn::parameters::SGD::Instance<SPEC>::USE_MASTER_PARAMETERS){
            malloc(device, p.master_parameters);
        }
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, nn::parameters::SGD::Instance<SPEC>& p){
        free(device, (nn::parameters::Gradient::Instance<SPEC>&) p);
        free(device, p.velocity);
        if constexpr(nn::parameters::SGD::Instance<SPEC>::USE_MASTER_PARAMETERS){
            free(device, p.master_parameters);
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void update(DEVICE& device, nn::parameters::SGD::Instance<PARAMETER_SPEC>& parameter, nn::optimizers::SGD<SPEC>& optimizer){
        // SGD with momentum: v = momentum * v + gradient (+ weight_decay * param)
        // Nesterov: param -= lr * (momentum * v + gradient)
        // Classical: param -= lr * v
        using TI = typename DEVICE::index_t;
        using T_VELOCITY = typename nn::parameters::SGD::Instance<PARAMETER_SPEC>::T_VELOCITY;
        using T_PARAMETER = typename decltype(parameter.parameters)::T;
        using T_MASTER_PARAMETER = typename nn::parameters::SGD::Instance<PARAMETER_SPEC>::T_MASTER_PARAMETER;
        const auto& optimizer_parameters = get(device, optimizer.parameters, 0);
        auto params = matrix_view(device, parameter.parameters);
        if constexpr(nn::parameters::SGD::Instance<PARAMETER_SPEC>::USE_MASTER_PARAMETERS){
            auto master_params = matrix_view(device, parameter.master_parameters);
            auto grad = matrix_view(device, parameter.gradient);
            auto vel = matrix_view(device, parameter.velocity);
            constexpr TI ROWS = decltype(params)::ROWS;
            constexpr TI COLS = decltype(params)::COLS;
            for(TI row_i = 0; row_i < ROWS; row_i++){
                for(TI col_i = 0; col_i < COLS; col_i++){
                    T_VELOCITY g = get(grad, row_i, col_i);
                    if constexpr(SPEC::ENABLE_WEIGHT_DECAY){
                        if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::CATEGORY_TAG, nn::parameters::categories::Weights>){
                            g += (T_VELOCITY)get(master_params, row_i, col_i) * optimizer_parameters.weight_decay;
                        }
                    }
                    T_VELOCITY v = optimizer_parameters.momentum * get(vel, row_i, col_i) + g;
                    set(vel, row_i, col_i, v);
                    T_VELOCITY param_update;
                    if(optimizer_parameters.nesterov){
                        param_update = optimizer_parameters.momentum * v + g;
                    }
                    else{
                        param_update = v;
                    }
                    T_VELOCITY value = (T_VELOCITY)get(master_params, row_i, col_i);
                    value -= optimizer_parameters.learning_rate * param_update;
                    set(master_params, row_i, col_i, (T_MASTER_PARAMETER)value);
                    set(params, row_i, col_i, (T_PARAMETER)value);
                }
            }
            return;
        }
        auto grad = matrix_view(device, parameter.gradient);
        auto vel = matrix_view(device, parameter.velocity);
        constexpr TI ROWS = decltype(params)::ROWS;
        constexpr TI COLS = decltype(params)::COLS;
        for(TI row_i = 0; row_i < ROWS; row_i++){
            for(TI col_i = 0; col_i < COLS; col_i++){
                T_VELOCITY g = get(grad, row_i, col_i);
                if constexpr(SPEC::ENABLE_WEIGHT_DECAY){
                    if constexpr(utils::typing::is_same_v<typename PARAMETER_SPEC::CATEGORY_TAG, nn::parameters::categories::Weights>){
                        g += get(params, row_i, col_i) * optimizer_parameters.weight_decay;
                    }
                }
                T_VELOCITY v = optimizer_parameters.momentum * get(vel, row_i, col_i) + g;
                set(vel, row_i, col_i, v);
                T_VELOCITY param_update;
                if(optimizer_parameters.nesterov){
                    param_update = optimizer_parameters.momentum * v + g;
                }
                else{
                    param_update = v;
                }
                T_VELOCITY value = get(params, row_i, col_i);
                value -= optimizer_parameters.learning_rate * param_update;
                set(params, row_i, col_i, (T_PARAMETER)value);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename PARAMETER_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void _reset_optimizer_state(DEVICE& device, nn::parameters::SGD::Instance<PARAMETER_SPEC>& parameter, nn::optimizers::SGD<SPEC>& optimizer){
        set_all(device, parameter.velocity, 0);
        if constexpr(nn::parameters::SGD::Instance<PARAMETER_SPEC>::USE_MASTER_PARAMETERS){
            copy(device, device, parameter.parameters, parameter.master_parameters);
        }
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const nn::parameters::SGD::Instance<SOURCE_SPEC>& source, nn::parameters::SGD::Instance<TARGET_SPEC>& target){
        copy(source_device, target_device, (const nn::parameters::Gradient::Instance<SOURCE_SPEC>&) source, (nn::parameters::Gradient::Instance<TARGET_SPEC>&) target);
        copy(source_device, target_device, source.velocity, target.velocity);
        if constexpr(nn::parameters::SGD::Instance<SOURCE_SPEC>::USE_MASTER_PARAMETERS && nn::parameters::SGD::Instance<TARGET_SPEC>::USE_MASTER_PARAMETERS){
            copy(source_device, target_device, source.master_parameters, target.master_parameters);
        }
        else if constexpr(!nn::parameters::SGD::Instance<SOURCE_SPEC>::USE_MASTER_PARAMETERS && nn::parameters::SGD::Instance<TARGET_SPEC>::USE_MASTER_PARAMETERS){
            copy(source_device, target_device, source.parameters, target.master_parameters);
        }
    }
    template<typename DEVICE, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC_1::TYPE_POLICY::DEFAULT abs_diff(DEVICE& device, const nn::parameters::SGD::Instance<SPEC_1>& p1, const nn::parameters::SGD::Instance<SPEC_2>& p2){
        typename SPEC_1::TYPE_POLICY::DEFAULT acc = 0;
        acc += abs_diff(device, static_cast<const nn::parameters::Gradient::Instance<SPEC_1>&>(p1), static_cast<const nn::parameters::Gradient::Instance<SPEC_2>&>(p2));
        acc += abs_diff(device, p1.velocity, p2.velocity);
        if constexpr(nn::parameters::SGD::Instance<SPEC_1>::USE_MASTER_PARAMETERS && nn::parameters::SGD::Instance<SPEC_2>::USE_MASTER_PARAMETERS){
            acc += abs_diff(device, p1.master_parameters, p2.master_parameters);
        }
        else if constexpr(nn::parameters::SGD::Instance<SPEC_1>::USE_MASTER_PARAMETERS){
            acc += abs_diff(device, p1.master_parameters, p2.parameters);
        }
        else if constexpr(nn::parameters::SGD::Instance<SPEC_2>::USE_MASTER_PARAMETERS){
            acc += abs_diff(device, p1.parameters, p2.master_parameters);
        }
        return acc;
    }
    template<typename DEVICE, typename SPEC, typename MODE = Mode<mode::Default<>>>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(DEVICE& device, const nn::parameters::SGD::Instance<SPEC>& p, const Mode<MODE>& mode = {}){
        bool upstream_nan = is_nan(device, static_cast<const nn::parameters::Gradient::Instance<SPEC>&>(p), mode);
        if constexpr(mode::is<MODE, nn::parameters::mode::ParametersOnly>){
            return upstream_nan;
        }
        bool downstream_nan = is_nan(device, p.velocity, mode);
        if constexpr(nn::parameters::SGD::Instance<SPEC>::USE_MASTER_PARAMETERS){
            downstream_nan = downstream_nan || is_nan(device, p.master_parameters, mode);
        }
        return upstream_nan || downstream_nan;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
