#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_INFERENCE_APPLICATIONS_L2F_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_INFERENCE_APPLICATIONS_L2F_OPERATIONS_GENERIC_H

#include "l2f.h"
#include "../../executor/operations_generic.h"
#include "../../../utils/string/operations_generic.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, inference::applications::L2F<SPEC>& executor){
        malloc(device, executor.input);
        malloc(device, executor.output);
        malloc(device, executor.executor);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, inference::applications::L2F<SPEC>& executor){
        free(device, executor.input);
        free(device, executor.output);
        free(device, executor.executor);
    }
    template <typename DEVICE, typename SPEC, typename POLICY, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, inference::applications::L2F<SPEC>& executor, POLICY& policy, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        constexpr T HOVERING_THROTTLE = 0.66;
        for(TI step_i = 0; step_i < SPEC::ACTION_HISTORY_LENGTH; step_i++){
            for(TI action_i = 0; action_i < SPEC::OUTPUT_DIM; action_i++){
                executor.action_history[step_i][action_i] = HOVERING_THROTTLE * 2 - 1;
            }
        }
        executor.steps_since_original_control_step = 0;
        reset(device, executor.executor, policy, rng);
    }
    namespace inference::applications::l2f{
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool extract_observation_from_meta(const char* meta_str, TI meta_len, char* obs_out, TI obs_out_capacity, TI& obs_out_len){
            obs_out_len = 0;
            bool in_string = false;
            TI string_start = 0;
            for(TI i = 0; i < meta_len; i++){
                char c = meta_str[i];
                if(in_string){
                    if(c == '\\' && i + 1 < meta_len){ i++; continue; }
                    if(c != '"') continue;
                    in_string = false;
                    TI string_len = i - string_start;
                    TI pos = i + 1;
                    while(pos < meta_len && meta_str[pos] == ' ') pos++;
                    if(pos >= meta_len || meta_str[pos] != ':') continue;
                    if(string_len != 11 || !utils::string::compare(meta_str + string_start, "observation", (TI)11)) continue;
                    pos++;
                    while(pos < meta_len && meta_str[pos] == ' ') pos++;
                    if(pos >= meta_len || meta_str[pos] != '"') continue;
                    pos++;
                    TI val_start = pos;
                    while(pos < meta_len){
                        if(meta_str[pos] == '\\' && pos + 1 < meta_len){ pos += 2; continue; }
                        if(meta_str[pos] == '"') break;
                        pos++;
                    }
                    obs_out_len = pos - val_start;
                    if(obs_out_len >= obs_out_capacity) return false;
                    for(TI k = 0; k < obs_out_len; k++) obs_out[k] = meta_str[val_start + k];
                    obs_out[obs_out_len] = '\0';
                    return true;
                } else {
                    if(c == '"'){ in_string = true; string_start = i + 1; }
                }
            }
            return false;
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool parse_observation_string(DEVICE& device, const char* obs_str, TI obs_str_len, ObservationLayout<TI>& layout, TI output_dim){
            layout.component_count = 0;
            layout.total_dim = 0;
            layout.action_history_length = 0;
            TI token_start = 0;
            for(TI i = 0; i <= obs_str_len; i++){
                char c = (i < obs_str_len) ? obs_str[i] : '\0';
                if(c == ','){
                    utils::assert_exit(device, false, "inference::applications::l2f: multi-branch observations not supported");
                    return false;
                }
                if(c != '.' && c != '\0') continue;
                TI token_len = i - token_start;
                if(token_len == 0){ token_start = i + 1; continue; }
                if(layout.component_count >= ObservationLayout<TI>::MAX_COMPONENTS){
                    utils::assert_exit(device, false, "inference::applications::l2f: too many observation components");
                    return false;
                }
                const char* token = obs_str + token_start;
                TI name_len = token_len;
                TI param = 0;
                for(TI j = 0; j < token_len; j++){
                    if(token[j] == '('){
                        name_len = j;
                        param = utils::string::string_to_int<TI>(token + j + 1, token_len - j - 1);
                        break;
                    }
                }
                auto& comp = layout.components[layout.component_count];
                comp.parameter = param;
                comp.offset = layout.total_dim;
                bool matched = false;
                if(name_len == 8 && utils::string::compare(token, "Position", name_len)){
                    comp.type = ObservationComponentType::POSITION; comp.dim = 3; matched = true;
                } else if(name_len == 25 && utils::string::compare(token, "OrientationRotationMatrix", name_len)){
                    comp.type = ObservationComponentType::ORIENTATION_ROTATION_MATRIX; comp.dim = 9; matched = true;
                } else if(name_len == 21 && utils::string::compare(token, "OrientationQuaternion", name_len)){
                    comp.type = ObservationComponentType::ORIENTATION_QUATERNION; comp.dim = 4; matched = true;
                } else if(name_len == 14 && utils::string::compare(token, "LinearVelocity", name_len)){
                    comp.type = ObservationComponentType::LINEAR_VELOCITY; comp.dim = 3; matched = true;
                } else if(name_len == 15 && utils::string::compare(token, "AngularVelocity", name_len)){
                    comp.type = ObservationComponentType::ANGULAR_VELOCITY; comp.dim = 3; matched = true;
                } else if(name_len == 22 && utils::string::compare(token, "AngularVelocityDelayed", name_len)){
                    comp.type = ObservationComponentType::ANGULAR_VELOCITY_DELAYED; comp.dim = 3; matched = true;
                } else if(name_len == 21 && utils::string::compare(token, "LinearVelocityDelayed", name_len)){
                    comp.type = ObservationComponentType::LINEAR_VELOCITY_DELAYED; comp.dim = 3; matched = true;
                } else if(name_len == 13 && utils::string::compare(token, "ActionHistory", name_len)){
                    comp.type = ObservationComponentType::ACTION_HISTORY; comp.dim = output_dim * param; layout.action_history_length = param; matched = true;
                } else if(name_len == 27 && utils::string::compare(token, "LinearAccelerationBodyFrame", name_len)){
                    comp.type = ObservationComponentType::LINEAR_ACCELERATION_BODY_FRAME; comp.dim = 3; matched = true;
                } else if(name_len == 23 && utils::string::compare(token, "LinearVelocityBodyFrame", name_len)){
                    comp.type = ObservationComponentType::LINEAR_VELOCITY_BODY_FRAME; comp.dim = 3; matched = true;
                } else if(name_len == 11 && utils::string::compare(token, "RotorSpeeds", name_len)){
                    comp.type = ObservationComponentType::ROTOR_SPEEDS; comp.dim = output_dim; matched = true;
                }
                if(!matched){
                    utils::assert_exit(device, false, "inference::applications::l2f: unknown observation component");
                    return false;
                }
                layout.total_dim += comp.dim;
                layout.component_count++;
                token_start = i + 1;
            }
            return layout.component_count > 0;
        }
        template <typename DEVICE, typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT bool validate_observation_layout(DEVICE& device, const ObservationLayout<TI>& layout, TI action_history_capacity){
            if(layout.component_count == 0) return false;
            if(layout.action_history_length > action_history_capacity){
                utils::assert_exit(device, false, "inference::applications::l2f: action history length exceeds capacity");
                return false;
            }
            for(TI i = 0; i < layout.component_count; i++){
                auto& comp = layout.components[i];
                switch(comp.type){
                    case ObservationComponentType::ANGULAR_VELOCITY_DELAYED:
                    case ObservationComponentType::LINEAR_VELOCITY_DELAYED:
                        if(comp.parameter > 0){
                            utils::assert_exit(device, false, "inference::applications::l2f: delayed observations with delay > 0 not supported");
                            return false;
                        }
                        break;
                    case ObservationComponentType::LINEAR_ACCELERATION_BODY_FRAME:
                    case ObservationComponentType::LINEAR_VELOCITY_BODY_FRAME:
                    case ObservationComponentType::ROTOR_SPEEDS:
                        utils::assert_exit(device, false, "inference::applications::l2f: unsupported observation component");
                        return false;
                    default:
                        break;
                }
            }
            return true;
        }
        template <typename DEVICE, typename SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT bool observe_dynamic(DEVICE& device, L2F<SPEC>& executor, Observation<SPEC>& observation, dyn::Tensor<dyn::TensorSpecification<typename SPEC::TI>>& observation_flat){
            using TI = typename SPEC::TI;
            auto& layout = executor.observation_layout;
            for(TI comp_i = 0; comp_i < layout.component_count; comp_i++){
                auto& comp = layout.components[comp_i];
                switch(comp.type){
                    case ObservationComponentType::POSITION:
                        if(!observation.position_set) return false;
                        for(TI i = 0; i < 3; i++) dyn::set(device, observation_flat, comp.offset + i, observation.position[i]);
                        break;
                    case ObservationComponentType::ORIENTATION_ROTATION_MATRIX:{
                        if(!observation.orientation_set) return false;
                        float qw = observation.orientation[0], qx = observation.orientation[1], qy = observation.orientation[2], qz = observation.orientation[3];
                        dyn::set(device, observation_flat, comp.offset + 0, (1 - 2*qy*qy - 2*qz*qz));
                        dyn::set(device, observation_flat, comp.offset + 1, (    2*qx*qy - 2*qw*qz));
                        dyn::set(device, observation_flat, comp.offset + 2, (    2*qx*qz + 2*qw*qy));
                        dyn::set(device, observation_flat, comp.offset + 3, (    2*qx*qy + 2*qw*qz));
                        dyn::set(device, observation_flat, comp.offset + 4, (1 - 2*qx*qx - 2*qz*qz));
                        dyn::set(device, observation_flat, comp.offset + 5, (    2*qy*qz - 2*qw*qx));
                        dyn::set(device, observation_flat, comp.offset + 6, (    2*qx*qz - 2*qw*qy));
                        dyn::set(device, observation_flat, comp.offset + 7, (    2*qy*qz + 2*qw*qx));
                        dyn::set(device, observation_flat, comp.offset + 8, (1 - 2*qx*qx - 2*qy*qy));
                        break;
                    }
                    case ObservationComponentType::ORIENTATION_QUATERNION:
                        if(!observation.orientation_set) return false;
                        for(TI i = 0; i < 4; i++) dyn::set(device, observation_flat, comp.offset + i, observation.orientation[i]);
                        break;
                    case ObservationComponentType::LINEAR_VELOCITY:
                    case ObservationComponentType::LINEAR_VELOCITY_DELAYED:
                        if(!observation.linear_velocity_set) return false;
                        for(TI i = 0; i < 3; i++) dyn::set(device, observation_flat, comp.offset + i, observation.linear_velocity[i]);
                        break;
                    case ObservationComponentType::ANGULAR_VELOCITY:
                    case ObservationComponentType::ANGULAR_VELOCITY_DELAYED:
                        if(!observation.angular_velocity_set) return false;
                        for(TI i = 0; i < 3; i++) dyn::set(device, observation_flat, comp.offset + i, observation.angular_velocity[i]);
                        break;
                    case ObservationComponentType::ACTION_HISTORY:
                        for(TI step_i = 0; step_i < comp.parameter; step_i++){
                            for(TI action_i = 0; action_i < SPEC::OUTPUT_DIM; action_i++){
                                dyn::set(device, observation_flat, comp.offset + step_i * SPEC::OUTPUT_DIM + action_i, executor.action_history[step_i][action_i]);
                            }
                        }
                        break;
                    default:
                        return false;
                }
            }
            return true;
        }
        template <typename DEVICE, typename SPEC, typename OBS_SPEC>
        RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, L2F<SPEC>& executor, Observation<SPEC>& observation, Tensor<OBS_SPEC>& observation_flat){
            using TI = typename DEVICE::index_t;
            static_assert(OBS_SPEC::SHAPE::template GET<0> == 1);
            static_assert(OBS_SPEC::SHAPE::template GET<1> == 18 + SPEC::OUTPUT_DIM * SPEC::ACTION_HISTORY_LENGTH); // position + orientation + linear_velocity + angular_velocity + action_history
            TI base = 0;
            set(device, observation_flat, observation.position[0], 0,  base++);
            set(device, observation_flat, observation.position[1], 0,  base++);
            set(device, observation_flat, observation.position[2], 0,  base++);
            float qw = observation.orientation[0];
            float qx = observation.orientation[1];
            float qy = observation.orientation[2];
            float qz = observation.orientation[3];
            set(device, observation_flat,   (1 - 2*qy*qy - 2*qz*qz), 0, base++);
            set(device, observation_flat,   (    2*qx*qy - 2*qw*qz), 0, base++);
            set(device, observation_flat,   (    2*qx*qz + 2*qw*qy), 0, base++);
            set(device, observation_flat,   (    2*qx*qy + 2*qw*qz), 0, base++);
            set(device, observation_flat,   (1 - 2*qx*qx - 2*qz*qz), 0, base++);
            set(device, observation_flat,   (    2*qy*qz - 2*qw*qx), 0, base++);
            set(device, observation_flat,   (    2*qx*qz - 2*qw*qy), 0, base++);
            set(device, observation_flat,   (    2*qy*qz + 2*qw*qx), 0, base++);
            set(device, observation_flat,   (1 - 2*qx*qx - 2*qy*qy), 0, base++);
            set(device, observation_flat, observation.linear_velocity[0], 0, base++);
            set(device, observation_flat, observation.linear_velocity[1], 0, base++);
            set(device, observation_flat, observation.linear_velocity[2], 0, base++);
            set(device, observation_flat, observation.angular_velocity[0], 0, base++);
            set(device, observation_flat, observation.angular_velocity[1], 0, base++);
            set(device, observation_flat, observation.angular_velocity[2], 0, base++);
            for(TI step_i = 0; step_i < SPEC::ACTION_HISTORY_LENGTH; step_i++){
                for(TI action_i = 0; action_i < SPEC::OUTPUT_DIM; action_i++){
                    set(device, observation_flat, executor.action_history[step_i][action_i], 0, base++);
                }
            }
        }
    }
    template <typename DEVICE, typename SPEC, typename POLICY, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT auto control(DEVICE& device, inference::applications::L2F<SPEC>& executor, typename SPEC::TIMESTAMP nanoseconds, POLICY& policy, inference::applications::l2f::Observation<SPEC>& observation, inference::applications::l2f::Action<SPEC>& action, RNG& rng){
        using TI = typename SPEC::TI;
        if(executor.steps_since_original_control_step == 0){
            for(TI action_i = 0; action_i < SPEC::OUTPUT_DIM; action_i++){
                executor.action_history[0][action_i] = observation.previous_action[action_i];
            }
        }
        else{
            for(TI action_i = 0; action_i < SPEC::OUTPUT_DIM; action_i++){
                executor.action_history[0][action_i] = (executor.action_history[0][action_i] * executor.steps_since_original_control_step + observation.previous_action[action_i]) / (executor.steps_since_original_control_step + 1);
            }
        }
        inference::applications::l2f::observe(device, executor, observation, executor.input);
        auto status = control(device, executor.executor, nanoseconds, policy, executor.input, executor.output, rng);
        for (TI output_i=0; output_i < SPEC::OUTPUT_DIM; output_i++){
            action.action[output_i] = get(device, executor.output, 0, output_i);
        }

        executor.steps_since_original_control_step++; // gets overwritten with 0 in the case of an original control step
        if(status.source == decltype(status.source)::CONTROL){
            if(status.step_type == decltype(status.step_type)::NATIVE){
                // step action history
                static_assert(SPEC::ACTION_HISTORY_LENGTH >= 1);
                for(TI step_i = SPEC::ACTION_HISTORY_LENGTH-1; step_i > 0; step_i--){
                    for(TI action_i = 0; action_i < SPEC::OUTPUT_DIM; action_i++){
                        executor.action_history[step_i][action_i] = executor.action_history[step_i-1][action_i];
                    }
                }
                executor.steps_since_original_control_step = 0;
            }
        }
        return status;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
