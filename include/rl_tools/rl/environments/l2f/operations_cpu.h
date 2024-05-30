#ifndef LEARNING_TO_FLY_SIMULATOR_OPERATIONS_CPU_H
#define LEARNING_TO_FLY_SIMULATOR_OPERATIONS_CPU_H
#include "operations_generic.h"

#include <random>
#ifdef RL_TOOLS_ENABLE_JSON
#include <nlohmann/json.hpp>
#endif
#include <string>
namespace rl_tools{
#ifdef RL_TOOLS_ENABLE_JSON
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::Dynamics& parameters, nlohmann::json config){
        if(config.contains("dynamics")){
            if(config["dynamics"].contains("model")){
                std::string config_model = config["dynamics"]["model"];
                utils::assert_exit(device, config_model == rl::environments::multirotor::parameters::dynamics::registry_name<SPEC>, "Model \"" + config_model + "\" in config file does not match model \"" + rl::environments::multirotor::parameters::dynamics::registry_name<SPEC> + "\" in parameters file");
            }
            if(config["dynamics"].contains("n_rotors")){
                typename SPEC::TI num_rotors = config["dynamics"]["n_rotors"];
                utils::assert_exit(device, num_rotors == SPEC::N, "Rotor number in config file does not match number of rotors in parameters file");
                if(config["dynamics"].contains("rotor_positions")){
                    auto rotor_positions = config["dynamics"]["rotor_positions"];
                    utils::assert_exit(device, rotor_positions.size() == SPEC::N, "Rotor number in config file does not match number of rotors in parameters file");
                    for(typename SPEC::TI rotor_i = 0; rotor_i < SPEC::N; rotor_i++){
                        utils::assert_exit(device, rotor_positions[rotor_i].size() == 3, "Positions are 3 dimensional");
                        for(typename SPEC::TI dim_i = 0; dim_i < 3; dim_i++){
                            parameters.rotor_positions[rotor_i][dim_i] = rotor_positions[rotor_i][dim_i];
                        }
                    }
                }
                if(config["dynamics"].contains("rotor_thrust_directions")){
                    auto rotor_thrust_directions = config["dynamics"]["rotor_thrust_directions"];
                    utils::assert_exit(device, rotor_thrust_directions.size() == SPEC::N, "Rotor number in config file does not match number of rotors in parameters file");
                    for(typename SPEC::TI rotor_i = 0; rotor_i < SPEC::N; rotor_i++){
                        utils::assert_exit(device, rotor_thrust_directions[rotor_i].size() == 3, "Positions are 3 dimensional");
                        for(typename SPEC::TI dim_i = 0; dim_i < 3; dim_i++){
                            parameters.rotor_thrust_directions[rotor_i][dim_i] = rotor_thrust_directions[rotor_i][dim_i];
                        }
                    }
                }
                if(config["dynamics"].contains("rotor_torque_directions")){
                    auto rotor_torque_directions = config["dynamics"]["rotor_torque_directions"];
                    utils::assert_exit(device, rotor_torque_directions.size() == SPEC::N, "Rotor number in config file does not match number of rotors in parameters file");
                    for(typename SPEC::TI rotor_i = 0; rotor_i < SPEC::N; rotor_i++){
                        utils::assert_exit(device, rotor_torque_directions[rotor_i].size() == 3, "Positions are 3 dimensional");
                        for(typename SPEC::TI dim_i = 0; dim_i < 3; dim_i++){
                            parameters.rotor_torque_directions[rotor_i][dim_i] = rotor_torque_directions[rotor_i][dim_i];
                        }
                    }
                }
                if(config["dynamics"].contains("rotor_thrust_coefficients")){
                    auto rotor_thrust_coefficients = config["dynamics"]["rotor_thrust_coefficients"];
                    utils::assert_exit(device, rotor_thrust_coefficients.size() == 3, "Please provide three orders (0, 1, 2) of thrust coefficients");
                    for(typename SPEC::TI order_i=0; order_i < 3; order_i++){
                        parameters.rotor_thrust_coefficients[order_i] = rotor_thrust_coefficients[order_i];
                    }
                }
                if(config["dynamics"].contains("rotor_torque_constant")){
                    parameters.rotor_torque_constant = config["dynamics"]["rotor_torque_constant"];
                }
                if(config["dynamics"].contains("mass")){
                    parameters.mass = config["dynamics"]["mass"];
                }
                if(config["dynamics"].contains("gravity")){
                    auto gravity = config["dynamics"]["gravity"];
                    utils::assert_exit(device, gravity.size() == 3, "Gravity is three dimensional");
                    for(typename SPEC::TI dim_i=0; dim_i < 3; dim_i++){
                        parameters.gravity[dim_i] = gravity[dim_i];
                    }
                }
                if(config["dynamics"].contains("J")){
                    auto J = config["dynamics"]["J"];
                    utils::assert_exit(device, J.size() == 3, "The moment of inertia matrix should be 3x3");
                    for(typename SPEC::TI row_i = 0; row_i < 3; row_i++){
                        utils::assert_exit(device, J[row_i].size() == 3, "The moment of inertia matrix should be 3x3");
                        for(typename SPEC::TI col_i = 0; col_i < 3; col_i++){
                            parameters.J[row_i][col_i] = J[row_i][col_i];
                        }
                    }
                }
                if(config["dynamics"].contains("J_inv")){
                    auto J_inv = config["dynamics"]["J_inv"];
                    utils::assert_exit(device, J_inv.size() == 3, "The moment of inertia matrix should be 3x3");
                    for(typename SPEC::TI row_i = 0; row_i < 3; row_i++){
                        utils::assert_exit(device, J_inv[row_i].size() == 3, "The moment of inertia matrix should be 3x3");
                        for(typename SPEC::TI col_i = 0; col_i < 3; col_i++){
                            parameters.J_inv[row_i][col_i] = J_inv[row_i][col_i];
                        }
                    }
                }
                if(config["dynamics"].contains("motor_time_constant")){
                    parameters.motor_time_constant = config["dynamics"]["motor_time_constant"];
                }
                if(config["dynamics"].contains("hovering_throttle_relative")){
                    parameters.hovering_throttle_relative = config["dynamics"]["hovering_throttle_relative"];
                }
                if(config["dynamics"].contains("action_limit")){
                    auto action_limit = config["dynamics"]["action_limit"];
                    if(action_limit.contains("upper_bound")){
                        parameters.action_limit.max = action_limit["upper_bound"];
                    }
                    if(action_limit.contains("lower_bound")){
                        parameters.action_limit.min = action_limit["lower_bound"];
                    }
                }
            }
            else{
                std::cout << "Config file does not contain rotor number, skipping..." << std::endl;
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::parameters::reward_functions::Absolute<typename SPEC::T>& parameters, nlohmann::json config){
        if(config.contains("mdp")){
            auto mdp_json = config["mdp"];
            if(mdp_json.contains("reward")){
                auto reward_json = mdp_json["reward"];
                utils::assert_exit(device, reward_json.contains("type"), "Parameters file does not contain reward type");
                utils::assert_exit(device, reward_json["type"] == name(device, parameters), "Parameters file reward type " + std::string(reward_json["type"]) + "  is not matching " + std::string(name(device, parameters)));

                if(reward_json.contains("scale")){
                    parameters.scale = reward_json["scale"];
                }
                if(reward_json.contains("constant")){
                    parameters.constant = reward_json["constant"];
                }
                if(reward_json.contains("termination_penalty")){
                    parameters.termination_penalty = reward_json["termination_penalty"];
                }
                if(reward_json.contains("position")){
                    parameters.position = reward_json["position"];
                }
                if(reward_json.contains("orientation")){
                    parameters.orientation = reward_json["orientation"];
                }
                if(reward_json.contains("linear_velocity")){
                    parameters.linear_velocity = reward_json["linear_velocity"];
                }
                if(reward_json.contains("angular_velocity")){
                    parameters.angular_velocity = reward_json["angular_velocity"];
                }
                if(reward_json.contains("linear_acceleration")){
                    parameters.linear_acceleration = reward_json["linear_acceleration"];
                }
                if(reward_json.contains("angular_acceleration")){
                    parameters.angular_acceleration = reward_json["angular_acceleration"];
                }
                if(reward_json.contains("action")){
                    parameters.action = reward_json["action"];
                }
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::parameters::reward_functions::Squared<typename SPEC::T>& parameters, nlohmann::json config){
        if(config.contains("mdp")){
            auto mdp_json = config["mdp"];
            if(mdp_json.contains("reward")){
                auto reward_json = mdp_json["reward"];
                utils::assert_exit(device, reward_json.contains("type"), "Parameters file does not contain reward type");
                utils::assert_exit(device, reward_json["type"] == name(device, parameters), "Parameters file reward type " + std::string(reward_json["type"]) + "  is not matching " + std::string(name(device, parameters)));

                if(reward_json.contains("scale")){
                    parameters.scale = reward_json["scale"];
                }
                if(reward_json.contains("constant")){
                    parameters.constant = reward_json["constant"];
                }
                if(reward_json.contains("termination_penalty")){
                    parameters.termination_penalty = reward_json["termination_penalty"];
                }
                if(reward_json.contains("position")){
                    parameters.position = reward_json["position"];
                }
                if(reward_json.contains("orientation")){
                    parameters.orientation = reward_json["orientation"];
                }
                if(reward_json.contains("linear_velocity")){
                    parameters.linear_velocity = reward_json["linear_velocity"];
                }
                if(reward_json.contains("angular_velocity")){
                    parameters.angular_velocity = reward_json["angular_velocity"];
                }
                if(reward_json.contains("linear_acceleration")){
                    parameters.linear_acceleration = reward_json["linear_acceleration"];
                }
                if(reward_json.contains("angular_acceleration")){
                    parameters.angular_acceleration = reward_json["angular_acceleration"];
                }
                if(reward_json.contains("action")){
                    parameters.action = reward_json["action"];
                }
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::MDP::ObservationNoise& parameters, nlohmann::json config){
        if(config.contains("mdp")) {
            auto mdp = config["mdp"];
            if (mdp.contains("observation_noise")) {
                auto observation_noise = mdp["observation_noise"];
                if (observation_noise.contains("position")) {
                    parameters.position = observation_noise["position"];
                }
                if (observation_noise.contains("orientation")) {
                    parameters.orientation = observation_noise["orientation"];
                }
                if (observation_noise.contains("linear_velocity")) {
                    parameters.linear_velocity = observation_noise["linear_velocity"];
                }
                if (observation_noise.contains("angular_velocity")) {
                    parameters.angular_velocity = observation_noise["angular_velocity"];
                }
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::MDP::Initialization& parameters, nlohmann::json config){
        if(config.contains("mdp")) {
            auto mdp_json = config["mdp"];
            if(mdp_json.contains("init")) {
                auto init_json = mdp_json["init"];
                if(init_json.contains("guidance")){
                    parameters.guidance = init_json["guidance"];
                }
                if(init_json.contains("max_position")){
                    parameters.max_position = init_json["max_position"];
                }
                if(init_json.contains("max_orientation")){
                    parameters.max_angle = init_json["max_angle"];
                }
                if(init_json.contains("max_linear_velocity")){
                    parameters.max_linear_velocity = init_json["max_linear_velocity"];
                }
                if(init_json.contains("max_angular_velocity")){
                    parameters.max_angular_velocity = init_json["max_angular_velocity"];
                }
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::MDP::Termination& parameters, nlohmann::json config){
        if(config.contains("mdp")) {
            auto mdp_json = config["mdp"];
            if(mdp_json.contains("termination")) {
                auto termination_json = mdp_json["termination"];
                if(termination_json.contains("enabled")){
                    parameters.enabled = termination_json["enabled"];
                }
                if(termination_json.contains("position_threshold")){
                    parameters.position_threshold = termination_json["position_threshold"];
                }
                if(termination_json.contains("linear_velocity_threshold")){
                    parameters.linear_velocity_threshold = termination_json["linear_velocity_threshold"];
                }
                if(termination_json.contains("angular_velocity_threshold")){
                    parameters.angular_velocity_threshold = termination_json["angular_velocity_threshold"];
                }
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::MDP& parameters, nlohmann::json config){
        load_config<DEV_SPEC, SPEC>(device, parameters.reward, config);
        load_config<DEV_SPEC, SPEC>(device, parameters.init, config);
        load_config<DEV_SPEC, SPEC>(device, parameters.termination, config);
        load_config<DEV_SPEC, SPEC>(device, parameters.observation_noise, config);
    }
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::DomainRandomization& parameters, nlohmann::json config){
        if(config.contains("domain_randomization")){
            auto dr = config["domain_randomization"];
            if(dr.contains("rotor_thrust_coefficients")){
                parameters.rotor_thrust_coefficients = dr["rotor_thrust_coefficients"];
            }
            if(dr.contains("rotor_torque_constant")){
                parameters.rotor_torque_constant = dr["rotor_torque_constant"];
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>::Integration& parameters, nlohmann::json config){
        if(config.contains("integration")){
            auto integration = config["integration"];
            if(integration.contains("dt")){
                parameters.dt = integration["dt"];
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersBase<SPEC>& parameters, nlohmann::json config){
        load_config<DEV_SPEC, SPEC>(device, parameters.dynamics, config);
        load_config<DEV_SPEC, SPEC>(device, parameters.mdp, config);
        load_config<DEV_SPEC, SPEC>(device, parameters.domain_randomization, config);
        load_config<DEV_SPEC, SPEC>(device, parameters.integration, config);
    }
    template <typename DEV_SPEC, typename T, typename TI, typename NEXT_COMPONENT>
    void load_config(devices::CPU<DEV_SPEC>& device, typename rl_tools::rl::environments::multirotor::ParametersDisturbances<T, TI, NEXT_COMPONENT>& parameters, nlohmann::json config){
        load_config(device, (NEXT_COMPONENT&)parameters, config);
        if(config.contains("disturbances")) {
            auto disturbances = config["disturbances"];
            if(disturbances.contains("random_force")) {
                auto random_force = disturbances["random_force"];
                parameters.disturbances.random_force.mean = random_force["mean"];
                parameters.disturbances.random_force.std = random_force["std"];
            }
            if(disturbances.contains("random_torque")) {
                auto random_torque = disturbances["random_torque"];
                parameters.disturbances.random_torque.mean = random_torque["mean"];
                parameters.disturbances.random_torque.std = random_torque["std"];
            }
        }
    }
#endif
}

#endif