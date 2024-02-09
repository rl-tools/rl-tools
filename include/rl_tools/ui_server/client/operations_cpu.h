#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UI_SERVER_CLIENT_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UI_SERVER_CLIENT_OPERATIONS_CPU_H

#include "client.h"

#include <nlohmann/json.hpp>
#include <thread>
#include <chrono>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename ENVIRONMENT>
    void init(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UIBuffered<ENVIRONMENT>& ui, std::string name_space){
        nlohmann::json parameters_json = json(dev, env.parameters);
        nlohmann::json parameters_message;
        parameters_message["channel"] = "setParameters";
        parameters_message["data"] = parameters_json;
        ui.buffer.push(parameters_message.dump());
        ui.ns = name_space;
    }
    template <typename DEVICE, typename ENVIRONMENT>
    void init(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UIBuffered<ENVIRONMENT>& ui){
        init(dev, env, ui, "default");
    }
    template <typename DEVICE, typename ENVIRONMENT>
    void set_state(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UIBuffered<ENVIRONMENT>& ui, const typename ENVIRONMENT::State& state){
        nlohmann::json message;
        message["namespace"] = ui.ns;
        message["channel"] = "setState";
        message["data"] = nlohmann::json();
        message["data"]["state"] = json(dev, state);
        ui.buffer.push(message.dump());
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ACTION_SPEC>
    void set_state(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UIBuffered<ENVIRONMENT>& ui, const typename ENVIRONMENT::State& state, const Matrix<ACTION_SPEC>& action){
        static_assert(ACTION_SPEC::COLS == ENVIRONMENT::ACTION_DIM);
        static_assert(ACTION_SPEC::ROWS == 1);
        using T = typename ACTION_SPEC::T;
        nlohmann::json message;
        message["namespace"] = ui.ns;
        message["channel"] = "setState";
        message["data"] = nlohmann::json();
        message["data"]["state"] = json(dev, state);
        std::vector<T> action_vector;
        for(int i = 0; i < ENVIRONMENT::ACTION_DIM; i++){
            action_vector.push_back(get(action, 0, i));
        }
        message["data"]["action"] = action_vector;
        ui.buffer.push(message.dump());
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ACTION_SPEC>
    void set_action(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UIBuffered<ENVIRONMENT>& ui, const Matrix<ACTION_SPEC>& action){
        static_assert(ACTION_SPEC::COLS == ENVIRONMENT::ACTION_DIM);
        static_assert(ACTION_SPEC::ROWS == 1);
        using T = typename ACTION_SPEC::T;
        nlohmann::json message;
        message["namespace"] = ui.ns;
        message["channel"] = "setAction";
        message["data"] = nlohmann::json();
        std::vector<T> action_vector;
        for(int i = 0; i < ENVIRONMENT::ACTION_DIM; i++){
            action_vector.push_back(get(action, 0, i));
        }
        message["data"]["action"] = action_vector;
        ui.buffer.push(message.dump());
    }
    template <typename DEVICE, typename ENVIRONMENT>
    void render(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UIBuffered<ENVIRONMENT>& ui){
//        std::this_thread::sleep_for(std::chrono::duration<decltype(env.parameters.dt)>((env.parameters.dt)));
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif