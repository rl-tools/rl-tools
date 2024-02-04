#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UI_SERVER_CLIENT_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UI_SERVER_CLIENT_OPERATIONS_CPU_H

#include "client.h"

#include <nlohmann/json.hpp>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename ENVIRONMENT>
    void init(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UI<ENVIRONMENT>& ui){
        namespace beast = boost::beast;         // from <boost/beast.hpp>
        namespace http = beast::http;           // from <boost/beast/http.hpp>
        namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
        namespace net = boost::asio;            // from <boost/asio.hpp>
        using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>
        tcp::resolver resolver{ui.ioc};
        auto const results = resolver.resolve(ui.host, ui.port);

        net::connect(ui.ws.next_layer(), results.begin(), results.end());
        ui.ws.handshake(ui.host, "/backend");
        std::cout << "Waiting for handshake" << std::endl;
        boost::beast::flat_buffer buffer;
        ui.ws.read(buffer);
        std::cout << beast::make_printable(buffer.data()) << std::endl;
        auto message_string = beast::buffers_to_string(buffer.data());
        std::cout << "Handshake received: " << message_string << std::endl;
        buffer.consume(buffer.size());
        auto message = nlohmann::json::parse(message_string);
        ui.ns = message["data"]["namespace"];

        nlohmann::json parameters_json = json(dev, env.parameters);
        nlohmann::json parameters_message;
        parameters_message["namespace"] = ui.ns;
        parameters_message["channel"] = "setParameters";
        parameters_message["data"] = parameters_json;
        ui.ws.write(net::buffer(parameters_message.dump()));
    }
    template <typename DEVICE, typename ENVIRONMENT>
    void set_state(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UI<ENVIRONMENT>& ui, const typename ENVIRONMENT::State& state){
        nlohmann::json message;
        message["namespace"] = ui.ns;
        message["channel"] = "setState";
        message["data"] = nlohmann::json();
        message["data"]["state"] = json(dev, state);
        if (ui.ws.is_open()) {
            ui.ws.write(boost::beast::net::buffer(message.dump()));
        }
        else{
            std::cerr << "Error: websocket is not open" << std::endl;
        }
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ACTION_SPEC>
    void set_state(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UI<ENVIRONMENT>& ui, const typename ENVIRONMENT::State& state, const Matrix<ACTION_SPEC>& action){
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
        if (ui.ws.is_open()) {
            ui.ws.write(boost::beast::net::buffer(message.dump()));
        }
        else{
            std::cerr << "Error: websocket is not open" << std::endl;
        }
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ACTION_SPEC>
    void set_action(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UI<ENVIRONMENT>& ui, const Matrix<ACTION_SPEC>& action){
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
        if (ui.ws.is_open()) {
            ui.ws.write(boost::beast::net::buffer(message.dump()));
        }
        else{
            std::cerr << "Error: websocket is not open" << std::endl;
        }
    }
    template <typename DEVICE, typename ENVIRONMENT>
    void render(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UI<ENVIRONMENT>& ui){
        std::this_thread::sleep_for(std::chrono::duration<decltype(env.parameters.dt)>((env.parameters.dt)));
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif