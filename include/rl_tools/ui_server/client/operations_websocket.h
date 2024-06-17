#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UI_SERVER_CLIENT_OPERATIONS_WEBSOCKET_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UI_SERVER_CLIENT_OPERATIONS_WEBSOCKET_H

#include "client.h"

#include <thread>
#include <chrono>

#include "operations_cpu.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace ui_server::client::websocket{
        template <typename ENVIRONMENT>
        static int callback(struct lws *wsi, enum lws_callback_reasons reason, void *user, void *in, size_t len) {
            using UI = ui_server::client::UIWebSocket<ENVIRONMENT>;
            UI *ui = (UI*) user;
            switch (reason) {
                case LWS_CALLBACK_CLIENT_ESTABLISHED:
                    lwsl_user("Client connected\n");
                    ui->connected = 1;
                    break;
                case LWS_CALLBACK_CLIENT_WRITEABLE:
                    ui->message_sent = true;
                    break;
                case LWS_CALLBACK_CLIENT_RECEIVE:
                    lwsl_user("Received: %s\n", (char *)in);
                    break;
                case LWS_CALLBACK_CLIENT_CLOSED:
                    lwsl_user("Client closed\n");
                    break;
                default:
                    std::cout << "Unhandled callback: " << reason << std::endl;
                    break;
            }
            return 0;
        }
        template <typename ENVIRONMENT>
        const struct lws_protocols protocols[] = {{"ui_server", callback<ENVIRONMENT>, sizeof(UI*), 0, 0, nullptr, 0}, LWS_PROTOCOL_LIST_TERM};

        template <typename DEVICE, typename ENVIRONMENT>
        void send_message(DEVICE& device, ui_server::client::UIWebSocket<ENVIRONMENT>& ui, std::string message){
            utils::assert_exit(device, ui.connected, "Cannot send message: UI not connected");
            ui.message_sent = false;
            size_t buf_size = LWS_PRE + message.size() + 1; // +1 for null terminator
            std::vector<unsigned char> buf(buf_size); // Use a vector for dynamic allocation
            std::memcpy(buf.data() + LWS_PRE, message.c_str(), message.size() + 1);
            lws_write(ui.wsi, buf.data() + LWS_PRE, message.size(), LWS_WRITE_TEXT);
            lws_callback_on_writable(ui.wsi);
            while(!ui.message_sent){
                lws_service(ui.context, 250);
            }
            log(device, device.logger, std::string("Message \"") + message + "\" sent.");
        }
    }

    template <typename DEVICE, typename ENVIRONMENT>
    void init(DEVICE& device, ENVIRONMENT& env, ui_server::client::UIWebSocket<ENVIRONMENT>& ui, std::string name_space = "default"){
        using UI = ui_server::client::UIWebSocket<ENVIRONMENT>;
        ui.ns = name_space;
        ui.connected = false;
        ui.interrupted = false;
        memset(&ui.ctx_info, 0, sizeof(ui.ctx_info));
        memset(&ui.conn_info, 0, sizeof(ui.conn_info));

        ui.ctx_info.port = CONTEXT_PORT_NO_LISTEN;
        ui.ctx_info.protocols = ui_server::client::websocket::protocols<ENVIRONMENT>;
        ui.ctx_info.options = 0;
        ui.context = lws_create_context(&ui.ctx_info);
        utils::assert_exit(device, ui.context, "lws_create_context failed");

        ui.conn_info.context = ui.context;
        ui.conn_info.address = "localhost";
        ui.conn_info.port = 8000;
        ui.conn_info.path = "/backend";
        ui.conn_info.host = lws_canonical_hostname(ui.context);
        ui.conn_info.origin = "origin";
        ui.conn_info.protocol = "ui_server";
        ui.conn_info.ssl_connection = 0;
        ui.conn_info.userdata = &ui;

        lws_set_log_level(LLL_ERR | LLL_WARN | LLL_NOTICE | LLL_INFO | LLL_DEBUG | LLL_PARSER | LLL_HEADER | LLL_EXT | LLL_CLIENT | LLL_LATENCY, NULL);

        ui.wsi = lws_client_connect_via_info(&ui.conn_info);
        utils::assert_exit(device, ui.wsi, "lws_client_connect_via_info failed");

        log(device, device.logger, "Waiting for UI connection...");
        while (!ui.connected) {
            lws_service(ui.context, 250);
        }
        log(device, device.logger, "UI connected.");

        ui_server::client::websocket::send_message(device, ui, parameters_message(device, env, ui, name_space));
    }
    template <typename DEVICE, typename ENVIRONMENT>
    void set_state(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UIWebSocket<ENVIRONMENT>& ui, const typename ENVIRONMENT::State& state){
        ui_server::client::websocket::send_message(dev, ui, set_state_message(dev, env, ui, state));
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ACTION_SPEC>
    void set_state(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UIWebSocket<ENVIRONMENT>& ui, const typename ENVIRONMENT::State& state, const Matrix<ACTION_SPEC>& action){
        ui_server::client::websocket::send_message(dev, ui, set_state_action_message(dev, env, ui, state, action));
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ACTION_SPEC>
    void set_action(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UIWebSocket<ENVIRONMENT>& ui, const Matrix<ACTION_SPEC>& action){
        ui_server::client::websocket::send_message(dev, ui, set_action_message(dev, env, ui, action));
    }
    template <typename DEVICE, typename ENVIRONMENT>
    void render(DEVICE& dev, ENVIRONMENT& env, ui_server::client::UIWebSocket<ENVIRONMENT>& ui){
//        std::this_thread::sleep_for(std::chrono::duration<decltype(env.parameters.dt)>((env.parameters.dt)));
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
