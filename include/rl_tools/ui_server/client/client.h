#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UI_SERVER_CLIENT_CLIENT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UI_SERVER_CLIENT_CLIENT_H
#include <queue>
#ifdef RL_TOOLS_ENABLE_LIBWEBSOCKETS
#include <libwebsockets.h>
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::ui_server::client{
    template<typename T_ENVIRONMENT>
    struct UIJSON{
        std::string ns = "";
    };

    template<typename T_ENVIRONMENT>
    struct UIBuffered: UIJSON<T_ENVIRONMENT>{
        std::queue<std::string> buffer;
    };
#ifdef RL_TOOLS_ENABLE_LIBWEBSOCKETS
    template<typename T_ENVIRONMENT>
    struct UIWebSocket: UIJSON<T_ENVIRONMENT>{
        bool connected;
        bool interrupted;
        bool message_sent;
        struct lws_context_creation_info ctx_info;
        struct lws_client_connect_info conn_info;
        struct lws_context *context;
        struct lws *wsi;
        struct lws_protocols protocols[];
    };
#endif



}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif