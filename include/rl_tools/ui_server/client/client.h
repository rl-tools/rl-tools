#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UI_SERVER_CLIENT_CLIENT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UI_SERVER_CLIENT_CLIENT_H
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::ui_server::client{
    namespace beast = boost::beast;
    namespace http = beast::http;
    namespace websocket = beast::websocket;
    namespace net = boost::asio;
    using tcp = boost::asio::ip::tcp;

    template<typename T_ENVIRONMENT>
    struct UI{
        using ENVIRONMENT = T_ENVIRONMENT;
        std::string host = "localhost";
        std::string port = "8000";
        net::io_context ioc;
        websocket::stream <tcp::socket> ws{ioc};
        std::string ns = "";
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif