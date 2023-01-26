#include "quaternion_helper.h"

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

namespace layer_in_c::rl::environments::multirotor {
    namespace beast = boost::beast;         // from <boost/beast.hpp>
    namespace http = beast::http;           // from <boost/beast/http.hpp>
    namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
    namespace net = boost::asio;            // from <boost/asio.hpp>
    using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

    template <typename T_ENVIRONMENT>
    struct UI{
        using ENVIRONMENT = T_ENVIRONMENT;
        std::string host;
        std::string port;
        net::io_context ioc;
        websocket::stream<tcp::socket> ws{ioc};
    };
    const nlohmann::json message_skeleton = {
            {"channel", "setDroneState"},
            {"data", {
                                {"id", "default"},
                                {"data", {
                                                 {"pose", {
                                                                  {"position", {0, 0, 0}},
                                                                  {"orientation", {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}
                                                          }
                                                 },
                                                 {"rotor_states", {
                                                                          {"rpm", 0},
                                                                          {"rpm", 0},
                                                                          {"rpm", 0},
                                                                          {"rpm", 0}
                                                                  }
                                                 }
                                         }
                                }
                        }
            }
    };
}


namespace layer_in_c{
    template <typename DEVICE, typename ENVIRONMENT>
    void init(DEVICE& dev, rl::environments::multirotor::UI<ENVIRONMENT>& ui){
        namespace beast = boost::beast;         // from <boost/beast.hpp>
        namespace http = beast::http;           // from <boost/beast/http.hpp>
        namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
        namespace net = boost::asio;            // from <boost/asio.hpp>
        using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>
        try
        {
            tcp::resolver resolver{ui.ioc};
            auto const results = resolver.resolve(ui.host, ui.port);

            net::connect(ui.ws.next_layer(), results.begin(), results.end());
            ui.ws.handshake(ui.host, "/");

        }
        catch(std::exception const& e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    template <typename DEVICE, typename ENVIRONMENT>
    void set_state(DEVICE& dev, rl::environments::multirotor::UI<ENVIRONMENT>& ui, const typename ENVIRONMENT::State& state){
        using namespace rl::environments::multirotor;

        if (ui.ws.is_open()) {
            auto message = message_skeleton;
            typename ENVIRONMENT::T orientation[3][3];
            quaternion_to_rotation_matrix<DEVICE, typename ENVIRONMENT::T>(&state.state[3], orientation);

            message["data"]["data"]["pose"]["position"] = {state.state[0], state.state[1], state.state[2]};
            message["data"]["data"]["pose"]["orientation"] = {
                    {orientation[0][0], orientation[0][1], orientation[0][2]},
                    {orientation[1][0], orientation[1][1], orientation[1][2]},
                    {orientation[2][0], orientation[2][1], orientation[2][2]}
            };
            ui.ws.write(net::buffer(message.dump()));
        }
        else{
            std::cerr << "Error: websocket is not open" << std::endl;
        }
    }

}
