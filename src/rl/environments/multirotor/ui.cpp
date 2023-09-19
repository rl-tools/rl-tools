#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <boost/beast/websocket.hpp>
#include <filesystem>
#include <fstream>
#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>
#include <backprop_tools/rl/environments/multirotor/ui.h>
namespace bpt = backprop_tools;

#include "td3/parameters.h"

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

namespace my_program_state
{
    std::size_t
    request_count()
    {
        static std::size_t count = 0;
        return ++count;
    }

    std::time_t
    now()
    {
        return std::time(0);
    }
}
class websocket_session : public std::enable_shared_from_this<websocket_session> {
    beast::websocket::stream<tcp::socket> ws_;
public:
    explicit websocket_session(tcp::socket socket) : ws_(std::move(socket)) {}

    template<class Body>
    void run(http::request<Body>&& req) {
        ws_.async_accept(
                req,
                beast::bind_front_handler(
                        &websocket_session::on_accept,
                        shared_from_this()
                )
        );
    }

    void on_accept(beast::error_code ec) {
        if(ec) return;
        // Start reading messages
        do_read();
        bpt::devices::DefaultCPU device;
        auto rng = bpt::random::default_engine(typename bpt::devices::DefaultCPU::SPEC::RANDOM{});
        using T = double;
        using TI = typename bpt::devices::DefaultCPU::index_t;
        using ENVIRONMENT = typename parameters_sim2real::environment<T, TI>::ENVIRONMENT;
        ENVIRONMENT env;
        env.parameters = parameters_0::environment<T, TI>::parameters;
        ENVIRONMENT::State state;
        bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
        bpt::malloc(device, action);
        bpt::set_all(device, action, 0);
        bpt::sample_initial_state(device, env, state, rng);
        using UI = bpt::rl::environments::multirotor::UI<ENVIRONMENT>;
        UI ui;
        ws_.write(
                net::buffer(bpt::rl::environments::multirotor::model_message(device, env, ui).dump())
        );
        ws_.write(
                net::buffer(bpt::rl::environments::multirotor::state_message(device, ui, state, action).dump())
        );
        bpt::free(device, action);
    }

    void do_read() {
        ws_.async_read(
                buffer_,
                beast::bind_front_handler(
                        &websocket_session::on_read,
                        shared_from_this()
                )
        );
    }

    void on_read(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        if(ec) return;

        // read message to string:
        bool send_message = false;
        if(send_message){
            std::string message = beast::buffers_to_string(buffer_.data());
            std::cout << message << std::endl;
            message = "Hello from server responding " + message;
            ws_.text(ws_.got_text());
            ws_.async_write(
                    net::buffer(message),
                    beast::bind_front_handler(
                            &websocket_session::on_write,
                            shared_from_this()
                    )
            );
        }
        else{
            do_read();
        }
    }

    void on_write(beast::error_code ec, std::size_t bytes_transferred) {
        boost::ignore_unused(bytes_transferred);
        if(ec) return;

        buffer_.consume(buffer_.size());

        do_read();
    }

private:
    beast::flat_buffer buffer_;
};


class http_connection: public std::enable_shared_from_this<http_connection>
{
public:
    http_connection(tcp::socket socket): socket_(std::move(socket)){}
    void start(){
        read_request();
        check_deadline();
    }

private:
    tcp::socket socket_;
    beast::flat_buffer buffer_{8192};
    http::request<http::dynamic_body> request_;
    http::response<http::dynamic_body> response_;
    net::steady_timer deadline_{socket_.get_executor(), std::chrono::seconds(60)};
    void read_request(){
        auto self = shared_from_this();

        http::async_read(
                socket_,
                buffer_,
                request_,
                [self](beast::error_code ec,
                       std::size_t bytes_transferred)
                {
                    boost::ignore_unused(bytes_transferred);
                    if(!ec)
                        self->process_request();
                });
    }

    void process_request(){
        response_.version(request_.version());
        response_.keep_alive(false);

        switch(request_.method())
        {
            case http::verb::get:
                response_.result(http::status::ok);
                response_.set(http::field::server, "Beast");
                create_response();
                break;

            default:
                // We return responses indicating an error if
                // we do not recognize the request method.
                response_.result(http::status::bad_request);
                response_.set(http::field::content_type, "text/plain");
                beast::ostream(response_.body())
                        << "Invalid request-method '"
                        << std::string(request_.method_string())
                        << "'";
                break;
        }
        write_response();
    }

    void create_response(){
        if(request_.target() == "/count"){
            response_.set(http::field::content_type, "text/html");
            beast::ostream(response_.body())
                    << "<html>\n"
                    <<  "<head><title>Request count</title></head>\n"
                    <<  "<body>\n"
                    <<  "<h1>Request count</h1>\n"
                    <<  "<p>There have been "
                    <<  my_program_state::request_count()
                    <<  " requests so far.</p>\n"
                    <<  "</body>\n"
                    <<  "</html>\n";
        }
        else if(request_.target() == "/time"){
            response_.set(http::field::content_type, "text/html");
            beast::ostream(response_.body())
                    <<  "<html>\n"
                    <<  "<head><title>Current time</title></head>\n"
                    <<  "<body>\n"
                    <<  "<h1>Current time</h1>\n"
                    <<  "<p>The current time is "
                    <<  my_program_state::now()
                    <<  " seconds since the epoch.</p>\n"
                    <<  "</body>\n"
                    <<  "</html>\n";
        }
        else if(request_.target() == "/ws"){
            maybe_upgrade();
        }
        else{
            std::filesystem::path path(std::string(request_.target()));
            if(path.empty() || path == "/"){
                path = "/index.html";
            }
            path = "src/rl/environments/multirotor/static" + path.string();
            // check if file at path exists

            if(std::filesystem::exists(path)){
                response_.result(http::status::ok);
                // check extension and use correct content_type
                if(path.extension() == ".html")
                    response_.set(http::field::content_type, "text/html");
                else if(path.extension() == ".js")
                    response_.set(http::field::content_type, "application/javascript");
                else if(path.extension() == ".css")
                    response_.set(http::field::content_type, "text/css");
                else if(path.extension() == ".png")
                    response_.set(http::field::content_type, "image/png");
                else if(path.extension() == ".jpg")
                    response_.set(http::field::content_type, "image/jpeg");
                else if(path.extension() == ".gif")
                    response_.set(http::field::content_type, "image/gif");
                else if(path.extension() == ".ico")
                    response_.set(http::field::content_type, "image/x-icon");
                else if(path.extension() == ".txt")
                    response_.set(http::field::content_type, "text/plain");
                else
                    response_.set(http::field::content_type, "application/octet-stream");
                beast::ostream(response_.body()) << std::ifstream(path).rdbuf();
            }
            else{
                response_.result(http::status::not_found);
                response_.set(http::field::content_type, "text/plain");
                beast::ostream(response_.body()) << "File not found\r\n";
            }

//            response_.result(http::status::not_found);
//            response_.set(http::field::content_type, "text/plain");
//            beast::ostream(response_.body()) << "File not found\r\n";
        }
    }
    void maybe_upgrade() {
        if (beast::websocket::is_upgrade(request_)) {
            // Construct the WebSocket session and run it
            std::make_shared<websocket_session>(std::move(socket_))->run(std::move(request_));
            return;
        }
    }


    void write_response(){
        auto self = shared_from_this();

        response_.content_length(response_.body().size());

        http::async_write(
                socket_,
                response_,
                [self](beast::error_code ec, std::size_t)
                {
                    self->socket_.shutdown(tcp::socket::shutdown_send, ec);
                    self->deadline_.cancel();
                });
    }

    void check_deadline(){
        auto self = shared_from_this();

        deadline_.async_wait(
                [self](beast::error_code ec){
                    if(!ec){
                        self->socket_.close(ec);
                    }
                });
    }
};

void http_server(tcp::acceptor& acceptor, tcp::socket& socket){
    acceptor.async_accept(socket,
                          [&](beast::error_code ec)
                          {
                              if(!ec)
                                  std::make_shared<http_connection>(std::move(socket))->start();
                              http_server(acceptor, socket);
                          });
}

int main(int argc, char* argv[]) {
    try{
        // Check command line arguments.
        if(argc != 3)
        {
            std::cerr << "Usage: " << argv[0] << " <address> <port>\n";
            std::cerr << "  For IPv4, try:\n";
            std::cerr << "    receiver 0.0.0.0 80\n";
            std::cerr << "  For IPv6, try:\n";
            std::cerr << "    receiver 0::0 80\n";
            return EXIT_FAILURE;
        }

        auto const address = net::ip::make_address(argv[1]);
        unsigned short port = static_cast<unsigned short>(std::atoi(argv[2]));

        net::io_context ioc{1};

        tcp::acceptor acceptor{ioc, {address, port}};
        tcp::socket socket{ioc};
        http_server(acceptor, socket);

        ioc.run();
    }
    catch(std::exception const& e){
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}