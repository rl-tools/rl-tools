#include <rl_tools/ui_server/server.h>
#include <rl_tools/operations/cpu.h>
#include <thread>
#include <chrono>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

//int main(int argc, char* argv[]) {
//    using namespace rlt::ui_server;
//    std::cout << "Note: This executable should be executed in the context (working directory) of the main repo e.g. ./build/src/rl_environments_multirotor_ui 0.0.0.0 8000" << std::endl;
//    State state;
//    std::string static_path = "include/rl_tools/ui_server/static/multirotor";
//
//    if(argc != 3){
//        std::cerr << "Usage: " << argv[0] << " <address> <port> (e.g. \'0.0.0.0 8000\' for localhost 8000)\n";
//        return EXIT_FAILURE;
//    }
//
//    auto const address = net::ip::make_address(argv[1]);
//    unsigned short port = static_cast<unsigned short>(std::atoi(argv[2]));
//
//    net::io_context ioc{1};
//
//    tcp::acceptor acceptor{ioc, {address, port}};
//    tcp::socket socket{ioc};
//    http_server(acceptor, socket, state, static_path);
//
//    std::cout << "Web interface coming up at: http://" << address << ":" << port << std::endl;
//
//    boost::asio::signal_set signals(ioc, SIGINT);
//
//    signals.async_wait(
//            [&](const boost::system::error_code& error, int signal_number) {
//                ioc.stop();
//            }
//    );
//
//    ioc.run();
//}

int main(int argc, char* argv[]) {
    using namespace rlt::ui_server;
    std::cout << "Note: This executable should be executed in the context (working directory) of the main repo e.g. ./build/src/ui_server/ui_server 0.0.0.0 8000" << std::endl;
    std::string static_path = "include/rl_tools/ui_server/static/multirotor";

    using DEVICE = rlt::devices::DefaultCPU;
    DEVICE device;

    if(argc != 3){
        std::cerr << "Usage: " << argv[0] << " <address> <port> (e.g. \'0.0.0.0 8000\' for localhost 8000)\n";
        return EXIT_FAILURE;
    }

    std::string address = argv[1];
    unsigned short port = static_cast<unsigned short>(std::atoi(argv[2]));

    rlt::ui_server::UIServer server;

    rlt::init(device, server, address, port, static_path);

    while(true){
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

}
