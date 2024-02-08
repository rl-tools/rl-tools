#define LOAD_TRACK_FROM_FILE
#include "training_td3.h"
#include "forwarder.h"

int main(int argc, char** argv) {
    TI seed = 0;
    if (argc > 1) {
        seed = std::atoi(argv[1]);
    }

    boost::asio::io_context io_context;
    boost::asio::io_context::strand strand(io_context);
    auto forwarder = std::make_shared<Forwarder>(io_context);
    forwarder->run("localhost", "8000");

    auto* state = create(seed);

    std::thread io_thread([&io_context]() {
        io_context.run();
    });

    io_thread.detach();

//    std::cout << "Track: " << message.dump(4) << std::endl;
    bool prev_mode_interactive = false;
    T sleep = 0;
    while(true){
        {
            std::string message = "";
            bool new_message = false;
            {
                std::lock_guard<std::mutex> lock(forwarder->mutex);
                if(!forwarder->receiving_message_queue.empty()){
                    message = forwarder->receiving_message_queue.front();
                    new_message = true;
                    forwarder->receiving_message_queue.pop();
                }
            }

            if(new_message){
                std::cout << "new message: " << message << std::endl;
                sleep = step(state, message.c_str());
            }
            else{
                sleep = step(state);
            }
            while(forwarder->handshook && !state->ts.ui.buffer.empty()){
//                std::lock_guard<std::mutex> lock(mutex);
                std::string message = state->ts.ui.buffer.front();
                state->ts.ui.buffer.pop();
                forwarder->send_message(message);
                if(state->mode_training){
                    std::this_thread::sleep_for(std::chrono::duration<T>(state->ts.env_eval.parameters.dt));
                }
            }
        }
        if(sleep > 0){
            std::this_thread::sleep_for(std::chrono::duration<T>(state->ts.env_eval.parameters.dt / 10));
        }
    }
    return 0;
}
