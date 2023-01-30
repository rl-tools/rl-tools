#ifndef LAYER_IN_C_LOGGING_OPERATIONS_CPU_TENSORBOARD_H
#define LAYER_IN_C_LOGGING_OPERATIONS_CPU_TENSORBOARD_H

#include <filesystem>
namespace layer_in_c{
    void construct(devices::logging::CPU_TENSORBOARD& dev){
        time_t now;
        time(&now);
        char buf[sizeof "2011-10-08T07:07:09Z"];
        strftime(buf, sizeof buf, "%FT%TZ", gmtime(&now));

        std::string logs_dir = "logs";
        if (!std::filesystem::is_directory(logs_dir.c_str()) || !std::filesystem::exists(logs_dir.c_str())) {
            std::filesystem::create_directory(logs_dir.c_str());
        }
        std::string log_dir = logs_dir + "/" + std::string(buf);
        if (!std::filesystem::is_directory(log_dir.c_str()) || !std::filesystem::exists(log_dir.c_str())) {
            std::filesystem::create_directory(log_dir.c_str());
        }

        std::string log_file = log_dir + "/" + std::string("data.tfevents");
        std::cout << "Logging to " << log_file << std::endl;
        dev.tb = new TensorBoardLogger(log_file);
    }
    void destruct(devices::logging::CPU_TENSORBOARD& dev){
        delete dev.tb;
    }
    void add_scalar(devices::logging::CPU_TENSORBOARD& dev, const char* key, const float value, const typename devices::logging::CPU_TENSORBOARD::index_t cadence = 1){
        std::lock_guard<std::mutex> lock(dev.mutex);
        if(dev.step % cadence == 0){
            dev.tb->add_scalar(key, dev.step, value);
        }
    }
}
#endif
