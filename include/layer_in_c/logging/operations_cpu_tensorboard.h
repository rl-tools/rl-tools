#ifndef LAYER_IN_C_LOGGING_OPERATIONS_CPU_TENSORBOARD_H
#define LAYER_IN_C_LOGGING_OPERATIONS_CPU_TENSORBOARD_H

#include <filesystem>
#include <cassert>
#include "operations_cpu.h"
namespace layer_in_c{
    template <typename DEVICE>
    void construct(DEVICE& device, devices::logging::CPU_TENSORBOARD* logger){
        assert(logger != nullptr);// "Cannot construct TensorBoard logger on null device");
        utils::assert_exit(device, device.logger == logger, "Device logger and passed logger are not the same");
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
        logger->tb = new TensorBoardLogger(log_file);
    }
    template <typename DEVICE>
    void destruct(DEVICE& device, devices::logging::CPU_TENSORBOARD* logger){
        assert(logger != nullptr);// "Cannot destruct TensorBoard logger on null device");
        delete logger->tb;
    }
    template <typename DEVICE>
    void add_scalar(DEVICE& device, devices::logging::CPU_TENSORBOARD* logger, const char* key, const float value, const typename devices::logging::CPU_TENSORBOARD::index_t cadence = 1){
        if(logger == nullptr){
            return;
        }
        std::lock_guard<std::mutex> lock(logger->mutex);
        if(logger->step % cadence == 0){
            logger->tb->add_scalar(key, logger->step, value);
        }
    }
}
#endif
