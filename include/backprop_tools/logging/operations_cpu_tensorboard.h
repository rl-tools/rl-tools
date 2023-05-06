#ifndef BACKPROP_TOOLS_LOGGING_OPERATIONS_CPU_TENSORBOARD_H
#define BACKPROP_TOOLS_LOGGING_OPERATIONS_CPU_TENSORBOARD_H

#include <filesystem>
#include <cassert>
#include "operations_cpu.h"
namespace backprop_tools{
    namespace logging::tensorboard{
        std::string sanitize_file_name(const std::string &input) {
            std::string output = input;

            const std::string invalid_chars = R"(<>:\"/\|?*)";

            std::replace_if(output.begin(), output.end(), [&invalid_chars](const char &c) {
                return invalid_chars.find(c) != std::string::npos;
            }, '_');

            return output;
        }
    }
    template <typename DEVICE>
    void construct(DEVICE& device, devices::logging::CPU_TENSORBOARD* logger, std::string logs_dir, std::string name){
        assert(logger != nullptr);// "Cannot construct TensorBoard logger on null device");
        utils::assert_exit(device, device.logger == logger, "Device logger and passed logger are not the same");
        if (!std::filesystem::is_directory(logs_dir.c_str()) || !std::filesystem::exists(logs_dir.c_str())) {
            std::filesystem::create_directory(logs_dir.c_str());
        }
        std::filesystem::path log_dir = std::filesystem::path(logs_dir) / name;
        if (!std::filesystem::is_directory(log_dir.c_str()) || !std::filesystem::exists(log_dir.c_str())) {
            std::filesystem::create_directory(log_dir.c_str());
        }

        auto log_file = log_dir / std::string("data.tfevents");
        std::cout << "Logging to " << log_file.string() << std::endl;
        TensorBoardLoggerOptions opts;
        opts.flush_period_s(1);
        logger->tb = new TensorBoardLogger(log_file.string(), opts);
    }
    template <typename DEVICE>
    void construct(DEVICE& device, devices::logging::CPU_TENSORBOARD* logger){
        time_t now;
        time(&now);
        char buf[sizeof "0000-00-00T00:00:00Z"];
        strftime(buf, sizeof buf, "%FT%TZ", localtime(&now));

        std::string run_name = std::string(buf);

        construct(device, logger, std::string("logs"), logging::tensorboard::sanitize_file_name(run_name));
    }
    template <typename DEVICE>
    void destruct(DEVICE& device, devices::logging::CPU_TENSORBOARD* logger){
        assert(logger != nullptr);// "Cannot destruct TensorBoard logger on null device");
        delete logger->tb;
    }
    template <typename DEVICE>
    void set_step(DEVICE& device, devices::logging::CPU_TENSORBOARD* logger, typename DEVICE::index_t step){
        device.logger->step = step;
    }
    template <typename DEVICE, typename KEY_TYPE, typename VALUE_TYPE, typename CADANCE_TYPE>
    void add_scalar(DEVICE& device, devices::logging::CPU_TENSORBOARD* logger, const KEY_TYPE key, const VALUE_TYPE value, const CADANCE_TYPE cadence){
        if(logger == nullptr){
            return;
        }
        std::lock_guard<std::mutex> lock(logger->mutex);
        if(logger->step % cadence == 0){
            logger->tb->add_scalar(key, logger->step, (float)value);
        }
    }
    template <typename DEVICE, typename KEY_TYPE, typename VALUE_TYPE>
    void add_scalar(DEVICE& device, devices::logging::CPU_TENSORBOARD* logger, const KEY_TYPE key, const VALUE_TYPE value){
        add_scalar(device, logger, key, value, (typename DEVICE::index_t)1);
    }
    template <typename DEVICE, typename KEY_TYPE, typename T, typename TI, typename CADANCE_TYPE>
    void add_histogram(DEVICE& device, devices::logging::CPU_TENSORBOARD* logger, const KEY_TYPE key, const T* values, const TI n_values, const CADANCE_TYPE cadence = (typename DEVICE::index_t)1){
        if(logger == nullptr){
            return;
        }
        std::lock_guard<std::mutex> lock(logger->mutex);
        if(logger->step % cadence == 0){
            logger->tb->add_histogram(key, logger->step, values, n_values);
        }
    }
    template <typename DEVICE, typename KEY_TYPE, typename T, typename TI>
    void add_histogram(DEVICE& device, devices::logging::CPU_TENSORBOARD* logger, const KEY_TYPE key, const T* values, const TI n_values){
        add_histogram(device, logger, key, values, n_values, (typename DEVICE::index_t)1);
    }
}
#endif
