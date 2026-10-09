#include "verbosity.h"
#include <rl_tools/operations/cpu.h>
#include <gtest/gtest.h>

#include <condition_variable>
#include <string>
#include <thread>

namespace rlt = rl_tools;
using DEVICE = rlt::devices::CPU<hyperdrone::CPUSpecification>;
using Verbosity = rlt::devices::logging::Verbosity;
using rlt::log;

namespace {
    struct ChangeDuringFormatting {};

    std::ostream& operator<<(std::ostream& out, const ChangeDuringFormatting&){
        hyperdrone::set_verbosity((int)Verbosity::error);
        return out << "already enabled";
    }

    struct CountFormatting {
        int& count;
    };

    std::ostream& operator<<(std::ostream& out, const CountFormatting& value){
        value.count++;
        return out;
    }
}

TEST(RL_TOOLS_LOGGING_HYPERDRONE, SHARED_POLICY_AND_FORMATTING_SNAPSHOT){
    DEVICE existing;
    hyperdrone::set_verbosity((int)Verbosity::warning);
    DEVICE future;
    EXPECT_THROW(hyperdrone::set_verbosity(-1), std::invalid_argument);
    EXPECT_THROW(hyperdrone::set_verbosity(4), std::invalid_argument);

    testing::internal::CaptureStderr();
    log(existing, existing.logger, Verbosity::info, "suppressed");
    log(future, future.logger, Verbosity::info, "suppressed");
    log(existing, existing.logger, Verbosity::warning, ChangeDuringFormatting{});
    log(future, future.logger, Verbosity::warning, "suppressed");
    EXPECT_EQ(testing::internal::GetCapturedStderr(), "already enabled\n");
    hyperdrone::set_verbosity((int)Verbosity::info);
}

TEST(RL_TOOLS_LOGGING_HYPERDRONE, CONCURRENT_POLICY_UPDATES){
    hyperdrone::set_verbosity((int)Verbosity::warning);
    std::mutex mutex;
    std::condition_variable ready;
    int arrived = 0;
    const auto start = [&](){
        std::unique_lock<std::mutex> lock(mutex);
        arrived++;
        ready.notify_all();
        ready.wait(lock, [&](){ return arrived == 3; });
    };
    constexpr int ITERATIONS = 20000;
    int formatted_debug[2] = {};
    int formatted_errors[2] = {};
    std::thread writer([&](){
        start();
        for(int i = 0; i < ITERATIONS; i++){
            hyperdrone::set_verbosity((int)(i % 2 == 0 ? Verbosity::error : Verbosity::warning));
        }
    });
    const auto read = [&](int reader){
        DEVICE device;
        start();
        for(int i = 0; i < ITERATIONS; i++){
            log(device, device.logger, Verbosity::error, CountFormatting{formatted_errors[reader]});
            log(device, device.logger, Verbosity::debug, CountFormatting{formatted_debug[reader]});
        }
    };
    testing::internal::CaptureStderr();
    std::thread first(read, 0);
    std::thread second(read, 1);
    writer.join();
    first.join();
    second.join();
    const std::string output = testing::internal::GetCapturedStderr();
    EXPECT_EQ(output, std::string(2 * ITERATIONS, '\n'));
    for(int reader = 0; reader < 2; reader++){
        EXPECT_EQ(formatted_errors[reader], ITERATIONS);
        EXPECT_EQ(formatted_debug[reader], 0);
    }
    hyperdrone::set_verbosity((int)Verbosity::info);
}
