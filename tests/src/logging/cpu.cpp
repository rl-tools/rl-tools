#include <rl_tools/operations/cpu.h>
#include <gtest/gtest.h>

namespace rlt = rl_tools;
using Verbosity = rlt::devices::logging::Verbosity;
using rlt::log;

namespace {
    struct CountFormatting {
        int& count;
    };

    std::ostream& operator<<(std::ostream& out, const CountFormatting& value){
        value.count++;
        return out << "formatted";
    }
}

TEST(RL_TOOLS_LOGGING_CPU, DIAGNOSTIC_LEVELS){
    rlt::devices::DefaultCPU device;
    EXPECT_EQ(device.logger.verbosity, Verbosity::info);
    const Verbosity levels[] = {Verbosity::error, Verbosity::warning, Verbosity::info, Verbosity::debug};
    const char* expected[] = {"0\n", "0\n1\n", "0\n1\n2\n", "0\n1\n2\n3\n"};
    for(int threshold = 0; threshold < 4; threshold++){
        device.logger.verbosity = levels[threshold];
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        for(int level = 0; level < 4; level++){
            log(device, device.logger, levels[level], level);
        }
        EXPECT_EQ(testing::internal::GetCapturedStderr(), expected[threshold]);
        EXPECT_EQ(testing::internal::GetCapturedStdout(), "");
    }
}

TEST(RL_TOOLS_LOGGING_CPU, REPORTING_AND_FORMATTING){
    rlt::devices::DefaultCPU device;
    device.logger.verbosity = Verbosity::error;
    int formatted = 0;
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    log(device, device.logger, Verbosity::info, CountFormatting{formatted});
    log(device, device.logger, Verbosity::error, std::hex, 16);
    log(device, device.logger, Verbosity::error, 16);
    log(device, device.logger, "report ", 16);
    EXPECT_EQ(testing::internal::GetCapturedStderr(), "10\n16\n");
    EXPECT_EQ(testing::internal::GetCapturedStdout(), "report 16\n");
    EXPECT_EQ(formatted, 0);
}
