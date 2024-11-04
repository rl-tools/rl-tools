#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_EXTRACK_UTILS_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_EXTRACK_UTILS_OPERATIONS_CPU_H

#include <chrono>
#include <iomanip>
#include <ctime>
#include <sstream>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::loop::steps::extrack::utils{
        template <typename DUMMY = bool>
        std::string get_timestamp_string(){
            // equivalent to date '+%Y-%m-%d_%H-%M-%S'
            auto now = std::chrono::system_clock::now();
            std::time_t now_c = std::chrono::system_clock::to_time_t(now);
            std::tm now_local;

#if defined(_WIN32) || defined(_WIN64)
            localtime_s(&now_local, &now_c);
#else
            localtime_r(&now_c, &now_local);
#endif
            std::stringstream ss;
            ss << std::put_time(&now_local, "%Y-%m-%d_%H-%M-%S");

            return ss.str();
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif