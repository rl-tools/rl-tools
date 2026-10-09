#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_LOGGING_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_LOGGING_OPERATIONS_CPU_H



#include <iostream>
#include <sstream>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE>
    void init(DEVICE& dev, devices::logging::CPU& logger){ }
    template <typename DEVICE, typename PARAM>
    void init(DEVICE& dev, devices::logging::CPU& logger, PARAM& param){ }
    template <typename DEVICE, typename... ARGS>
    void log(DEVICE& device, devices::logging::CPU& logger, const ARGS&... args){
        (std::cout << ... << args) << std::endl;
    }
    template <typename DEVICE, typename... ARGS>
    void log(DEVICE& device, devices::logging::CPU& logger, devices::logging::Verbosity verbosity, const ARGS&... args){
        if(verbosity > logger.verbosity){
            return;
        }
        std::ostringstream message;
        (message << ... << args) << '\n';
        std::cerr << message.str() << std::flush;
    }

    template <typename DEVICE>
    void set_step(DEVICE& device, devices::logging::CPU& logger, typename DEVICE::index_t step){ /* noop */ }
    template <typename DEVICE>
    typename DEVICE::index_t get_step(DEVICE& device, devices::logging::CPU& logger){return 0;}
    template <typename DEVICE, typename ARG_1, typename ARG_2>
    void construct(DEVICE& device, devices::logging::CPU& logger, ARG_1, ARG_2){ /* noop */ }
    template <typename DEVICE>
    void construct(DEVICE& device, devices::logging::CPU& logger){ /* noop */ }
    template <typename DEVICE>
    void free(DEVICE& device, devices::logging::CPU& logger){ /* noop */ }
    template <typename DEVICE, typename TOPIC, typename ARG>
    void add_scalar(DEVICE& device, devices::logging::CPU& logger, const TOPIC, const ARG){ /* noop */ }
    template <typename DEVICE, typename TOPIC, typename ARG, typename CADENCE>
    void add_scalar(DEVICE& device, devices::logging::CPU& logger, const TOPIC, const ARG, const CADENCE){ /* noop */ }
    template <typename DEVICE, typename TOPIC, typename ARG, typename ARG_LEN, typename CADENCE>
    void add_histogram(DEVICE& device, devices::logging::CPU& logger, const TOPIC, const ARG*, const ARG_LEN, const CADENCE){ /* noop */ }
    template <typename DEVICE, typename TOPIC, typename ARG, typename ARG_LEN>
    void add_histogram(DEVICE& device, devices::logging::CPU& logger, const TOPIC, const ARG*, const ARG_LEN){ /* noop */ }
    template <typename DEVICE, typename TOPIC, typename TEXT>
    void add_text(DEVICE& device, devices::logging::CPU& logger, const TOPIC, const TEXT){ /* noop */ }
    template <typename DEVICE, typename TOPIC, typename TEXT, typename CADENCE>
    void add_text(DEVICE& device, devices::logging::CPU& logger, const TOPIC, const TEXT, const CADENCE){ /* noop */ }
    template <typename DEVICE, typename HPARAMS, typename METRICS>
    void add_hparams(DEVICE& device, devices::logging::CPU& logger, const HPARAMS&, const METRICS&){ /* noop */ }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
