#include "benchmark.h"
#include "../simulator_matrix.h"

#import <Metal/Metal.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int rl_tools_paper_benchmark_run(const char* scene_path, const char* output_directory, bool smoke_test) {
    std::ofstream log(std::string(output_directory) + "/benchmark.log");
    if(!log) {
        return 1;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if(device == nil || !device.supportsRaytracing || device.argumentBuffersSupport != MTLArgumentBuffersTier2) {
        log << "This device does not support the Metal features required by the benchmark.\n";
        return 1;
    }
    std::vector<std::string> arguments = {
        "hyperlux_paper_benchmark", "--procthor-path", scene_path,
        "--output-dir", output_directory
    };
    if(smoke_test) {
        arguments.insert(arguments.end(), {"--iterations", "1", "--warmup-seconds", "0", "--repeats", "1"});
    }
    std::vector<char*> argv;
    for(auto& argument : arguments) {
        argv.push_back(argument.data());
    }
    auto* previous_out = std::cout.rdbuf(log.rdbuf());
    auto* previous_error = std::cerr.rdbuf(log.rdbuf());
    int result = 1;
    try {
        result = rl_tools::hyperlux::benchmark::benchmark_main(static_cast<int>(argv.size()), argv.data());
    }
    catch(const std::exception& error) {
        std::cerr << error.what() << std::endl;
    }
    std::cout.flush();
    std::cerr.flush();
    std::cout.rdbuf(previous_out);
    std::cerr.rdbuf(previous_error);
    return result;
}
