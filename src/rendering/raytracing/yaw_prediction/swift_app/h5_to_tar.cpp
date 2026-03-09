#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/operations_generic.h>

#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>

#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/persist/backends/tar/operations_cpu.h>

#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>

#include "../model.h"

#include <iostream>
#include <fstream>

namespace rlt = rl_tools;

using T = float;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = DEVICE::index_t;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

static constexpr TI BATCH_SIZE = 1;
using CAPABILITY = rlt::nn::capability::Forward<>;
using MODEL = rlt::rendering::raytracing::yaw_prediction::MODEL<CAPABILITY, TYPE_POLICY, TI, BATCH_SIZE>;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.h5> <output.tar>" << std::endl;
        return 1;
    }

    DEVICE device;
    rlt::init(device);

    MODEL model;
    rlt::malloc(device, model);

    try {
        auto file = HighFive::File(std::string(argv[1]), HighFive::File::ReadOnly);
        auto model_group = rlt::get_group(device, file, "model");
        if (!rlt::load(device, model, model_group)) {
            std::cerr << "Failed to load model from " << argv[1] << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading HDF5: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Model loaded from " << argv[1] << std::endl;

    rlt::persist::backends::tar::Writer writer;
    using WRITER_GROUP_SPEC = rlt::persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>;
    rlt::persist::backends::tar::WriterGroup<WRITER_GROUP_SPEC> writer_group;
    writer_group.writer = &writer;

    auto model_group = rlt::create_group(device, writer_group, "model");
    rlt::save(device, model, model_group);
    rlt::persist::backends::tar::finalize(device, writer);

    std::ofstream out(argv[2], std::ios::binary);
    out.write(writer.buffer.data(), writer.buffer.size());
    out.close();

    std::cout << "Model saved to " << argv[2] << " (" << writer.buffer.size() << " bytes)" << std::endl;

    rlt::free(device, model);
    return 0;
}
