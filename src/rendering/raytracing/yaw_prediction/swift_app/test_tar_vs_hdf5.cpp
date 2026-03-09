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
#include <rl_tools/persist/backends/tar/operations_posix.h>

#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>

#include "../model.h"

#include <cstdio>

namespace rlt = rl_tools;

using T = float;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = DEVICE::index_t;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

static constexpr TI BATCH_SIZE = 1;

using CAPABILITY_FORWARD = rlt::nn::capability::Forward<>;
using CAPABILITY_GRADIENT = rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient, true>;
using MODEL_FORWARD = rlt::rendering::raytracing::yaw_prediction::MODEL<CAPABILITY_FORWARD, TYPE_POLICY, TI, BATCH_SIZE>;
using MODEL_GRADIENT = rlt::rendering::raytracing::yaw_prediction::MODEL<CAPABILITY_GRADIENT, TYPE_POLICY, TI, BATCH_SIZE>;

using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, 64, 64, 3>;
using INPUT_SPEC = rlt::tensor::Specification<T, TI, INPUT_SHAPE>;
using OUTPUT_SHAPE = typename MODEL_GRADIENT::OUTPUT_SHAPE;
using OUTPUT_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <model.h5> <model.tar>\n", argv[0]);
        return 1;
    }

    DEVICE device;
    rlt::init(device);

    MODEL_FORWARD model_hdf5_fwd, model_tar_fwd;
    MODEL_GRADIENT model_hdf5, model_tar;
    typename MODEL_GRADIENT::template Buffer<true> buffer_hdf5, buffer_tar;
    rlt::Tensor<INPUT_SPEC> input_a, input_b;
    rlt::Tensor<OUTPUT_SPEC> output_hdf5, output_tar;
    typename DEVICE::SPEC::RANDOM::ENGINE<> rng;

    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    rlt::malloc(device, model_hdf5_fwd);
    rlt::malloc(device, model_tar_fwd);
    rlt::malloc(device, model_hdf5);
    rlt::malloc(device, model_tar);
    rlt::malloc(device, buffer_hdf5);
    rlt::malloc(device, buffer_tar);
    rlt::malloc(device, input_a);
    rlt::malloc(device, input_b);
    rlt::malloc(device, output_hdf5);
    rlt::malloc(device, output_tar);

    // Load HDF5 with Forward capability
    {
        auto file = HighFive::File(std::string(argv[1]), HighFive::File::ReadOnly);
        auto group = rlt::get_group(device, file, "model");
        if (!rlt::load(device, model_hdf5_fwd, group)) {
            fprintf(stderr, "Failed to load HDF5 model\n");
            return 1;
        }
    }
    printf("HDF5 model loaded\n");

    // Load tar with Forward capability
    {
        FILE* f = fopen(argv[2], "rb");
        if (!f) {
            fprintf(stderr, "Failed to open tar file\n");
            return 1;
        }
        fseek(f, 0, SEEK_END);
        long file_size = ftell(f);
        fseek(f, 0, SEEK_SET);

        rlt::persist::backends::tar::PosixFileData<TI> file_data;
        file_data.f = f;
        file_data.size = file_size;

        using READER_GROUP_SPEC = rlt::persist::backends::tar::ReaderGroupSpecification<TI, rlt::persist::backends::tar::PosixFileData<TI>>;
        rlt::persist::backends::tar::ReaderGroup<READER_GROUP_SPEC> reader_group;
        reader_group.data = file_data;

        auto group = rlt::get_group(device, reader_group, "model");
        if (!rlt::load(device, model_tar_fwd, group)) {
            fprintf(stderr, "Failed to load tar model\n");
            fclose(f);
            return 1;
        }
        fclose(f);
    }
    printf("Tar model loaded\n");

    // Compare weights (Forward models)
    T weight_diff = rlt::abs_diff(device, model_hdf5_fwd, model_tar_fwd);
    printf("Weight abs_diff: %e\n", weight_diff);

    // Copy to Gradient models for forward pass with intermediate outputs
    rlt::copy(device, device, model_hdf5_fwd, model_hdf5);
    rlt::copy(device, device, model_tar_fwd, model_tar);

    // Generate random input
    rlt::randn(device, input_a, rng);
    rlt::randn(device, input_b, rng);

    // Forward pass on both
    auto mode = rlt::Mode<rlt::mode::Default<>>{};
    rlt::forward(device, model_hdf5, input_a, input_b, output_hdf5, buffer_hdf5, rng, mode);
    rlt::forward(device, model_tar, input_a, input_b, output_tar, buffer_tar, rng, mode);

    // Compare final output
    T output_diff = rlt::abs_diff(device, output_hdf5, output_tar);
    T output_hdf5_val = rlt::get(device, output_hdf5, 0, 0);
    T output_tar_val = rlt::get(device, output_tar, 0, 0);
    printf("Output HDF5: %f\n", output_hdf5_val);
    printf("Output tar:  %f\n", output_tar_val);
    printf("Output abs_diff: %e\n", output_diff);

    // Compare intermediate layers
    auto out_a_hdf5 = rlt::output(device, model_hdf5.pipeline_a);
    auto out_a_tar = rlt::output(device, model_tar.pipeline_a);
    T pipeline_a_diff = rlt::abs_diff(device, out_a_hdf5, out_a_tar);
    printf("Pipeline A output abs_diff: %e\n", pipeline_a_diff);

    auto out_b_hdf5 = rlt::output(device, model_hdf5.pipeline_b);
    auto out_b_tar = rlt::output(device, model_tar.pipeline_b);
    T pipeline_b_diff = rlt::abs_diff(device, out_b_hdf5, out_b_tar);
    printf("Pipeline B output abs_diff: %e\n", pipeline_b_diff);

    auto out_head_hdf5 = rlt::output(device, model_hdf5.head);
    auto out_head_tar = rlt::output(device, model_tar.head);
    T head_diff = rlt::abs_diff(device, out_head_hdf5, out_head_tar);
    printf("Head output abs_diff: %e\n", head_diff);

    // Summary
    bool pass = weight_diff == 0 && output_diff == 0;
    printf("\n%s\n", pass ? "PASS: HDF5 and tar models are identical" : "FAIL: Models differ");

    rlt::free(device, output_tar);
    rlt::free(device, output_hdf5);
    rlt::free(device, input_b);
    rlt::free(device, input_a);
    rlt::free(device, buffer_tar);
    rlt::free(device, buffer_hdf5);
    rlt::free(device, model_tar);
    rlt::free(device, model_hdf5);
    rlt::free(device, model_tar_fwd);
    rlt::free(device, model_hdf5_fwd);
    rlt::free(device, rng);

    return pass ? 0 : 1;
}
