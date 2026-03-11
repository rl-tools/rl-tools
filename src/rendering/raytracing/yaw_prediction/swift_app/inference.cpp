#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/operations_generic.h>

#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>

#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#endif
#include <rl_tools/persist/backends/tar/operations_posix.h>

#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>

#include "../model.h"
#include "inference.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace rlt = rl_tools;

using T = float;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = DEVICE::index_t;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

static constexpr TI BATCH_SIZE = 1;
static constexpr TI CAM_WIDTH = 64;
static constexpr TI CAM_HEIGHT = 64;

using CAPABILITY = rlt::nn::capability::Forward<>;
using MODEL = rlt::rendering::raytracing::yaw_prediction::MODEL<CAPABILITY, TYPE_POLICY, TI, BATCH_SIZE>;

using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, CAM_HEIGHT, CAM_WIDTH, 3>;
using INPUT_SPEC = rlt::tensor::Specification<T, TI, INPUT_SHAPE>;
using OUTPUT_SHAPE = typename MODEL::OUTPUT_SHAPE;
using OUTPUT_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>;

struct YawPredictorHandle {
    DEVICE device;
    MODEL model;
    typename MODEL::template Buffer<true> buffer;
    rlt::Tensor<INPUT_SPEC> input_a;
    rlt::Tensor<INPUT_SPEC> input_b;
    rlt::Tensor<OUTPUT_SPEC> output;
    typename DEVICE::SPEC::RANDOM::ENGINE<> rng;
};

static bool load_from_tar(YawPredictorHandle* handle, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        return false;
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

    auto model_group = rlt::get_group(handle->device, reader_group, "model");
    bool success = rlt::load(handle->device, handle->model, model_group);

    fclose(f);
    return success;
}

extern "C" {

YawPredictorHandle* yaw_predictor_create(const char* model_path) {
    auto* handle = new YawPredictorHandle();

    rlt::init(handle->device);
    rlt::malloc(handle->device, handle->rng);
    rlt::init(handle->device, handle->rng, 0);

    rlt::malloc(handle->device, handle->model);
    rlt::malloc(handle->device, handle->buffer);
    rlt::malloc(handle->device, handle->input_a);
    rlt::malloc(handle->device, handle->input_b);
    rlt::malloc(handle->device, handle->output);

    bool success = false;

    // Detect format by extension
    size_t len = strlen(model_path);
    bool is_tar = (len > 4 && strcmp(model_path + len - 4, ".tar") == 0);

    if (is_tar) {
        success = load_from_tar(handle, model_path);
    } else {
#ifdef RL_TOOLS_ENABLE_HDF5
        try {
            auto file = HighFive::File(std::string(model_path), HighFive::File::ReadOnly);
            auto model_group = rlt::get_group(handle->device, file, "model");
            success = rlt::load(handle->device, handle->model, model_group);
        } catch (const std::exception& e) {
            fprintf(stderr, "Error loading HDF5: %s\n", e.what());
        }
#else
        fprintf(stderr, "HDF5 support not compiled in. Use .tar format.\n");
#endif
    }

    if (!success) {
        fprintf(stderr, "Failed to load model from: %s\n", model_path);
        yaw_predictor_destroy(handle);
        return nullptr;
    }

    return handle;
}

void yaw_predictor_evaluate(YawPredictorHandle* handle,
                            const float* image_a,
                            const float* image_b,
                            float* output) {
    constexpr TI IMAGE_SIZE = BATCH_SIZE * CAM_HEIGHT * CAM_WIDTH * 3;
    std::memcpy(rlt::data(handle->input_a), image_a, IMAGE_SIZE * sizeof(float));
    std::memcpy(rlt::data(handle->input_b), image_b, IMAGE_SIZE * sizeof(float));

    auto mode = rlt::Mode<rlt::mode::Default<>>{};
    rlt::evaluate(handle->device, handle->model,
                  handle->input_a, handle->input_b,
                  handle->output, handle->buffer,
                  handle->rng, mode);

    for (TI i = 0; i < 3; i++) {
        output[i] = rlt::get(handle->device, handle->output, 0, i);
    }
}

void yaw_predictor_destroy(YawPredictorHandle* handle) {
    if (!handle) return;
    rlt::free(handle->device, handle->output);
    rlt::free(handle->device, handle->input_b);
    rlt::free(handle->device, handle->input_a);
    rlt::free(handle->device, handle->buffer);
    rlt::free(handle->device, handle->model);
    rlt::free(handle->device, handle->rng);
    delete handle;
}

} // extern "C"
