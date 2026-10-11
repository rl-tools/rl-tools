#pragma once

#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/flatten/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>

namespace rl_tools{
    namespace yaw_checkpoint = rl::environments::hyperdrone::tasks::target_frame::yaw;

    template<yaw_checkpoint::TI SAVE_BATCH_SIZE, typename INPUT_SPEC, typename OUTPUT_SPEC>
    yaw_checkpoint::T save_yaw_checkpoint(yaw_checkpoint::DEVICE& host, yaw_checkpoint::CPU_STUDENT_TYPE& student, Tensor<INPUT_SPEC>& examples, Tensor<OUTPUT_SPEC>& predictions, yaw_checkpoint::RNG& rng, const std::filesystem::path& directory, const std::string& metadata){
        using namespace yaw_checkpoint;
        using MODEL = typename CPU_STUDENT_TYPE::template CHANGE_BATCH_SIZE<TI, SAVE_BATCH_SIZE>;
        using IMAGE_SHAPE = tensor::Shape<TI, 1, SAVE_BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>;
        using OUTPUT_SHAPE = tensor::Shape<TI, 1, SAVE_BATCH_SIZE, TARGET_DIM>;
        MODEL model;
        typename MODEL::template Buffer<true> buffer;
        Tensor<tensor::Specification<T, TI, IMAGE_SHAPE>> loaded_examples;
        Tensor<tensor::Specification<T, TI, OUTPUT_SHAPE>> loaded_predictions, reloaded_output;
        malloc(host, model);
        malloc(host, buffer);
        malloc(host, loaded_examples);
        malloc(host, loaded_predictions);
        malloc(host, reloaded_output);
        copy(host, host, student, model);
        auto example_view = view_range(host, examples, (TI)0, tensor::ViewSpec<1, SAVE_BATCH_SIZE>{});
        auto prediction_view = view_range(host, predictions, (TI)0, tensor::ViewSpec<1, SAVE_BATCH_SIZE>{});
        const auto path = directory / ("checkpoint_" + std::to_string(SAVE_BATCH_SIZE) + "examples.h5");
        {
            std::lock_guard<std::mutex> lock(persist::backends::hdf5::global_mutex());
            persist::backends::hdf5::File file(path.string(), persist::backends::hdf5::Mode::WRITE);
            auto actor = create_group(host, file, "actor");
            set_attribute(host, actor, "checkpoint_name", directory.string().c_str());
            set_attribute(host, actor, "meta", metadata.c_str());
            save(host, model, actor);
            auto example = create_group(host, file, "example");
            auto inputs = create_group(host, example, "inputs");
            save(host, example_view, inputs, "0");
            auto outputs = create_group(host, example, "outputs");
            save(host, prediction_view, outputs, "0");
        }
        {
            std::lock_guard<std::mutex> lock(persist::backends::hdf5::global_mutex());
            persist::backends::hdf5::File file(path.string(), persist::backends::hdf5::Mode::READ);
            auto actor = get_group(host, file, "actor");
            if(!load(host, model, actor)){
                throw std::runtime_error("Failed to reload yaw checkpoint model");
            }
            auto example = get_group(host, file, "example");
            auto inputs = get_group(host, example, "inputs");
            if(!load(host, loaded_examples, inputs, "0")){
                throw std::runtime_error("Failed to reload canonical yaw checkpoint inputs");
            }
            auto outputs = get_group(host, example, "outputs");
            if(!load(host, loaded_predictions, outputs, "0")){
                throw std::runtime_error("Failed to reload canonical yaw checkpoint outputs");
            }
        }
        auto inputs = nn_models::parallel::pack_inputs(loaded_examples);
        evaluate(host, model, inputs, reloaded_output, buffer, rng, Mode<mode::Evaluation<>>{});
        T max_error = 0;
        for(TI sample = 0; sample < SAVE_BATCH_SIZE; sample++){
            for(TI component = 0; component < TARGET_DIM; component++){
                const T error = std::abs(get(host, reloaded_output, 0, sample, component) - get(host, loaded_predictions, 0, sample, component));
                if(!std::isfinite(error)){
                    throw std::runtime_error("Non-finite yaw checkpoint reload prediction");
                }
                max_error = std::max(max_error, error);
            }
        }
        free(host, reloaded_output);
        free(host, loaded_predictions);
        free(host, loaded_examples);
        free(host, buffer);
        free(host, model);
        return max_error;
    }

    template<typename OBSERVATION_SPEC>
    yaw_checkpoint::T save_checkpoint(yaw_checkpoint::DEVICE& host, yaw_checkpoint::DEVICE_COMPUTE& device, const yaw_checkpoint::STUDENT_TYPE& student, Tensor<OBSERVATION_SPEC>& observations, const std::filesystem::path& directory){
        using namespace yaw_checkpoint;
        static_assert(get<0>(typename OBSERVATION_SPEC::SHAPE{}) >= STEPS_TOTAL);
        static_assert(get<1>(typename OBSERVATION_SPEC::SHAPE{}) == COMBINED_OBS_DIM);
        static_assert(BATCH_SIZE == N_EXAMPLES);
        CPU_STUDENT_TYPE model;
        typename CPU_STUDENT_TYPE::template Buffer<true> buffer;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1, N_EXAMPLES, IMG_H, IMG_W, COMBINED_IMG_C>>> examples;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1, N_EXAMPLES, TARGET_DIM>>> predictions;
        RNG rng;
        malloc(host, model);
        malloc(host, buffer);
        malloc(host, examples);
        malloc(host, predictions);
        malloc(host, rng);
        init(host, rng, 0);
        copy(device, host, student, model);
        auto source = view_range(device, observations, (TI)(STEPS_TOTAL - N_EXAMPLES), tensor::ViewSpec<0, N_EXAMPLES>{});
        auto destination = reshape_row_major(host, examples, tensor::Shape<TI, N_EXAMPLES, COMBINED_OBS_DIM>{});
        copy(device, host, source, destination);
        auto inputs = nn_models::parallel::pack_inputs(examples);
        evaluate(host, model, inputs, predictions, buffer, rng, Mode<mode::Evaluation<>>{});
        std::filesystem::create_directories(directory);
        const std::string observation = "CameraRGBStackedWithTarget(" + std::to_string(CAMERA_FOV) + ", " + std::to_string(CAM_HEIGHT) + ", " + std::to_string(CAM_WIDTH) + ", " + std::to_string(FRAME_STACK_STRIDE) + ", " + std::to_string(FRAME_STACK_N) + ")";
        const std::string rendering = "{\"rgb_encoding\":\"float before brightness\",\"anti_aliasing\":true,\"anti_aliasing_grid_size\":" + std::to_string(RENDER_ANTI_ALIASING_GRID_SIZE) + ",\"motion_blur\":true,\"motion_blur_samples\":" + std::to_string(RENDER_MOTION_BLUR_SAMPLES) + ",\"shutter_fraction_min\":" + std::to_string(RENDER_SHUTTER_FRACTION_MIN) + ",\"shutter_fraction_max\":" + std::to_string(RENDER_SHUTTER_FRACTION_MAX) + ",\"target_frame_roll_pitch_randomization_range\":" + std::to_string(TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE) + ",\"target_frame_brightness_mismatch_range\":" + std::to_string(TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE) + "}";
        const std::string metadata = "{\"environment\":{\"name\":\"hyperdrone\",\"observation\":\"" + observation + "\",\"output\":\"YawOffsetCosSin\",\"rendering\":" + rendering + "},\"task\":\"target_frame_yaw\",\"baseline_commit\":\"c0c7c31d95e9be0a1e5e926e3b419fe54e0fb164\",\"run_manifest\":\"../../run.json\"}";
        T max_error = save_yaw_checkpoint<REDUCED_BATCH_SIZE>(host, model, examples, predictions, rng, directory, metadata);
        max_error = std::max(max_error, save_yaw_checkpoint<N_EXAMPLES>(host, model, examples, predictions, rng, directory, metadata));
        free(host, rng);
        free(host, predictions);
        free(host, examples);
        free(host, buffer);
        free(host, model);
        return max_error;
    }
}
