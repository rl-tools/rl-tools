#pragma once

#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/yaw/operations_generic.h>

#include <array>
#include <cmath>
#include <stdexcept>

namespace rl_tools::rl::environments::hyperdrone::tasks::target_frame::yaw{
    struct ValidationSummary{
        T yaw_mse_loss = 0;
        T yaw_abs_error_rad = 0;
        T yaw_mse_rad = 0;
        T yaw_output_norm = 0;
        T yaw_null_mse_loss = 0;
        T yaw_null_mse_rad = 0;
        T yaw_angle_r2_uniform = 0;
        T frontier_yaw_mse_rad = 0;
        std::array<T, N_VALIDATION_YAW_BINS> bin_yaw_mse_rad{};
    };

    struct ValidationState{
        using MODEL = typename STUDENT_TYPE::template CHANGE_CAPABILITY<nn::capability::Forward<true>>::template CHANGE_BATCH_SIZE<TI, VALIDATION_BATCH_SIZE>;
        using PARAMETERS_SPEC = tensor::Specification<typename TASK_WORLD::Parameters, TI, tensor::Shape<TI, VALIDATION_BATCH_SIZE>>;
        using STATES_SPEC = tensor::Specification<typename TASK_WORLD::State, TI, tensor::Shape<TI, VALIDATION_BATCH_SIZE>>;
        using RESET_SPEC = tensor::Specification<bool, TI, tensor::Shape<TI, VALIDATION_BATCH_SIZE>>;
        using OBSERVATIONS_SPEC = tensor::Specification<T_ACTIVATION, TI, tensor::Shape<TI, VALIDATION_BATCH_SIZE, COMBINED_OBS_DIM>>;
        using OUTPUT_SPEC = tensor::Specification<T_ACTIVATION, TI, tensor::Shape<TI, 1, VALIDATION_BATCH_SIZE, TARGET_DIM>>;
        using HOST_OUTPUT_SPEC = tensor::Specification<T, TI, tensor::Shape<TI, 1, VALIDATION_BATCH_SIZE, TARGET_DIM>>;
        MODEL model;
        typename MODEL::template Buffer<true> buffer;
        Tensor<PARAMETERS_SPEC> parameters, parameters_host;
        Tensor<STATES_SPEC> states, states_host;
        Tensor<RESET_SPEC> reset;
        Tensor<OBSERVATIONS_SPEC> observations;
        Tensor<OUTPUT_SPEC> output;
        Tensor<HOST_OUTPUT_SPEC> output_host;
        std::array<T, VALIDATION_BATCH_SIZE> target_yaw{};
        RNG_COMPUTE rng;
    };
}

namespace rl_tools{
    namespace yaw_validation = rl::environments::hyperdrone::tasks::target_frame::yaw;

    inline void malloc(yaw_validation::DEVICE& host, yaw_validation::DEVICE_COMPUTE& device, yaw_validation::ValidationState& state){
        malloc(device, state.model);
        malloc(device, state.buffer);
        malloc(device, state.parameters);
        malloc(host, state.parameters_host);
        malloc(device, state.states);
        malloc(host, state.states_host);
        malloc(device, state.reset);
        malloc(device, state.observations);
        malloc(device, state.output);
        malloc(host, state.output_host);
        malloc(device, state.rng);
    }

    inline void init(yaw_validation::DEVICE&, yaw_validation::DEVICE_COMPUTE& device, yaw_validation::ValidationState& state, yaw_validation::TI seed){
        init(device, state.rng, seed);
        set_all(device, state.reset, true);
    }

    inline void free(yaw_validation::DEVICE& host, yaw_validation::DEVICE_COMPUTE& device, yaw_validation::ValidationState& state){
        free(device, state.model);
        free(device, state.buffer);
        free(device, state.parameters);
        free(host, state.parameters_host);
        free(device, state.states);
        free(host, state.states_host);
        free(device, state.reset);
        free(device, state.observations);
        free(device, state.output);
        free(host, state.output_host);
        free(device, state.rng);
    }

    inline yaw_validation::ValidationSummary validate_batch(yaw_validation::DEVICE& host, yaw_validation::DEVICE_COMPUTE& device, yaw_validation::TASK_WORLD& world, yaw_validation::ValidationState& state){
        using namespace yaw_validation;
        copy(host, device, state.parameters_host, state.parameters);
        copy(host, device, state.states_host, state.states);
        render_target_frames(device, world, state.parameters, state.reset);
        request_render(device, world, state.reset);
        observe(device, world, state.parameters, state.states, typename TASK_WORLD::Observation{}, state.observations, state.rng);
        auto images = reshape_row_major(device, state.observations, tensor::Shape<TI, 1, VALIDATION_BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>{});
        auto inputs = nn_models::parallel::pack_inputs(images);
        evaluate(device, state.model, inputs, state.output, state.buffer, state.rng, Mode<mode::Evaluation<>>{});
        copy(device, host, state.output, state.output_host);
        ValidationSummary result;
        for(TI sample = 0; sample < VALIDATION_BATCH_SIZE; sample++){
            const T predicted_cos = get(host, state.output_host, 0, sample, 0);
            const T predicted_sin = get(host, state.output_host, 0, sample, 1);
            if(!std::isfinite(predicted_cos) || !std::isfinite(predicted_sin)){
                throw std::runtime_error("Non-finite yaw validation prediction");
            }
            const T target_cos = static_cast<T>(static_cast<T_ACTIVATION>(std::cos(state.target_yaw[sample])));
            const T target_sin = static_cast<T>(static_cast<T_ACTIVATION>(std::sin(state.target_yaw[sample])));
            const T error = yaw_error(host, predicted_cos, predicted_sin, target_cos, target_sin);
            const T cos_difference = predicted_cos - target_cos;
            const T sin_difference = predicted_sin - target_sin;
            if(!std::isfinite(error) || !std::isfinite(cos_difference * cos_difference + sin_difference * sin_difference)){
                throw std::runtime_error("Non-finite yaw validation error");
            }
            result.yaw_mse_loss += (cos_difference * cos_difference + sin_difference * sin_difference) / (T)2;
            result.yaw_abs_error_rad += std::abs(error);
            result.yaw_mse_rad += error * error;
            result.yaw_output_norm += std::sqrt(predicted_cos * predicted_cos + predicted_sin * predicted_sin);
            result.yaw_null_mse_loss += ((1 - target_cos) * (1 - target_cos) + target_sin * target_sin) / (T)2;
            const T null_error = yaw_error(host, (T)1, (T)0, target_cos, target_sin);
            result.yaw_null_mse_rad += null_error * null_error;
        }
        result.yaw_mse_loss /= VALIDATION_BATCH_SIZE;
        result.yaw_abs_error_rad /= VALIDATION_BATCH_SIZE;
        result.yaw_mse_rad /= VALIDATION_BATCH_SIZE;
        result.yaw_output_norm /= VALIDATION_BATCH_SIZE;
        result.yaw_null_mse_loss /= VALIDATION_BATCH_SIZE;
        result.yaw_null_mse_rad /= VALIDATION_BATCH_SIZE;
        if(!std::isfinite(result.yaw_mse_loss)){
            throw std::runtime_error("Non-finite yaw validation loss");
        }
        return result;
    }

    inline yaw_validation::ValidationSummary validate(yaw_validation::DEVICE& host, yaw_validation::DEVICE_COMPUTE& device, yaw_validation::TASK_WORLD& world, const yaw_validation::STUDENT_TYPE& student, yaw_validation::ValidationState& state, yaw_validation::T frontier_min_deg, yaw_validation::T frontier_max_deg){
        using namespace yaw_validation;
        static constexpr std::array<T, N_VALIDATION_YAW_BINS> yaw_degrees = {-2, -4, -8, -16, -32, 0, 2, 4, 8, 16, 32};
        copy(device, device, student, state.model);
        ValidationSummary summary;
        for(TI scene = 0; scene < N_VALIDATION_SCENES; scene++){
            select_scene(device, world, scene);
            const auto& annotations = world.slots[scene].annotations;
            if(annotations.num_positions == 0){
                throw std::runtime_error("Yaw validation scene has no indoor positions");
            }
            for(TI bin = 0; bin <= N_VALIDATION_YAW_BINS; bin++){
                const bool frontier = bin == N_VALIDATION_YAW_BINS;
                for(TI sample = 0; sample < VALIDATION_BATCH_SIZE; sample++){
                    auto& parameters = get_ref(host, state.parameters_host, sample);
                    parameters = {};
                    static_cast<typename BASE_WORLD::Parameters&>(parameters) = world.parameters;
                    parameters.agents[0].cameras[0] = {};
                    parameters.agents[0].cameras[0].fov = CAMERA_FOV;
                    parameters.brightness_scale = 1;
                    parameters.brightness_mismatch = 1;
                    parameters.target_roll = 0;
                    parameters.target_pitch = 0;
                    const auto& position = annotations.positions[sample % annotations.num_positions];
                    for(TI axis = 0; axis < 3; axis++){
                        parameters.scene_translation[axis] = position.position[axis];
                    }
                    const TI scene_yaw_index = frontier ? (sample + scene * 17 + 19) % VALIDATION_BATCH_SIZE : (sample + scene * 7 + bin * 13) % VALIDATION_BATCH_SIZE;
                    const T scene_yaw = 2 * math::PI<T> * (T)scene_yaw_index / (T)VALIDATION_BATCH_SIZE;
                    parameters.scene_yaw_cos = std::cos(scene_yaw);
                    parameters.scene_yaw_sin = std::sin(scene_yaw);
                    T angle = yaw_degrees[frontier ? 0 : bin] * math::PI<T> / (T)180;
                    if(frontier){
                        const T alpha = ((T)(sample / 2) + (T)0.5) / (T)(VALIDATION_BATCH_SIZE / 2);
                        const T frontier_min_rad = frontier_min_deg / (T)180 * math::PI<T>;
                        const T frontier_max_rad = frontier_max_deg / (T)180 * math::PI<T>;
                        const T angle_abs = frontier_min_rad + (frontier_max_rad - frontier_min_rad) * alpha;
                        angle = sample % 2 == 0 ? -angle_abs : angle_abs;
                    }
                    state.target_yaw[sample] = angle;
                    auto& fixture_state = get_ref(host, state.states_host, sample);
                    fixture_state = {};
                    fixture_state.orientation[0] = std::cos(angle / (T)2);
                    fixture_state.orientation[3] = std::sin(angle / (T)2);
                }
                const auto batch = validate_batch(host, device, world, state);
                if(frontier){
                    summary.frontier_yaw_mse_rad += batch.yaw_mse_rad / (T)N_VALIDATION_SCENES;
                }
                else{
                    const T weight = (T)1 / (T)(N_VALIDATION_SCENES * N_VALIDATION_YAW_BINS);
                    summary.yaw_mse_loss += batch.yaw_mse_loss * weight;
                    summary.yaw_abs_error_rad += batch.yaw_abs_error_rad * weight;
                    summary.yaw_mse_rad += batch.yaw_mse_rad * weight;
                    summary.yaw_output_norm += batch.yaw_output_norm * weight;
                    summary.yaw_null_mse_loss += batch.yaw_null_mse_loss * weight;
                    summary.yaw_null_mse_rad += batch.yaw_null_mse_rad * weight;
                    summary.bin_yaw_mse_rad[bin] += batch.yaw_mse_rad / (T)N_VALIDATION_SCENES;
                }
            }
        }
        summary.yaw_angle_r2_uniform = 1 - summary.yaw_mse_rad / (summary.yaw_null_mse_rad > (T)1e-8 ? summary.yaw_null_mse_rad : (T)1e-8);
        return summary;
    }
}
