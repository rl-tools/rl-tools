#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_cuda.h>
#include <rl_tools/nn/layers/dense/operations_cuda.h>
#include <rl_tools/nn/layers/gru/helper_operations_cuda.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_cuda.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include "scenes.h"
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cuda.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/yaw/operations_cuda.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_mux.h>
#include <rl_tools/rendering/datasets/procthor/operations_cpu.h>
#include <rl_tools/nn/loss_functions/mse/operations_generic.h>
#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>
#include "../../../../../../nn_models/port_checkpoint/raptor/policy.h"
#include <rl_tools/utils/extrack/operations_cpu.h>
#include <metra/metra.h>
#include <nlohmann/json.hpp>
#include <cuda_bf16.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace rlt = rl_tools;
#include "config.h"
#include "validation.h"
#include "checkpoint.h"

namespace rl_tools{
    namespace yaw_training = rl::environments::hyperdrone::tasks::target_frame::yaw;

    inline void yaw_synchronize(yaw_training::DEVICE_COMPUTE& device){
        const auto status = cudaStreamSynchronize(device.stream);
        if(status != cudaSuccess){
            throw std::runtime_error(cudaGetErrorString(status));
        }
    }

    template<typename OUTPUT_SPEC, typename TARGET_SPEC>
    yaw_training::T yaw_batch_mse(yaw_training::DEVICE& host, const Tensor<OUTPUT_SPEC>& output, const Tensor<TARGET_SPEC>& targets){
        using namespace yaw_training;
        T loss = 0;
        for(TI sample = 0; sample < BATCH_SIZE; sample++){
            for(TI component = 0; component < TARGET_DIM; component++){
                const T difference = get(host, output, 0, sample, component) - get(host, targets, sample, component);
                loss += difference * difference / (T)(BATCH_SIZE * TARGET_DIM);
            }
        }
        if(!std::isfinite(loss)){
            throw std::runtime_error("Non-finite yaw predictions or labels");
        }
        return loss;
    }

    inline yaw_training::Scenes yaw_scenes(const std::filesystem::path& path){
        using namespace yaw_training;
        Scenes scenes;
        if(std::filesystem::is_directory(path)){
            for(const auto& entry: std::filesystem::directory_iterator(path)){
                if(entry.is_regular_file() && entry.path().extension() == ".glb"){
                    scenes.references.push_back(std::filesystem::absolute(entry.path()).string());
                }
            }
            std::sort(scenes.references.begin(), scenes.references.end(), rendering::datasets::natural_less);
            if(scenes.references.size() < N_TOTAL_SCENES){
                throw std::runtime_error("Need at least " + std::to_string(N_TOTAL_SCENES) + " local GLB scenes");
            }
            scenes.references.resize(N_TOTAL_SCENES);
        } else if(SMOKE && path.extension() == ".glb" && std::filesystem::is_regular_file(path)){
            scenes.references.assign(N_TOTAL_SCENES, std::filesystem::absolute(path).string());
        } else {
            throw std::runtime_error("Expected a scene directory (the smoke target also accepts one local .glb)");
        }
        return scenes;
    }

    inline void yaw_manifest(yaw_training::DEVICE& host, const yaw_training::Scenes& scenes, const yaw_training::MULTI_ENVIRONMENT& environment, const yaw_training::TASK_WORLD& validation_world, const utils::extrack::Paths& paths, yaw_training::TI seed){
        using namespace yaw_training;
        nlohmann::json manifest = {
            {"target", RL_TOOLS_HYPERDRONE_TARGET_NAME}, {"seed", seed}, {"profile", SMOKE ? "smoke" : "production"},
            {"baseline_commit", "c0c7c31d95e9be0a1e5e926e3b419fe54e0fb164"}, {"baseline_checkpoint", "c0c7c31_000000659206400_checkpoint_512examples.h5"},
            {"teacher", {{"header_sha256", RL_TOOLS_HYPERDRONE_YAW_TEACHER_SHA256}, {"training_commit", checkpoint::meta::commit_hash}, {"name", checkpoint::meta::name}, {"observation", "Position+RotationMatrix+LinearVelocity+AngularVelocity+GyroBias+ActionHistory1"}, {"dimensions", RAPTOR_OBS_DIM}}},
            {"training", {{"epochs", NUM_EPOCHS}, {"train_scenes", N_TRAIN_SCENES}, {"validation_scenes", N_VALIDATION_SCENES}, {"active_scenes", N_ACTIVE_SCENES}, {"instances_per_scene", N_ENVIRONMENTS_PER_SCENE}, {"steps_per_epoch", STEPS_TOTAL}, {"steps_per_instance", STEPS_PER_ENV}, {"episode_limit", EPISODE_STEP_LIMIT}, {"batch_size", BATCH_SIZE}, {"batches_per_pass", N_BATCHES}, {"passes", N_TRAIN_PASSES}, {"trained_rows_per_pass", TRAINED_ROWS_PER_PASS}, {"shuffle", "batch order only"}, {"optimizer", "Adam"}, {"learning_rate", 0.001}, {"beta1", 0.9}, {"beta2", 0.999}, {"epsilon", 1e-8}, {"epsilon_sqrt", 1e-8}, {"loss", "0.5*MSE(cos,sin)"}, {"precision", "BF16 parameters/activations/gradients/labels; float masters and teacher"}}},
            {"model", {{"input", {1, BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C}}, {"channels", "current RGB,target RGB,0,0"}, {"convolutions", {32, 64, 128, 256}}, {"kernel", 3}, {"stride", 2}, {"padding", 1}, {"dense", {64, 64, 64, 2}}, {"hidden_activation", "ReLU"}, {"output_activation", "identity"}}},
            {"environment", {{"dynamics", "crazyflie_openmv"}, {"frequency_hz", SIMULATION_FREQUENCY}, {"action_interface", "direct motor"}, {"substeps", 1}, {"guidance", 0}, {"initial_position", 0}, {"initial_linear_velocity", 0}, {"initial_angular_velocity", 1}, {"gyro_bias", 0.02}, {"thrust_to_weight", {1.7, 2.0}}, {"other_physical_randomization", false}, {"disturbances", 0}}},
            {"rendering", {{"backend", "OptiX"}, {"shading", "High/PBR"}, {"rgb_encoding", "float before brightness"}, {"ambient", {0.5, 0.5, 0.5}}, {"directional_fill_lights", 3}, {"honor_unlit_materials", false}, {"background", "solid black"}, {"camera_fov_deg", CAMERA_FOV}, {"fov_randomization_deg", CAMERA_FOV_RANDOMIZATION_RANGE}, {"mount_translation_range_m", CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE}, {"mount_rotation_range_rad", CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE}, {"target_roll_pitch_range_rad", TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE}, {"brightness_range", {0.5, 1.5}}, {"target_brightness_multiplier", {0.75, 1.25}}, {"aa_grid", RENDER_ANTI_ALIASING_GRID_SIZE}, {"motion_blur_samples", RENDER_MOTION_BLUR_SAMPLES}, {"shutter_fraction", {RENDER_SHUTTER_FRACTION_MIN, RENDER_SHUTTER_FRACTION_MAX}}, {"target_bank_refresh_steps", TASK_SPEC::TARGET_BANK_REFRESH_STEPS}}},
            {"curriculum", {{"orientation_sampler", "uniform 3D axis and angle"}, {"start_deg", INIT_ORIENTATION_CURRICULUM_START_DEG}, {"full_deg", INIT_ORIENTATION_CURRICULUM_FULL_DEG}, {"increment_deg", INIT_ORIENTATION_CURRICULUM_STEP_DEG}, {"frontier_ratio", INIT_ORIENTATION_CURRICULUM_FRONTIER_RATIO}, {"threshold_min_deg", INIT_ORIENTATION_CURRICULUM_FRONTIER_MIN_DEG}, {"threshold_max_deg", INIT_ORIENTATION_CURRICULUM_FRONTIER_MAX_DEG}, {"validation_cadence", VALIDATION_CADENCE}, {"fixed_bins_deg", {-2, -4, -8, -16, -32, 0, 2, 4, 8, 16, 32}}}},
            {"checkpoint", {{"cadence", CHECKPOINT_CADENCE}, {"examples", {REDUCED_BATCH_SIZE, N_EXAMPLES}}, {"example_row_offset", STEPS_TOTAL - N_EXAMPLES}, {"optimizer_state", false}}},
            {"accepted_differences", {"Float RGB instead of RGB8 before brightness; exact-count free-space stopping instead of completing 64-candidate blocks; native validation batch 64 instead of padded 512", "Current metallic ambient shading, black instead of checkerboard background, GLB loading and random-number draw order", "Early resets reuse the epoch target bank setup while resampling dynamics", "Same-step autoreset and a final untrained next-observation row", "Current HDF5 model format; no legacy OpenMV serialization or exact optimizer resume"}}
        };
        manifest["scenes"] = nlohmann::json::array();
        for(TI scene = 0; scene < N_TOTAL_SCENES; scene++){
            std::string hash;
            if(!rendering::datasets::content_hash(host, scenes.references[scene], hash)){
                throw std::runtime_error("Could not hash scene: " + scenes.references[scene]);
            }
            manifest["scenes"].push_back({{"path", scenes.references[scene]}, {"sha1", hash}, {"split", scene < N_TRAIN_SCENES ? "train" : "validation"}});
            const auto& annotations = scene < N_TRAIN_SCENES ? environment.environments[0].slots[scene].annotations : validation_world.slots[scene - N_TRAIN_SCENES].annotations;
            auto& positions = manifest["scenes"].back()["free_positions"];
            positions = nlohmann::json::array();
            for(TI index = 0; index < annotations.num_positions; index++){
                const auto& position = annotations.positions[index];
                positions.push_back({{"position", position.position}, {"yaw", position.yaw}, {"score", position.score}});
            }
        }
        bool held_out = true;
        for(TI train = 0; train < N_TRAIN_SCENES; train++){
            for(TI validation = N_TRAIN_SCENES; validation < N_TOTAL_SCENES; validation++){
                held_out = held_out && manifest["scenes"][train]["sha1"] != manifest["scenes"][validation]["sha1"];
            }
        }
        manifest["validation_is_held_out"] = held_out;
        std::filesystem::create_directories(paths.seed);
        std::ofstream output(paths.seed / "run.json");
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output << manifest.dump(2) << '\n';
    }

    inline int train_yaw(const yaw_training::Scenes& scenes, yaw_training::TI seed){
        using namespace yaw_training;
        DEVICE_COMPUTE compute;
        auto& host = compute.rendering;
        init(host);
        init(compute);
        host.logger.verbosity = devices::logging::Verbosity::warning;
        utils::extrack::Config<TI> extrack_config;
        utils::extrack::Paths paths;
        extrack_config.name = "hyperdrone-yaw";
        extrack_config.population_variates = "profile";
        extrack_config.population_values = SMOKE ? "smoke" : "production";
        init(host, extrack_config, paths, seed);
        std::ofstream metrics(paths.seed / "metrics.jsonl");
        metrics.exceptions(std::ios::failbit | std::ios::badbit);

        RNG rng;
        std::mt19937 scene_rng(seed);
        std::array<TI, N_TRAIN_SCENES> scene_permutation;
        for(TI scene = 0; scene < N_TRAIN_SCENES; scene++){
            scene_permutation[scene] = scene;
        }
        RNG_COMPUTE compute_rng;
        malloc(host, rng);
        malloc(compute, compute_rng);
        init(host, rng, seed);
        init(compute, compute_rng, seed);

        RAPTOR_MODEL teacher_host, teacher;
        typename RAPTOR_MODEL::Buffer<true> teacher_buffer;
        RUNNER runner;
        RUNNER_BUFFER runner_buffer;
        malloc(host, teacher_host);
        malloc(compute, teacher);
        malloc(compute, teacher_buffer);
        malloc(compute, runner);
        malloc(compute, runner_buffer);
        copy(host, host, checkpoint::actor::module, teacher_host);
        copy(host, compute, teacher_host, teacher);
        reset(compute, teacher, runner.policy_state, compute_rng);

        CPU_STUDENT_INIT_TYPE student_host;
        STUDENT_TYPE student;
        STUDENT_BUFFERS student_buffer;
        OPTIMIZER optimizer;
        malloc(host, student_host);
        init_weights(host, student_host, rng);
        malloc(compute, student);
        malloc(compute, student_buffer);
        malloc(compute, optimizer);
        copy(host, compute, student_host, student);
        init(compute, optimizer);
        reset_optimizer_state(compute, optimizer, student);
        free(host, student_host);

        auto environment_storage = std::make_unique<MULTI_ENVIRONMENT>();
        auto validation_world_storage = std::make_unique<TASK_WORLD>();
        auto& environment = *environment_storage;
        auto& validation_world = *validation_world_storage;
        malloc(compute, environment);
        malloc(compute, validation_world);
        Scenes::Corpus corpus;
        enumerate(host, scenes, corpus);
        for(TI member = 0; member < N_ACTIVE_SCENES; member++){
            auto& world = environment.environments[member];
            world.parameters.agents[0].cameras[0].fov = CAMERA_FOV;
            rl_tools::init(compute, world, environment.shared, scenes, corpus, 0, N_TRAIN_SCENES, member);
            world.rng_offset = member * N_ENVIRONMENTS_PER_SCENE;
        }
        validation_world.parameters.agents[0].cameras[0].fov = CAMERA_FOV;
        rl_tools::init(compute, validation_world, environment.shared, scenes, corpus, N_TRAIN_SCENES, N_VALIDATION_SCENES, N_ACTIVE_SCENES);
        yaw_manifest(host, scenes, environment, validation_world, paths, seed);
        Tensor<typename BASE_WORLD::DRONE_PARAMETERS_SPEC> nominal_host;
        malloc(host, nominal_host);
        copy(compute, host, environment.environments[0].drone_parameters, nominal_host);
        ValidationState validation;
        malloc(host, compute, validation);
        init(host, compute, validation, seed);

        DATASET dataset;
        Tensor<tensor::Specification<T_ACTIVATION, TI, tensor::Shape<TI, STEPS_TOTAL, TARGET_DIM>>> targets;
        Tensor<tensor::Specification<T_ACTIVATION, TI, tensor::Shape<TI, 1, BATCH_SIZE, TARGET_DIM>>> output;
        Matrix<matrix::Specification<T_GRADIENT, TI, BATCH_SIZE, TARGET_DIM>> d_output;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1, BATCH_SIZE, TARGET_DIM>>> output_host;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, BATCH_SIZE, TARGET_DIM>>> targets_host;
        Matrix<matrix::Specification<T, TI, STEPS_TOTAL, 1>> reset_host;
        malloc(compute, dataset);
        malloc(compute, targets);
        malloc(compute, output);
        malloc(compute, d_output);
        malloc(host, output_host);
        malloc(host, targets_host);
        malloc(host, reset_host);

        T orientation_bound = INIT_ORIENTATION_CURRICULUM_START_DEG;
        TI global_step = 0;
        for(TI epoch = 0; epoch < NUM_EPOCHS; epoch++){
            const auto epoch_start = std::chrono::steady_clock::now();
            std::shuffle(scene_permutation.begin(), scene_permutation.end(), scene_rng);
            std::array<TI, N_ACTIVE_SCENES> selected;
            for(TI member = 0; member < N_ACTIVE_SCENES; member++){
                selected[member] = scene_permutation[member];
            }
            get_ref(host, nominal_host, 0).dynamics.mdp.init.max_angle = orientation_bound / (T)180 * math::PI<T>;
            for(TI member = 0; member < N_ACTIVE_SCENES; member++){
                auto& world = environment.environments[member];
                copy(host, compute, nominal_host, world.drone_parameters);
                select_scene(compute, world, selected[member]);
            }
            if(epoch == 0){
                init(compute, runner, environment, compute_rng);
            } else {
                reset(compute, runner, environment, compute_rng);
            }
            std::array<TI, N_ACTIVE_SCENES> bank_steps;
            for(TI member = 0; member < N_ACTIVE_SCENES; member++){
                bank_steps[member] = environment.environments[member].bank_step;
            }
            prologue(compute, dataset, runner, environment, compute_rng);
            for(TI time = 0; time < STEPS_PER_ENV; time++){
                auto mask = matrix_view(compute, runner.reset);
                Mode<mode::sequential::ResetMask<mode::Default<>, mode::sequential::ResetMaskSpecification<decltype(mask)>>> reset_mode;
                reset_mode.mask = mask;
                reset(compute, teacher, runner.policy_state, compute_rng, reset_mode);
                auto teacher_observation = view_range(compute, dataset.all_observations_privileged, time * N_ENVIRONMENTS, tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                evaluate_step(compute, teacher, teacher_observation, runner.policy_state, runner_buffer.actions, teacher_buffer, compute_rng, Mode<nn::layers::gru::NoAutoResetMode<mode::Default<>>>{});
                auto target_row = view_range(compute, targets, time * N_ENVIRONMENTS, tensor::ViewSpec<0, N_ENVIRONMENTS>{});
                yaw_targets(compute, runner.states, target_row);
                epilogue(compute, dataset, runner, runner_buffer, environment, compute_rng, time);
                global_step += N_ENVIRONMENTS;
            }
            for(TI member = 0; member < N_ACTIVE_SCENES; member++){
                if(environment.environments[member].bank_step != bank_steps[member]){
                    throw std::runtime_error("Yaw target bank refreshed inside an epoch");
                }
            }
            yaw_synchronize(compute);
            const auto collection_end = std::chrono::steady_clock::now();
            copy(compute, host, dataset.reset, reset_host);
            TI early_resets = 0;
            for(TI row = N_ENVIRONMENTS; row < STEPS_TOTAL; row++){
                early_resets += get(reset_host, row, 0) != 0;
            }

            auto fixed_batch_loss = [&]{
                auto image_batch = view_range(compute, dataset.all_observations, (TI)0, tensor::ViewSpec<0, BATCH_SIZE>{});
                auto images = reshape_row_major(compute, image_batch, tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>{});
                auto inputs = nn_models::parallel::pack_inputs(images);
                auto target_batch = view_range(compute, targets, (TI)0, tensor::ViewSpec<0, BATCH_SIZE>{});
                forward(compute, student, inputs, output, student_buffer, compute_rng);
                copy(compute, host, output, output_host);
                copy(compute, host, target_batch, targets_host);
                return yaw_batch_mse(host, output_host, targets_host);
            };
            T fixed_loss_before = 0;
            if constexpr(SMOKE){
                fixed_loss_before = fixed_batch_loss();
            }
            T loss_sum = 0;
            T yaw_squared_error = 0;
            T yaw_null_squared_error = 0;
            for(TI pass = 0; pass < N_TRAIN_PASSES; pass++){
                std::array<TI, N_BATCHES> order;
                for(TI batch = 0; batch < N_BATCHES; batch++){
                    order[batch] = batch;
                }
                for(TI i = N_BATCHES - 1; i > 0; i--){
                    const TI j = random::uniform_int_distribution(host.random, (TI)0, i, rng);
                    std::swap(order[i], order[j]);
                }
                for(const TI batch: order){
                    const TI offset = batch * BATCH_SIZE;
                    auto image_batch = view_range(compute, dataset.all_observations, offset, tensor::ViewSpec<0, BATCH_SIZE>{});
                    auto images = reshape_row_major(compute, image_batch, tensor::Shape<TI, 1, BATCH_SIZE, IMG_H, IMG_W, COMBINED_IMG_C>{});
                    auto inputs = nn_models::parallel::pack_inputs(images);
                    auto target_batch = view_range(compute, targets, offset, tensor::ViewSpec<0, BATCH_SIZE>{});
                    zero_gradient(compute, student);
                    forward(compute, student, inputs, output, student_buffer, compute_rng);
                    auto prediction_matrix = matrix_view(compute, output);
                    auto target_matrix = matrix_view(compute, target_batch);
                    nn::loss_functions::mse::gradient(compute, prediction_matrix, target_matrix, d_output, (T)0.5);
                    if(pass == 0){
                        copy(compute, host, output, output_host);
                        copy(compute, host, target_batch, targets_host);
                        loss_sum += yaw_batch_mse(host, output_host, targets_host) / (T)N_BATCHES;
                        for(TI sample = 0; sample < BATCH_SIZE; sample++){
                            const T cosine = get(host, targets_host, sample, 0);
                            const T sine = get(host, targets_host, sample, 1);
                            const T error = yaw_error(host, get(host, output_host, 0, sample, 0), get(host, output_host, 0, sample, 1), cosine, sine);
                            const T null_error = yaw_error(host, (T)1, (T)0, cosine, sine);
                            yaw_squared_error += error * error / (T)(BATCH_SIZE * N_BATCHES);
                            yaw_null_squared_error += null_error * null_error / (T)(BATCH_SIZE * N_BATCHES);
                        }
                    }
                    auto gradient_tensor = to_tensor(compute, d_output);
                    auto gradient = reshape_row_major(compute, gradient_tensor, tensor::Shape<TI, 1, BATCH_SIZE, TARGET_DIM>{});
                    backward(compute, student, inputs, gradient, student_buffer);
                    step(compute, optimizer, student);
                }
            }
            yaw_synchronize(compute);
            const auto training_end = std::chrono::steady_clock::now();
            if(!std::isfinite(loss_sum)){
                throw std::runtime_error("Non-finite training loss");
            }
            nlohmann::json record = {{"epoch", epoch}, {"step", global_step}, {"scenes", selected}, {"updates", N_TRAIN_PASSES * N_BATCHES}, {"mse", loss_sum}, {"orientation_bound_deg", orientation_bound}, {"early_resets", early_resets}, {"reused_setup_fraction", (T)early_resets / (T)(N_ENVIRONMENTS + early_resets)}, {"collection_seconds", std::chrono::duration<double>(collection_end - epoch_start).count()}, {"training_seconds", std::chrono::duration<double>(training_end - collection_end).count()}};
            if constexpr(SMOKE){
                const T fixed_loss_after = fixed_batch_loss();
                record["fixed_batch_mse_before"] = fixed_loss_before;
                record["fixed_batch_mse_after"] = fixed_loss_after;
                if(epoch == 0 && fixed_loss_after >= fixed_loss_before){
                    throw std::runtime_error("Smoke training did not reduce fixed-batch loss");
                }
            }
            metra::log("hyperdrone/yaw/training/mse", loss_sum);
            record["yaw_rmse_deg"] = std::sqrt(yaw_squared_error) * (T)180 / math::PI<T>;
            record["label_yaw_rms_deg"] = std::sqrt(yaw_null_squared_error) * (T)180 / math::PI<T>;
            metra::log("hyperdrone/yaw/training/rmse_deg", record["yaw_rmse_deg"].get<double>());
            metra::log("hyperdrone/yaw/training/reused_setup_fraction", record["reused_setup_fraction"].get<double>());
            metra::log("hyperdrone/yaw/training/collection_seconds", record["collection_seconds"].get<double>());
            metra::log("hyperdrone/yaw/training/training_seconds", record["training_seconds"].get<double>());
            set_step(host, host.logger, global_step);
            add_scalar(host, host.logger, "yaw/mse", loss_sum);
            if(epoch % VALIDATION_CADENCE == 0){
                const T frontier_min = std::min(orientation_bound, INIT_ORIENTATION_CURRICULUM_FULL_DEG - INIT_ORIENTATION_CURRICULUM_STEP_DEG);
                const T frontier_max = frontier_min + INIT_ORIENTATION_CURRICULUM_STEP_DEG;
                const T threshold = yaw_curriculum_threshold(host, orientation_bound, INIT_ORIENTATION_CURRICULUM_FRONTIER_RATIO, INIT_ORIENTATION_CURRICULUM_FRONTIER_MIN_DEG, INIT_ORIENTATION_CURRICULUM_FRONTIER_MAX_DEG);
                const auto result = validate(host, compute, validation_world, student, validation, frontier_min, frontier_max);
                const T frontier_rmse_deg = std::sqrt(result.frontier_yaw_mse_rad) / math::PI<T> * (T)180;
                const T fixed_rmse_deg = std::sqrt(result.yaw_mse_rad) / math::PI<T> * (T)180;
                record["validation"] = {{"mse", result.yaw_mse_loss}, {"rmse_deg", fixed_rmse_deg}, {"mae_rad", result.yaw_abs_error_rad}, {"output_norm", result.yaw_output_norm}, {"null_mse", result.yaw_null_mse_loss}, {"null_mse_rad", result.yaw_null_mse_rad}, {"r2", result.yaw_angle_r2_uniform}, {"bin_mse_rad", result.bin_yaw_mse_rad}, {"frontier_deg", {frontier_min, frontier_max}}, {"frontier_rmse_deg", frontier_rmse_deg}, {"threshold_deg", threshold}};
                orientation_bound = yaw_curriculum_advance(host, orientation_bound, INIT_ORIENTATION_CURRICULUM_FULL_DEG, INIT_ORIENTATION_CURRICULUM_STEP_DEG, frontier_rmse_deg, threshold);
                metra::log("hyperdrone/yaw/validation/rmse_deg", fixed_rmse_deg);
                metra::log("hyperdrone/yaw/validation/frontier_rmse_deg", frontier_rmse_deg);
                metra::log("hyperdrone/yaw/validation/r2", result.yaw_angle_r2_uniform);
                metra::log("hyperdrone/yaw/curriculum/orientation_deg", orientation_bound);
                add_scalar(host, host.logger, "yaw/validation_rmse_deg", fixed_rmse_deg);
                add_scalar(host, host.logger, "yaw/frontier_rmse_deg", frontier_rmse_deg);
                add_scalar(host, host.logger, "yaw/orientation_bound_deg", orientation_bound);
            }
            record["next_orientation_bound_deg"] = orientation_bound;
            if(epoch % CHECKPOINT_CADENCE == 0 || epoch + 1 == NUM_EPOCHS){
                const auto directory = get_step_folder(host, extrack_config, paths, global_step);
                const T reload_error = save_checkpoint(host, compute, student, dataset.all_observations, directory);
                record["checkpoint_reload_max_error"] = reload_error;
                metra::log("hyperdrone/yaw/checkpoint/reload_max_error", reload_error);
                if(reload_error > (T)1e-4){
                    throw std::runtime_error("Yaw checkpoint reload predictions differ");
                }
            }
            std::size_t free_bytes, total_bytes;
            const auto memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
            if(memory_status != cudaSuccess){
                throw std::runtime_error(cudaGetErrorString(memory_status));
            }
            record["device_used_bytes"] = total_bytes - free_bytes;
            record["epoch_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
            metrics << record.dump() << std::endl;
            std::cout << "Epoch " << epoch << " step " << global_step << " MSE " << loss_sum << " orientation " << orientation_bound << " degrees, early resets " << early_resets << std::endl;
        }

        free(host, compute, validation);
        rl_tools::free(compute, validation_world);
        free(compute, environment);
        free(host, nominal_host);
        free(compute, runner);
        free(compute, runner_buffer);
        free(compute, dataset);
        free(compute, targets);
        free(compute, output);
        free(compute, d_output);
        free(host, output_host);
        free(host, targets_host);
        free(host, reset_host);
        free(compute, student);
        free(compute, student_buffer);
        free(compute, optimizer);
        free(host, teacher_host);
        free(compute, teacher);
        free(compute, teacher_buffer);
        free(host, rng);
        free(compute, compute_rng);
        free(compute);
#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
        free(host, host.logger);
#endif
        std::cout << "Training finished: " << paths.seed << std::endl;
        return 0;
    }
}

int main(int argc, char** argv){
    if(argc < 2 || argc > 3){
        std::cerr << "Usage: " << argv[0] << " <scene_directory> [seed] (smoke also accepts one .glb)" << std::endl;
        return 1;
    }
    try{
        using TI = rl_tools::yaw_training::TI;
        TI seed = 0;
        if(argc == 3){
            const std::string argument = argv[2];
            std::size_t parsed = 0;
            const auto value = std::stoull(argument, &parsed);
            seed = static_cast<TI>(value);
            if(argument.empty() || argument.front() == '-' || parsed != argument.size() || static_cast<unsigned long long>(seed) != value){
                throw std::runtime_error("Seed must be a nonnegative integer");
            }
        }
        return rl_tools::train_yaw(rl_tools::yaw_scenes(argv[1]), seed);
    } catch(const std::exception& error){
        std::cerr << "Yaw training failed: " << error.what() << std::endl;
        return 1;
    }
}
