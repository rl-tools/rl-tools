#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/autopilot.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/metrics.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/baseline.h>

#include "../../../../nn_models/port_checkpoint/raptor/policy.h"

#include "../demo_common.h"

#include <metra/metra.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace task = rlt::rl::environments::hyperdrone::tasks::visual_inertial_localization;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

struct WORLD_SPEC: rlt::rl::environments::hyperdrone::presets::X500FPVIMU<T, TI> {
    static constexpr TI INSTANCES_PER_ENVIRONMENT = 2;
    static constexpr bool SELF_VISIBLE = false;
};
using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
using TASK_SPEC = task::Specification<BASE_WORLD>;
using WORLD = task::World<TASK_SPEC>;

constexpr TI INSTANCES = WORLD::INSTANCES;
constexpr TI OBSERVATION_DIM = BASE_WORLD::OBSERVATION_DIM;
constexpr TI CAM_WIDTH = WORLD_SPEC::CAM_WIDTH;
constexpr TI CAM_HEIGHT = WORLD_SPEC::CAM_HEIGHT;
constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
constexpr TI FRAME_STRIDE = WORLD::FRAME_STRIDE;
constexpr TI IMU_RATE = WORLD_SPEC::DYNAMICS_STATIC_PARAMETERS::IMU_RATE;
constexpr TI STEPS = WORLD_SPEC::DYNAMICS_STATIC_PARAMETERS::EPISODE_STEP_LIMIT;
constexpr TI FRAMERATE = IMU_RATE / FRAME_STRIDE;
constexpr TI UPSCALE = 8;

static_assert(BASE_WORLD::IMAGE_CHANNELS == 3);

constexpr TI RAPTOR_INPUT_DIM = rl_tools::checkpoint::actor::layer_0::INPUT_SHAPE::LAST;
constexpr TI RAPTOR_ACTION_DIM = rl_tools::checkpoint::actor::TYPE::OUTPUT_SHAPE::LAST;
using RAPTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, INSTANCES, RAPTOR_INPUT_DIM>;
using RAPTOR_CAPABILITY = rlt::nn::capability::Forward<true, false>;
using RAPTOR_MODEL = rlt::nn_models::sequential::Build<RAPTOR_CAPABILITY, rl_tools::checkpoint::actor::TEMPLATE, RAPTOR_INPUT_SHAPE>;
using AUTOPILOT = task::Autopilot<WORLD, RAPTOR_MODEL>;
static_assert(RAPTOR_INPUT_DIM == AUTOPILOT::OBSERVATION_DIM);
static_assert(RAPTOR_ACTION_DIM == WORLD::ACTION_DIM);

struct Tensors {
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, OBSERVATION_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ObservationIMU::DIM>>> observations_imu;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> terminated_flags;
    void allocate(DEVICE& device){
        rlt::malloc(device, parameters);
        rlt::malloc(device, states);
        rlt::malloc(device, next_states);
        rlt::malloc(device, reset_mask);
        rlt::malloc(device, actions);
        rlt::malloc(device, observations);
        rlt::malloc(device, observations_imu);
        rlt::malloc(device, terminated_flags);
    }
    void deallocate(DEVICE& device){
        rlt::free(device, parameters);
        rlt::free(device, states);
        rlt::free(device, next_states);
        rlt::free(device, reset_mask);
        rlt::free(device, actions);
        rlt::free(device, observations);
        rlt::free(device, observations_imu);
        rlt::free(device, terminated_flags);
    }
};

int main(int argc, char** argv){
#ifdef RL_TOOLS_TEST_DATA_PATH
    std::string scene_path = std::string(HYPERDRONE_DEMO_STRINGIFY(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
    std::string scene_path = "";
#endif
    std::string output_path = "visual_inertial_localization.mp4";
    TI seed = 0;
    if(argc > 1){
        scene_path = argv[1];
    }
    if(argc > 2){
        output_path = argv[2];
    }
    if(argc > 3){
        seed = std::stoul(argv[3]);
    }
    if(scene_path.empty()){
        std::fprintf(stderr, "usage: %s <scene.glb> [output.mp4] [seed]\n", argv[0]);
        return 1;
    }

    DEVICE device;
    rlt::init(device);
    WORLD world;
    typename BASE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {scene_path}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world, shared, dataset, corpus, 0, 1, 0);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);

    AUTOPILOT autopilot;
    task::init(device, autopilot, rl_tools::checkpoint::actor::module, rng);

    Tensors tensors;
    tensors.allocate(device);
    rlt::set_all(device, tensors.reset_mask, true);
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);

    const typename WORLD::Parameters reference_parameters = rlt::get(device, tensors.parameters, (TI)0);
    const T dt = reference_parameters.dynamics.integration.dt;

    // benchmark protocol: the estimator receives NO ground truth — episodes start level and at
    // rest (preset init), the estimate is anchored at identity by convention, and scoring
    // compares against the pose relative to the initial pose. The origin poses live on the
    // harness side for evaluation only; the oracle is fed ground truth and must score exactly
    // zero (metric-pipeline sanity)
    task::DeadReckoningState<T> estimators[INSTANCES];
    T origin_positions[INSTANCES][3];
    T origin_orientations[INSTANCES][4];
    task::TrajectoryMetricsAccumulator<T, TI> metrics[INSTANCES];
    task::TrajectoryMetricsAccumulator<T, TI> oracle_metrics[INSTANCES];
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename WORLD::State state = rlt::get(device, tensors.states, instance_i);
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            estimators[instance_i].position[dim_i] = 0;
            estimators[instance_i].linear_velocity[dim_i] = 0;
            origin_positions[instance_i][dim_i] = state.position[dim_i];
        }
        estimators[instance_i].orientation[0] = 1;
        for(TI dim_i = 1; dim_i < 4; dim_i++){
            estimators[instance_i].orientation[dim_i] = 0;
        }
        for(TI dim_i = 0; dim_i < 4; dim_i++){
            origin_orientations[instance_i][dim_i] = state.orientation[dim_i];
        }
        task::reset(device, metrics[instance_i]);
        task::reset(device, oracle_metrics[instance_i]);
    }

    FILE* video_pipe = hyperdrone_demo::open_video_pipe(CAM_WIDTH, CAM_HEIGHT * INSTANCES, FRAMERATE, UPSCALE, output_path);
    if(video_pipe == nullptr){
        std::fprintf(stderr, "failed to open ffmpeg pipe for %s\n", output_path.c_str());
        return 1;
    }
    std::vector<std::uint8_t> frame(INSTANCES * CAM_PIXELS * 3);

    TI frames_written = 0;
    const auto episode_start_time = std::chrono::steady_clock::now();
    for(TI step_i = 0; step_i < STEPS; step_i++){
        rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
        rlt::set_all(device, tensors.reset_mask, false);
        if(step_i % FRAME_STRIDE == 0){
            rlt::observe(device, world, tensors.parameters, tensors.states, typename BASE_WORLD::Observation{}, tensors.observations, rng);
            for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
                for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
                    for(TI channel_i = 0; channel_i < 3; channel_i++){
                        frame[(instance_i * CAM_PIXELS + pixel_i) * 3 + channel_i] = hyperdrone_demo::quantize_pixel(rlt::get(device, tensors.observations, instance_i, pixel_i * 3 + channel_i));
                    }
                }
            }
            std::fwrite(frame.data(), 1, frame.size(), video_pipe);
            frames_written++;
        }
        task::control(device, world, tensors.parameters, tensors.states, autopilot, tensors.actions, rng);
        rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
        rlt::observe(device, world, tensors.parameters, tensors.next_states, typename WORLD::ObservationIMU{}, tensors.observations_imu, rng);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            T accelerometer[3], gyroscope[3];
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                accelerometer[dim_i] = rlt::get(device, tensors.observations_imu, instance_i, dim_i);
                gyroscope[dim_i] = rlt::get(device, tensors.observations_imu, instance_i, 3 + dim_i);
            }
            task::dead_reckoning_step(device, estimators[instance_i], accelerometer, gyroscope, reference_parameters.dynamics.dynamics.gravity, dt);
            typename WORLD::State next_state = rlt::get(device, tensors.next_states, instance_i);
            T relative_position[3], relative_orientation[4];
            task::relative_pose(device, origin_positions[instance_i], origin_orientations[instance_i], next_state.position, next_state.orientation, relative_position, relative_orientation);
            task::accumulate(device, metrics[instance_i], relative_position, relative_orientation, estimators[instance_i].position, estimators[instance_i].orientation);
            task::accumulate(device, oracle_metrics[instance_i], relative_position, relative_orientation, relative_position, relative_orientation);
        }
        rlt::copy(device, device, tensors.next_states, tensors.states);
    }
    const double episode_wall_clock_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - episode_start_time).count();
    rlt::terminated(device, world, tensors.parameters, tensors.states, tensors.terminated_flags, rng);

    bool video_ok = hyperdrone_demo::close_video_pipe(video_pipe);
    if(!video_ok){
        std::fprintf(stderr, "ffmpeg exited with an error for %s\n", output_path.c_str());
    }
    else{
        std::printf("wrote %s (%lu frames at %lu fps, %lu IMU samples at %lu Hz)\n", output_path.c_str(), (unsigned long)frames_written, (unsigned long)FRAMERATE, (unsigned long)STEPS, (unsigned long)IMU_RATE);
    }
    std::vector<double> ate_position_rmse_values, rotation_error_mean_values, rotation_error_max_values, drift_per_distance_values, distance_traveled_values, oracle_ate_values;
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        std::printf("instance %lu (terminated: %s):\n", (unsigned long)instance_i, rlt::get(device, tensors.terminated_flags, instance_i) ? "yes" : "no");
        std::printf("  oracle          ATE %.6f m (must be 0)\n", (double)task::ate_position_rmse(device, oracle_metrics[instance_i]));
        std::printf("  dead reckoning  ATE %.3f m | rot mean %.4f rad max %.4f rad | drift %.4f m/m | distance %.2f m\n",
            (double)task::ate_position_rmse(device, metrics[instance_i]),
            (double)task::mean_rotation_error(device, metrics[instance_i]),
            (double)task::max_rotation_error(device, metrics[instance_i]),
            (double)task::drift_per_distance(device, metrics[instance_i]),
            (double)metrics[instance_i].distance_traveled);
        ate_position_rmse_values.push_back((double)task::ate_position_rmse(device, metrics[instance_i]));
        rotation_error_mean_values.push_back((double)task::mean_rotation_error(device, metrics[instance_i]));
        rotation_error_max_values.push_back((double)task::max_rotation_error(device, metrics[instance_i]));
        drift_per_distance_values.push_back((double)task::drift_per_distance(device, metrics[instance_i]));
        distance_traveled_values.push_back((double)metrics[instance_i].distance_traveled);
        oracle_ate_values.push_back((double)task::ate_position_rmse(device, oracle_metrics[instance_i]));
    }
    const std::string metra_prefix = "hyperdrone/visual_inertial_localization";
    metra::log(metra_prefix + "/dead_reckoning/ate_position_rmse", ate_position_rmse_values);
    metra::log(metra_prefix + "/dead_reckoning/rotation_error_mean", rotation_error_mean_values);
    metra::log(metra_prefix + "/dead_reckoning/rotation_error_max", rotation_error_max_values);
    metra::log(metra_prefix + "/dead_reckoning/drift_per_distance", drift_per_distance_values);
    metra::log(metra_prefix + "/distance_traveled", distance_traveled_values);
    metra::log(metra_prefix + "/oracle/ate_position_rmse", oracle_ate_values);
    metra::log(metra_prefix + "/episode/wall_clock_time", episode_wall_clock_time);
    metra::log(metra_prefix + "/episode/steps_per_second", (double)(STEPS * INSTANCES) / episode_wall_clock_time);

    tensors.deallocate(device);
    task::free(device, autopilot);
    rlt::free(device, world);
    rlt::free(device, shared.library);
    rlt::free(device, rng);
    return video_ok ? 0 : 1;
}
