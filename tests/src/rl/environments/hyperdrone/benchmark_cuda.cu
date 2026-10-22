// Phase-2 throughput gate: the World's render+observe verbs vs the hand-rolled
// training_cuda.cu-style per-step path (pose kernel on the main stream, event, per-renderer
// camera D2D staging, render_launch, scatter) at the matched configuration
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cuda.h>

#include "../../../utils/utils.h"

#include <metra/metra.h>

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <sys/resource.h>

namespace rlt = rl_tools;
namespace rro = rlt::hyperlux::outputs;
using rlt::output;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_GPU_SPEC = rlt::hyperlux::device::Specification<rlt::devices::DefaultCUDASpecification, DEVICE>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<DEVICE_GPU_SPEC>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string DATA_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH));
#else
static const std::string DATA_PATH = "";
#endif
static const std::string SCENE_PATH = DATA_PATH.empty() ? "" : DATA_PATH + "/ProcTHOR-Train-1.glb";

namespace benchmark_hyperdrone {
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI EPISODE_STEP_LIMIT = 10000;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;
    struct DYNAMICS_STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
        using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
                l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
                l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
                l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = PARAMETERS_TYPE;
        static constexpr auto dynamics = l2f::parameters::dynamics::registry<l2f::parameters::dynamics::REGISTRY::crazyflie, PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {(T)0.01};
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {init, REWARD_FUNCTION{}, {}, {}, {}};
        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
        static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {{dynamics, integration, mdp}, disturbances};
        static constexpr T STATE_LIMIT_POSITION_X = 100000;
        static constexpr T STATE_LIMIT_POSITION_Y = 100000;
        static constexpr T STATE_LIMIT_POSITION_Z = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
    };

    // training_cuda.cu's configuration: 2 active scenes x 64 envs, 80x50, High shading, AA 2
    struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, DYNAMICS_STATIC_PARAMETERS> {
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI INSTANCES_PER_WORLD = 64;
        static constexpr TI CAM_WIDTH = 80;
        static constexpr TI CAM_HEIGHT = 50;
        using SHADING = rlt::hyperlux::High;
        static constexpr bool ENABLE_ANTI_ALIASING = true;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = 2;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.5;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    constexpr TI N_WORLDS = 2;
    using ENVIRONMENT = rlt::rl::environments::HyperDrone<WORLD, N_WORLDS>;
    constexpr TI INSTANCES = ENVIRONMENT::INSTANCES;
    constexpr TI WARMUP_STEPS = 5;
    constexpr TI MEASURE_STEPS = 50;
}

// the same configuration with the drones rendered into their own cameras, motion blurred: the
// device drone rows (per-instance bodies from the drone table) and the dynamic overlay rebuild
namespace benchmark_self_visible {
    using PRESET = rlt::rl::environments::hyperdrone::presets::X500FPV<T, TI>;
    using PARAMETERS_SPEC = typename PRESET::DYNAMICS_STATIC_PARAMETERS::PARAMETERS_SPEC;
    struct WORLD_SPEC: PRESET {
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI INSTANCES_PER_WORLD = 64;
        static constexpr TI CAM_WIDTH = 80;
        static constexpr TI CAM_HEIGHT = 50;
        using SHADING = rlt::hyperlux::High;
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr TI MOTION_BLUR_SAMPLES = 4;
        static constexpr TI MAX_DRONES = 4;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    using ENVIRONMENT = rlt::rl::environments::HyperDrone<WORLD, benchmark_hyperdrone::N_WORLDS>;
    template <l2f::parameters::dynamics::REGISTRY MODEL>
    rlt::rl::environments::hyperdrone::Drone<WORLD_SPEC> drone(const char* asset){
        rlt::rl::environments::hyperdrone::Drone<WORLD_SPEC> entry;
        entry.asset = DATA_PATH + "/" + asset;
        entry.parameters.dynamics.dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
        return entry;
    }
}

using namespace benchmark_hyperdrone;

static bool cuda_available(){
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_BENCHMARK, WORLD_VS_HAND_ROLLED){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    if(!cuda_available()){
        GTEST_SKIP() << "CUDA device unavailable";
    }
    DEVICE_GPU device_gpu;
    auto& device = device_gpu.rendering;
    rlt::init(device);
    rlt::init(device_gpu);

    std::size_t free_before, total_bytes;
    ASSERT_EQ(cudaMemGetInfo(&free_before, &total_bytes), cudaSuccess);
    ENVIRONMENT env;
    rlt::malloc(device_gpu, env);
    rlt::hyperlux::datasets::procthor::GLB dataset{{}, {SCENE_PATH, SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::hyperlux::datasets::procthor::enumerate(device, dataset, corpus);
    for(TI world_i = 0; world_i < N_WORLDS; world_i++){
        rlt::init(device_gpu, env.worlds[world_i], env.shared, dataset, corpus, world_i, 1, world_i);
    }

    RNG_GPU rng;
    rlt::malloc(device_gpu, rng);
    rlt::init(device_gpu, rng, 0);
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM>>> observations;
    rlt::malloc(device_gpu, parameters);
    rlt::malloc(device_gpu, states);
    rlt::malloc(device_gpu, reset_mask);
    rlt::malloc(device_gpu, observations);
    rlt::set_all(device_gpu, reset_mask, true);
    rlt::sample_initial_parameters(device_gpu, env, parameters, reset_mask, rng);
    rlt::sample_initial_state(device_gpu, env, parameters, states, reset_mask, rng);
    rlt::set_all(device_gpu, reset_mask, false);

    auto request_frame = [&](){
        for(auto& world: env.worlds){
            using rlt::request_render;
            request_render(device_gpu, world);
        }
    };
    // World verb path
    auto world_path = [&](){
        request_frame();
        rlt::render(device_gpu, env, parameters, states, reset_mask);
        rlt::observe(device_gpu, env, parameters, states, typename WORLD::Observation{}, observations, rng);
    };
    // hand-rolled training_cuda.cu-style path over the same renderers: pose kernel on the main
    // stream into a staging buffer, event, per-renderer camera D2D + render_launch + scatter into
    // the frame history, then the observation assembly kernel (training_cuda scatters into
    // gpu_frame_stack_history and assembles the policy input from it)
    rlt::hyperlux::Camera<T>* staging_cameras;
    cudaMalloc(&staging_cameras, INSTANCES * sizeof(rlt::hyperlux::Camera<T>));
    cudaEvent_t cameras_ready;
    cudaEventCreateWithFlags(&cameras_ready, cudaEventDisableTiming);
    float* hand_rolled_history;
    cudaMalloc(&hand_rolled_history, INSTANCES * WORLD::OBSERVATION_DIM * sizeof(float));
    float* hand_rolled_observations;
    cudaMalloc(&hand_rolled_observations, INSTANCES * WORLD::OBSERVATION_DIM * sizeof(float));
    // training_cuda joins the render streams back into the main stream before the policy
    // evaluation consumes the observations (render_scatter_done events)
    cudaEvent_t render_scatter_done[N_WORLDS];
    for(TI world_i = 0; world_i < N_WORLDS; world_i++){
        cudaEventCreateWithFlags(&render_scatter_done[world_i], cudaEventDisableTiming);
    }
    auto hand_rolled_path = [&](){
        using TAG = rlt::devices::cuda::TAG<DEVICE_GPU, true>;
        TAG tag_device{};
        constexpr TI BLOCKSIZE = 32;
        for(TI world_i = 0; world_i < N_WORLDS; world_i++){
            auto& world = env.worlds[world_i];
            constexpr TI M = WORLD::INSTANCES;
            auto parameters_block = rlt::view_range(device_gpu, parameters, world_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto states_block = rlt::view_range(device_gpu, states, world_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto reset_block = rlt::view_range(device_gpu, reset_mask, world_i * M, rlt::tensor::ViewSpec<0, M>{});
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(M, BLOCKSIZE);
            rlt::rl::environments::hyperdrone::cuda::pose_kernel<TAG, WORLD_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device_gpu.stream>>>(tag_device, parameters_block, states_block, reset_block, staging_cameras + world_i * M, staging_cameras + world_i * M, rlt::data(world.prev_cameras), world.history_step, (T)WORLD_SPEC::CAM_WIDTH / (T)WORLD_SPEC::CAM_HEIGHT);
        }
        cudaEventRecord(cameras_ready, device_gpu.stream);
        for(TI world_i = 0; world_i < N_WORLDS; world_i++){
            auto& world = env.worlds[world_i];
            constexpr TI M = WORLD::INSTANCES;
            auto& renderer = rlt::get_renderer(device_gpu, world);
            cudaStream_t render_stream = rlt::stream(device_gpu, renderer);
            cudaStreamWaitEvent(render_stream, cameras_ready, 0);
            cudaMemcpyAsync(rlt::data(rlt::cameras(device_gpu, renderer)), staging_cameras + world_i * M, M * sizeof(rlt::hyperlux::Camera<T>), cudaMemcpyDeviceToDevice, render_stream);
            rlt::render_launch(device_gpu, renderer);
            auto parameters_block = rlt::view_range(device_gpu, parameters, world_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto reset_block = rlt::view_range(device_gpu, reset_mask, world_i * M, rlt::tensor::ViewSpec<0, M>{});
            constexpr TI SCATTER_BLOCKSIZE = 256;
            constexpr TI SCATTER_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(M * WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT, SCATTER_BLOCKSIZE);
            rlt::rl::environments::hyperdrone::cuda::scatter_kernel<TAG, WORLD_SPEC><<<dim3(SCATTER_BLOCKS), dim3(SCATTER_BLOCKSIZE), 0, render_stream>>>(tag_device, rlt::data(output(device, renderer, rro::RGB<rro::format::Float32>{})), hand_rolled_history + world_i * M * WORLD::OBSERVATION_DIM, parameters_block, rlt::data(world.episode_start), reset_block, world.history_step);
            auto observations_block = rlt::view_range(device_gpu, observations, world_i * M, rlt::tensor::ViewSpec<0, M>{});
            constexpr TI ASSEMBLY_BLOCKSIZE = 256;
            constexpr TI ASSEMBLY_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(M * WORLD::FRAME_DIM, ASSEMBLY_BLOCKSIZE);
            rlt::rl::environments::hyperdrone::cuda::observe_kernel<TAG, WORLD_SPEC><<<dim3(ASSEMBLY_BLOCKS), dim3(ASSEMBLY_BLOCKSIZE), 0, render_stream>>>(tag_device, hand_rolled_history + world_i * M * WORLD::OBSERVATION_DIM, observations_block);
            cudaEventRecord(render_scatter_done[world_i], render_stream);
            cudaStreamWaitEvent(device_gpu.stream, render_scatter_done[world_i], 0);
        }
    };

    auto measure = [&](auto&& body, const char* label, const char* metric){
        for(TI step_i = 0; step_i < WARMUP_STEPS; step_i++){
            body();
        }
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(TI step_i = 0; step_i < MEASURE_STEPS; step_i++){
            body();
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        double steps_per_second = MEASURE_STEPS / seconds;
        std::cout << label << ": " << steps_per_second << " frames/s (" << seconds * 1000 / MEASURE_STEPS << " ms/frame, " << INSTANCES << " instances)" << std::endl;
        metra::log(std::string("hyperdrone/benchmark_cuda/") + metric + "/frames_per_second", steps_per_second);
        return steps_per_second;
    };

    auto world_render_only = [&](){
        request_frame();
        rlt::render(device_gpu, env, parameters, states, reset_mask);
    };
    auto hand_rolled_render_only = [&](){
        using TAG = rlt::devices::cuda::TAG<DEVICE_GPU, true>;
        TAG tag_device{};
        constexpr TI BLOCKSIZE = 32;
        for(TI world_i = 0; world_i < N_WORLDS; world_i++){
            auto& world = env.worlds[world_i];
            constexpr TI M = WORLD::INSTANCES;
            auto parameters_block = rlt::view_range(device_gpu, parameters, world_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto states_block = rlt::view_range(device_gpu, states, world_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto reset_block = rlt::view_range(device_gpu, reset_mask, world_i * M, rlt::tensor::ViewSpec<0, M>{});
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(M, BLOCKSIZE);
            rlt::rl::environments::hyperdrone::cuda::pose_kernel<TAG, WORLD_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device_gpu.stream>>>(tag_device, parameters_block, states_block, reset_block, staging_cameras + world_i * M, staging_cameras + world_i * M, rlt::data(world.prev_cameras), world.history_step, (T)WORLD_SPEC::CAM_WIDTH / (T)WORLD_SPEC::CAM_HEIGHT);
        }
        cudaEventRecord(cameras_ready, device_gpu.stream);
        for(TI world_i = 0; world_i < N_WORLDS; world_i++){
            auto& world = env.worlds[world_i];
            constexpr TI M = WORLD::INSTANCES;
            auto& renderer = rlt::get_renderer(device_gpu, world);
            cudaStream_t render_stream = rlt::stream(device_gpu, renderer);
            cudaStreamWaitEvent(render_stream, cameras_ready, 0);
            cudaMemcpyAsync(rlt::data(rlt::cameras(device_gpu, renderer)), staging_cameras + world_i * M, M * sizeof(rlt::hyperlux::Camera<T>), cudaMemcpyDeviceToDevice, render_stream);
            // the host device: cameras_ready alone orders the launch, so the renderers run concurrently
            rlt::render_launch(device, renderer);
        }
    };
    measure(hand_rolled_render_only, "hand-rolled render only", "hand_rolled_render");
    measure(world_render_only, "world render only", "world_render");
    double hand_rolled = measure(hand_rolled_path, "hand-rolled (training_cuda style)", "hand_rolled_step");
    double world_verbs = measure(world_path, "world verbs", "world_step");
    EXPECT_GE(world_verbs, hand_rolled * 0.9) << "the World verb path regressed against the hand-rolled baseline";

    decltype(states) next_states;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    using rlt::malloc;
    using rlt::free;
    using rlt::set_all;
    using rlt::step;
    malloc(device_gpu, next_states);
    malloc(device_gpu, actions);
    set_all(device_gpu, actions, (T)0);
    auto world_step_render_observe = [&](){
        step(device_gpu, env, parameters, states, actions, next_states, rng);
        std::swap(states, next_states);
        rlt::render(device_gpu, env, parameters, states, reset_mask);
        rlt::observe(device_gpu, env, parameters, states, typename WORLD::Observation{}, observations, rng);
    };
    measure(world_step_render_observe, "world step/render/observe", "world_step_render_observe");
    std::size_t free_after;
    ASSERT_EQ(cudaMemGetInfo(&free_after, &total_bytes), cudaSuccess);
    struct rusage usage{};
    ASSERT_EQ(getrusage(RUSAGE_SELF, &usage), 0);
    metra::log("hyperdrone/benchmark_cuda/world_step_render_observe/device_bytes", double(free_before) - double(free_after));
    metra::log("hyperdrone/benchmark_cuda/world_step_render_observe/peak_rss_bytes", double(usage.ru_maxrss) * 1024);
    free(device_gpu, actions);
    free(device_gpu, next_states);

    cudaFree(staging_cameras);
    cudaFree(hand_rolled_history);
    cudaFree(hand_rolled_observations);
    cudaEventDestroy(cameras_ready);
    for(TI world_i = 0; world_i < N_WORLDS; world_i++){
        cudaEventDestroy(render_scatter_done[world_i]);
    }
    rlt::free(device_gpu, parameters);
    rlt::free(device_gpu, states);
    rlt::free(device_gpu, reset_mask);
    rlt::free(device_gpu, observations);
    rlt::free(device_gpu, rng);
    rlt::free(device_gpu, env);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_BENCHMARK, SELF_VISIBLE_MOTION_BLUR){
    using WORLD_SPEC = benchmark_self_visible::WORLD_SPEC;
    using WORLD = benchmark_self_visible::WORLD;
    using ENVIRONMENT = benchmark_self_visible::ENVIRONMENT;
    using benchmark_self_visible::drone;
    using REGISTRY = l2f::parameters::dynamics::REGISTRY;
    if(SCENE_PATH.empty() || !std::filesystem::exists(DATA_PATH + "/crazyflie.glb")){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH with the rigged drones not set";
    }
    if(!cuda_available()){
        GTEST_SKIP() << "CUDA device unavailable";
    }
    DEVICE_GPU device_gpu;
    auto& device = device_gpu.rendering;
    rlt::init(device);
    rlt::init(device_gpu);
    rlt::hyperlux::datasets::procthor::GLB dataset{{}, {SCENE_PATH, SCENE_PATH}};
    // one body for every instance, or a body drawn per reset from four
    const std::vector<rlt::rl::environments::hyperdrone::Drone<WORLD_SPEC>> tables[] = {
        {drone<REGISTRY::x500>("x500.glb")},
        {drone<REGISTRY::x500>("x500.glb"), drone<REGISTRY::crazyflie>("crazyflie.glb"), drone<REGISTRY::arpl>("arpl.glb"), drone<REGISTRY::soft>("soft.glb")},
    };
    const char* metrics[] = {"one_drone", "four_drones"};
    for(TI table_i = 0; table_i < 2; table_i++){
        auto env_owner = std::make_unique<ENVIRONMENT>();
        auto& env = *env_owner;
        rlt::malloc(device_gpu, env);
        env.shared.drones = tables[table_i];
        rlt::init(device_gpu, env, dataset);
        ASSERT_EQ(env.shared.drone_meshes.size(), tables[table_i].size());
        RNG_GPU rng;
        rlt::malloc(device_gpu, rng);
        rlt::init(device_gpu, rng, 0);
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, ENVIRONMENT::INSTANCES>>> parameters;
        rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, ENVIRONMENT::INSTANCES>>> states;
        rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, ENVIRONMENT::INSTANCES>>> reset_mask;
        rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, ENVIRONMENT::INSTANCES, WORLD::OBSERVATION_DIM>>> observations;
        rlt::malloc(device_gpu, parameters);
        rlt::malloc(device_gpu, states);
        rlt::malloc(device_gpu, reset_mask);
        rlt::malloc(device_gpu, observations);
        rlt::set_all(device_gpu, reset_mask, true);
        rlt::sample_initial_parameters(device_gpu, env, parameters, reset_mask, rng);
        rlt::sample_initial_state(device_gpu, env, parameters, states, reset_mask, rng);
        rlt::set_all(device_gpu, reset_mask, false);
        auto body = [&](){
            for(auto& world: env.worlds){
                using rlt::request_render;
                request_render(device_gpu, world);
            }
            rlt::render(device_gpu, env, parameters, states, reset_mask);
            rlt::observe(device_gpu, env, parameters, states, typename WORLD::Observation{}, observations, rng);
        };
        for(TI step_i = 0; step_i < WARMUP_STEPS; step_i++){
            body();
        }
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(TI step_i = 0; step_i < MEASURE_STEPS; step_i++){
            body();
        }
        cudaDeviceSynchronize();
        double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
        double frames_per_second = MEASURE_STEPS / seconds;
        std::cout << "self-visible motion blur, " << metrics[table_i] << ": " << frames_per_second << " frames/s (" << seconds * 1000 / MEASURE_STEPS << " ms/frame, " << ENVIRONMENT::INSTANCES << " instances)" << std::endl;
        metra::log(std::string("hyperdrone/benchmark_cuda/self_visible_motion_blur/") + metrics[table_i] + "/frames_per_second", frames_per_second);
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
        rlt::free(device_gpu, parameters);
        rlt::free(device_gpu, states);
        rlt::free(device_gpu, reset_mask);
        rlt::free(device_gpu, observations);
        rlt::free(device_gpu, rng);
        rlt::free(device_gpu, env);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
