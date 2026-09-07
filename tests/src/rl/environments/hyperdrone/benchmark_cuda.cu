// Phase-2 throughput gate: the World's render+observe verbs vs the hand-rolled
// training_cuda.cu-style per-step path (pose kernel on the main stream, event, per-renderer
// camera D2D staging, render_launch, scatter) at the matched configuration
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cuda.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_GPU_SPEC = rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, DEVICE>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<DEVICE_GPU_SPEC>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
static const std::string SCENE_PATH = "";
#endif

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
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 64;
        static constexpr TI CAM_WIDTH = 80;
        static constexpr TI CAM_HEIGHT = 50;
        using SHADING = rlt::rendering::raytracing::High;
        static constexpr bool ENABLE_ANTI_ALIASING = true;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = 2;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.5;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    constexpr TI NUMBER_OF_ENVIRONMENTS = 2;
    using ENVIRONMENT = rlt::rl::environments::hyperdrone::MultiEnvironment<WORLD, NUMBER_OF_ENVIRONMENTS>;
    constexpr TI INSTANCES = ENVIRONMENT::INSTANCES;
    constexpr TI WARMUP_STEPS = 5;
    constexpr TI MEASURE_STEPS = 50;
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
    DEVICE device;
    rlt::init(device);
    DEVICE_GPU device_gpu;
    rlt::init(device_gpu);
    device_gpu.rendering = &device;

    ENVIRONMENT env;
    rlt::malloc(device_gpu, env);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH, SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
        rlt::init(device_gpu, env.environments[environment_i], env.shared, dataset, corpus, environment_i, 1, environment_i);
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

    // World verb path
    auto world_path = [&](){
        rlt::render(device_gpu, env, parameters, states, reset_mask);
        rlt::observe(device_gpu, env, parameters, states, typename WORLD::Observation{}, observations, rng);
    };
    // hand-rolled training_cuda.cu-style path over the same renderers: pose kernel on the main
    // stream into a staging buffer, event, per-renderer camera D2D + render_launch + scatter into
    // the frame history, then the observation assembly kernel (training_cuda scatters into
    // gpu_frame_stack_history and assembles the policy input from it)
    rlt::rendering::raytracing::Camera<T>* staging_cameras;
    cudaMalloc(&staging_cameras, INSTANCES * sizeof(rlt::rendering::raytracing::Camera<T>));
    cudaEvent_t cameras_ready;
    cudaEventCreateWithFlags(&cameras_ready, cudaEventDisableTiming);
    float* hand_rolled_history;
    cudaMalloc(&hand_rolled_history, INSTANCES * WORLD::OBSERVATION_DIM * sizeof(float));
    float* hand_rolled_observations;
    cudaMalloc(&hand_rolled_observations, INSTANCES * WORLD::OBSERVATION_DIM * sizeof(float));
    // training_cuda joins the render streams back into the main stream before the policy
    // evaluation consumes the observations (render_scatter_done events)
    cudaEvent_t render_scatter_done[NUMBER_OF_ENVIRONMENTS];
    for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
        cudaEventCreateWithFlags(&render_scatter_done[environment_i], cudaEventDisableTiming);
    }
    auto hand_rolled_path = [&](){
        using TAG = rlt::devices::cuda::TAG<DEVICE_GPU, true>;
        TAG tag_device{};
        constexpr TI BLOCKSIZE = 32;
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            auto& world = env.environments[environment_i];
            constexpr TI M = WORLD::INSTANCES;
            auto parameters_block = rlt::view_range(device_gpu, parameters, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto states_block = rlt::view_range(device_gpu, states, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto reset_block = rlt::view_range(device_gpu, reset_mask, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(M, BLOCKSIZE);
            rlt::rl::environments::hyperdrone::cuda::pose_kernel<TAG, WORLD_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device_gpu.stream>>>(tag_device, parameters_block, states_block, reset_block, staging_cameras + environment_i * M, staging_cameras + environment_i * M, rlt::data(world.prev_cameras), world.history_step, (T)WORLD_SPEC::CAM_WIDTH / (T)WORLD_SPEC::CAM_HEIGHT);
        }
        cudaEventRecord(cameras_ready, device_gpu.stream);
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            auto& world = env.environments[environment_i];
            constexpr TI M = WORLD::INSTANCES;
            cudaStream_t render_stream = rlt::stream(device_gpu, world.renderer);
            cudaStreamWaitEvent(render_stream, cameras_ready, 0);
            cudaMemcpyAsync(rlt::data(rlt::cameras(device_gpu, world.renderer)), staging_cameras + environment_i * M, M * sizeof(rlt::rendering::raytracing::Camera<T>), cudaMemcpyDeviceToDevice, render_stream);
            rlt::render_launch(device_gpu, world.renderer);
            auto parameters_block = rlt::view_range(device_gpu, parameters, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto reset_block = rlt::view_range(device_gpu, reset_mask, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            constexpr TI SCATTER_BLOCKSIZE = 256;
            constexpr TI SCATTER_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(M * WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT, SCATTER_BLOCKSIZE);
            rlt::rl::environments::hyperdrone::cuda::scatter_kernel<TAG, WORLD_SPEC><<<dim3(SCATTER_BLOCKS), dim3(SCATTER_BLOCKSIZE), 0, render_stream>>>(tag_device, rlt::data(world.renderer.observation), hand_rolled_history + environment_i * M * WORLD::OBSERVATION_DIM, parameters_block, rlt::data(world.episode_start), reset_block, world.history_step);
            auto observations_block = rlt::view_range(device_gpu, observations, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            constexpr TI ASSEMBLY_BLOCKSIZE = 256;
            constexpr TI ASSEMBLY_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(M * WORLD::FRAME_DIM, ASSEMBLY_BLOCKSIZE);
            rlt::rl::environments::hyperdrone::cuda::observe_kernel<TAG, WORLD_SPEC><<<dim3(ASSEMBLY_BLOCKS), dim3(ASSEMBLY_BLOCKSIZE), 0, render_stream>>>(tag_device, hand_rolled_history + environment_i * M * WORLD::OBSERVATION_DIM, observations_block);
            cudaEventRecord(render_scatter_done[environment_i], render_stream);
            cudaStreamWaitEvent(device_gpu.stream, render_scatter_done[environment_i], 0);
        }
    };

    auto measure = [&](auto&& body, const char* label){
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
        return steps_per_second;
    };

    auto world_render_only = [&](){
        rlt::render(device_gpu, env, parameters, states, reset_mask);
    };
    auto hand_rolled_render_only = [&](){
        using TAG = rlt::devices::cuda::TAG<DEVICE_GPU, true>;
        TAG tag_device{};
        constexpr TI BLOCKSIZE = 32;
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            auto& world = env.environments[environment_i];
            constexpr TI M = WORLD::INSTANCES;
            auto parameters_block = rlt::view_range(device_gpu, parameters, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto states_block = rlt::view_range(device_gpu, states, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            auto reset_block = rlt::view_range(device_gpu, reset_mask, environment_i * M, rlt::tensor::ViewSpec<0, M>{});
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(M, BLOCKSIZE);
            rlt::rl::environments::hyperdrone::cuda::pose_kernel<TAG, WORLD_SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device_gpu.stream>>>(tag_device, parameters_block, states_block, reset_block, staging_cameras + environment_i * M, staging_cameras + environment_i * M, rlt::data(world.prev_cameras), world.history_step, (T)WORLD_SPEC::CAM_WIDTH / (T)WORLD_SPEC::CAM_HEIGHT);
        }
        cudaEventRecord(cameras_ready, device_gpu.stream);
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            auto& world = env.environments[environment_i];
            constexpr TI M = WORLD::INSTANCES;
            cudaStream_t render_stream = rlt::stream(device_gpu, world.renderer);
            cudaStreamWaitEvent(render_stream, cameras_ready, 0);
            cudaMemcpyAsync(rlt::data(rlt::cameras(device_gpu, world.renderer)), staging_cameras + environment_i * M, M * sizeof(rlt::rendering::raytracing::Camera<T>), cudaMemcpyDeviceToDevice, render_stream);
            rlt::render_launch(device_gpu, world.renderer);
        }
    };
    measure(hand_rolled_render_only, "hand-rolled render only");
    measure(world_render_only, "world render only");
    double hand_rolled = measure(hand_rolled_path, "hand-rolled (training_cuda style)");
    double world_verbs = measure(world_path, "world verbs");
    EXPECT_GE(world_verbs, hand_rolled * 0.9) << "the World verb path regressed against the hand-rolled baseline";

    cudaFree(staging_cameras);
    cudaFree(hand_rolled_history);
    cudaFree(hand_rolled_observations);
    cudaEventDestroy(cameras_ready);
    for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
        cudaEventDestroy(render_scatter_done[environment_i]);
    }
    rlt::free(device_gpu, parameters);
    rlt::free(device_gpu, states);
    rlt::free(device_gpu, reset_mask);
    rlt::free(device_gpu, observations);
    rlt::free(device_gpu, env);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
