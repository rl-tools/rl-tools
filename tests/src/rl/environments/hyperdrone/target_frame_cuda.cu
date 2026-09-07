#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cuda.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cstring>
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

namespace test_target_frame_cuda {
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
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
    struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, DYNAMICS_STATIC_PARAMETERS> {
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 4;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        static constexpr TI HISTORY_LENGTH = 4;
        using SHADING = rlt::rendering::raytracing::Low;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.25;
    };
    using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<BASE_WORLD> {
        static constexpr TI IMAGE_STACK_N = 2;
        static constexpr TI IMAGE_STACK_STRIDE = 1;
        static constexpr TI PAD_CHANNELS_TO = 8;
        static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = 0.1;
        static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = 0.25;
    };
    using WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI IMAGE_CHANNELS = BASE_WORLD::IMAGE_CHANNELS;
    constexpr TI CAM_PIXELS = WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT;
}

using namespace test_target_frame_cuda;

static bool cuda_available(){
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TARGET_FRAME_CUDA, CACHE_AND_STACK_SEMANTICS){
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
    WORLD world;
    typename BASE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device_gpu, world);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device_gpu, world, shared, dataset, corpus, 0, 1, 0);

    RNG_GPU rng;
    rlt::malloc(device_gpu, rng);
    rlt::init(device_gpu, rng, 1337);

    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM>>> observations;
    rlt::malloc(device_gpu, parameters);
    rlt::malloc(device_gpu, states);
    rlt::malloc(device_gpu, reset_mask);
    rlt::malloc(device_gpu, observations);

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM>>> observations_host;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states_host;
    DEVICE host_device;
    rlt::malloc(host_device, observations_host);
    rlt::malloc(host_device, states_host);

    rlt::set_all(device_gpu, reset_mask, true);
    rlt::sample_initial_parameters(device_gpu, world, parameters, reset_mask, rng);
    rlt::sample_initial_state(device_gpu, world, parameters, states, reset_mask, rng);
    rlt::render(device_gpu, world, parameters, states, reset_mask);
    rlt::observe(device_gpu, world, parameters, states, typename WORLD::Observation{}, observations, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, host_device, observations, observations_host);

    auto channel_value = [&](TI instance_i, TI pixel_i, TI channel){
        return rlt::get(host_device, observations_host, instance_i, pixel_i * WORLD::OBSERVATION_CHANNELS + channel);
    };
    std::vector<T> target_step0(INSTANCES * CAM_PIXELS * IMAGE_CHANNELS);
    std::vector<T> frame0_step0(INSTANCES * CAM_PIXELS * IMAGE_CHANNELS);
    bool target_nonzero = false;
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
            for(TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++){
                T target_value = channel_value(instance_i, pixel_i, TASK_SPEC::IMAGE_STACK_N * IMAGE_CHANNELS + channel_i);
                target_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i] = target_value;
                frame0_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i] = channel_value(instance_i, pixel_i, channel_i);
                target_nonzero = target_nonzero || target_value != (T)0;
                ASSERT_EQ(channel_value(instance_i, pixel_i, channel_i), channel_value(instance_i, pixel_i, IMAGE_CHANNELS + channel_i)) << "stack should clamp to the episode start";
            }
            for(TI pad_i = (TASK_SPEC::IMAGE_STACK_N + 1) * IMAGE_CHANNELS; pad_i < WORLD::OBSERVATION_CHANNELS; pad_i++){
                ASSERT_EQ(channel_value(instance_i, pixel_i, pad_i), (T)0);
            }
        }
    }
    EXPECT_TRUE(target_nonzero);

    // no reset, moved states: cached target byte-identical, stack shifts
    rlt::set_all(device_gpu, reset_mask, false);
    rlt::copy(device_gpu, host_device, states, states_host);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        typename WORLD::State state = rlt::get(host_device, states_host, instance_i);
        state.position[0] += (T)0.2;
        rlt::set(host_device, states_host, state, instance_i);
    }
    rlt::copy(host_device, device_gpu, states_host, states);
    rlt::render(device_gpu, world, parameters, states, reset_mask);
    rlt::observe(device_gpu, world, parameters, states, typename WORLD::Observation{}, observations, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, host_device, observations, observations_host);
    bool student_changed = false;
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
            for(TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++){
                ASSERT_EQ(channel_value(instance_i, pixel_i, TASK_SPEC::IMAGE_STACK_N * IMAGE_CHANNELS + channel_i), target_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i]) << "target frame not cached";
                ASSERT_EQ(channel_value(instance_i, pixel_i, IMAGE_CHANNELS + channel_i), frame0_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i]) << "stack frame 1 should be the previous frame 0";
                student_changed = student_changed || channel_value(instance_i, pixel_i, channel_i) != frame0_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i];
            }
        }
    }
    EXPECT_TRUE(student_changed);

    rlt::free(device_gpu, parameters);
    rlt::free(device_gpu, states);
    rlt::free(device_gpu, reset_mask);
    rlt::free(device_gpu, observations);
    rlt::free(host_device, observations_host);
    rlt::free(host_device, states_host);
    rlt::free(device_gpu, world);
    rlt::free(device, shared.library);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
