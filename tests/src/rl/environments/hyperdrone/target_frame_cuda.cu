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
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;
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

template <typename MEMBER>
void check_rng_partitioning(){
    DEVICE host;
    DEVICE_GPU device;
    rlt::init(host);
    rlt::init(device);
    using MULTI = rlt::rl::environments::hyperdrone::MultiEnvironment<MEMBER, 2>;
    MULTI multi{};
    MEMBER isolated{};
    multi.environments[1].rng_offset = MEMBER::INSTANCES;
    isolated.rng_offset = MEMBER::INSTANCES;
    using PARAMETERS_SPEC = rlt::tensor::Specification<typename MEMBER::Parameters, TI, rlt::tensor::Shape<TI, MULTI::INSTANCES>>;
    using LOCAL_SPEC = rlt::tensor::Specification<typename MEMBER::Parameters, TI, rlt::tensor::Shape<TI, MEMBER::INSTANCES>>;
    rlt::Tensor<PARAMETERS_SPEC> parameters, parameters_host;
    rlt::Tensor<LOCAL_SPEC> local, local_host;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, MULTI::INSTANCES>>> reset_mask, mask_host;
    rlt::malloc(device, parameters);
    rlt::malloc(host, parameters_host);
    rlt::malloc(device, local);
    rlt::malloc(host, local_host);
    rlt::malloc(device, reset_mask);
    rlt::malloc(host, mask_host);
    RNG_GPU shared_rng, isolated_rng;
    rlt::malloc(device, shared_rng);
    rlt::malloc(device, isolated_rng);
    rlt::init(device, shared_rng, 1337);
    rlt::init(device, isolated_rng, 1337);
    auto local_mask = rlt::view_range(device, reset_mask, MEMBER::INSTANCES, rlt::tensor::ViewSpec<0, MEMBER::INSTANCES>{});
    for(TI round = 0; round < 4; round++){
        for(TI i = 0; i < MULTI::INSTANCES; i++){
            rlt::set(host, mask_host, round == 0 || (i + round) % 3 == 0, i);
        }
        rlt::copy(host, device, mask_host, reset_mask);
        rlt::sample_initial_parameters(device, multi, parameters, reset_mask, shared_rng);
        rlt::sample_initial_parameters(device, isolated, local, local_mask, isolated_rng);
        rlt::copy(device, host, parameters, parameters_host);
        rlt::copy(device, host, local, local_host);
        for(TI i = 0; i < MEMBER::INSTANCES; i++){
            const auto& a = rlt::get_ref(host, parameters_host, MEMBER::INSTANCES + i);
            const auto& b = rlt::get_ref(host, local_host, i);
            EXPECT_EQ(a.fov, b.fov) << "round " << round << " instance " << i;
            EXPECT_EQ(a.brightness_scale, b.brightness_scale);
            if constexpr(rlt::utils::typing::is_same_v<MEMBER, WORLD>){
                EXPECT_EQ(a.target_roll, b.target_roll);
                EXPECT_EQ(a.target_pitch, b.target_pitch);
                EXPECT_EQ(a.brightness_mismatch, b.brightness_mismatch);
            }
        }
    }
    EXPECT_NE(rlt::get_ref(host, parameters_host, 0).brightness_scale,
              rlt::get_ref(host, parameters_host, MEMBER::INSTANCES).brightness_scale);
    isolated.rng_offset = RNG_GPU::NUM_RNGS - MEMBER::INSTANCES + 1;
    EXPECT_THROW(rlt::sample_initial_parameters(device, isolated, local, local_mask, isolated_rng), std::out_of_range);
    rlt::free(device, shared_rng);
    rlt::free(device, isolated_rng);
    rlt::free(device, parameters);
    rlt::free(host, parameters_host);
    rlt::free(device, local);
    rlt::free(host, local_host);
    rlt::free(device, reset_mask);
    rlt::free(host, mask_host);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TARGET_FRAME_CUDA, GLOBAL_RNG_PARTITIONS){
    if(!cuda_available()){
        GTEST_SKIP() << "CUDA device unavailable";
    }
    check_rng_partitioning<BASE_WORLD>();
    check_rng_partitioning<WORLD>();
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TARGET_FRAME_CUDA, CACHE_AND_STACK_SEMANTICS){
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
    WORLD world;
    typename BASE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world, shared, dataset, corpus, 0, 1, 0);

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
    rlt::free(device, world);
    rlt::free(device, shared.library);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
