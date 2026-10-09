#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cuda.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_GPU_SPEC = rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, DEVICE>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<DEVICE_GPU_SPEC>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
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
        static constexpr TI TARGET_BANK_REFRESH_STEPS = 3;
    };
    using WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI IMAGE_CHANNELS = BASE_WORLD::IMAGE_CHANNELS;
    constexpr TI CAM_PIXELS = WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT;
    constexpr TI FRAME_DIM = BASE_WORLD::FRAME_DIM;
}

using namespace test_target_frame_cuda;

static bool cuda_available(){
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TARGET_FRAME_CUDA, GLOBAL_RNG_PARTITIONS){
    if(!cuda_available()){
        GTEST_SKIP() << "CUDA device unavailable";
    }
    using MEMBER = BASE_WORLD;
    DEVICE host;
    DEVICE_GPU device;
    rlt::init(host);
    rlt::init(device);
    using MULTI = rlt::rl::environments::hyperdrone::MultiEnvironment<MEMBER, 2>;
    MULTI multi{};
    MEMBER isolated{};
    rlt::malloc(device, multi);
    rlt::malloc(device, isolated);
    multi.shared.drones.push_back({});
    rlt::init(device, multi.shared);
    using DRONE_SPEC = typename MEMBER::DRONE_PARAMETERS_SPEC;
    std::vector<typename DRONE_SPEC::T> nominals(DRONE_SPEC::SIZE, multi.shared.drones[0].parameters);
    rlt::Tensor<DRONE_SPEC> table_host;
    table_host._data = nominals.data();
    auto init_sampling = [&](MEMBER& world){
        world.num_drones = multi.shared.num_drones;
        rlt::copy(host, device, table_host, world.drone_parameters);
    };
    for(auto& world : multi.environments){
        init_sampling(world);
    }
    init_sampling(isolated);
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
            EXPECT_EQ(a.camera.fov, b.camera.fov) << "round " << round << " instance " << i;
            EXPECT_EQ(a.brightness_scale, b.brightness_scale);
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
    rlt::free(device, isolated);
    rlt::free(device, multi);
}

// host copies of the World's device-resident bank state
struct BankMirror {
    rlt::Tensor<typename WORLD::SETUPS_SPEC> setups;
    rlt::Tensor<typename WORLD::FRAMES_SPEC> bank_frames;
    rlt::Tensor<typename WORLD::SETUP_INDEX_SPEC> setup_index;
    rlt::Tensor<typename WORLD::BANK_CURSOR_SPEC> bank_cursor;
    rlt::Tensor<typename WORLD::FRAMES_SPEC> target_frames;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    void allocate(DEVICE& host){
        rlt::malloc(host, setups);
        rlt::malloc(host, bank_frames);
        rlt::malloc(host, setup_index);
        rlt::malloc(host, bank_cursor);
        rlt::malloc(host, target_frames);
        rlt::malloc(host, parameters);
    }
    void deallocate(DEVICE& host){
        rlt::free(host, setups);
        rlt::free(host, bank_frames);
        rlt::free(host, setup_index);
        rlt::free(host, bank_cursor);
        rlt::free(host, target_frames);
        rlt::free(host, parameters);
    }
    template <typename PARAMETERS>
    void download(DEVICE_GPU& device_gpu, DEVICE& host, WORLD& world, PARAMETERS& device_parameters){
        cudaDeviceSynchronize();
        rlt::copy(device_gpu, host, world.setups, setups);
        rlt::copy(device_gpu, host, world.bank_frames, bank_frames);
        rlt::copy(device_gpu, host, world.setup_index, setup_index);
        rlt::copy(device_gpu, host, world.bank_cursor, bank_cursor);
        rlt::copy(device_gpu, host, world.target_frames, target_frames);
        rlt::copy(device_gpu, host, device_parameters, parameters);
    }
};

static void expect_adopted(DEVICE& host, BankMirror& mirror, TI instance_i){
    const TI index = rlt::get(host, mirror.setup_index, instance_i);
    ASSERT_LT(index, INSTANCES);
    const auto& setup = rlt::get_ref(host, mirror.setups, index);
    const typename WORLD::Parameters parameters = rlt::get(host, mirror.parameters, instance_i);
    EXPECT_EQ(parameters.camera.fov, setup.camera.fov);
    EXPECT_EQ(parameters.brightness_scale, setup.brightness_scale);
    EXPECT_EQ(parameters.scene_translation[0], setup.scene_translation[0]);
    EXPECT_EQ(parameters.target_roll, setup.target_roll);
    EXPECT_EQ(parameters.brightness_mismatch, setup.brightness_mismatch);
    for(TI element_i = 0; element_i < FRAME_DIM; element_i++){
        ASSERT_EQ(rlt::get(host, mirror.target_frames, instance_i, element_i), rlt::get(host, mirror.bank_frames, index, element_i)) << "instance " << instance_i << " element " << element_i;
    }
}

template<bool RAW_FIRST>
void bank_and_stack_semantics(){
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
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH, SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device_gpu, world, shared, dataset, corpus, 0, 2, 0);
    EXPECT_TRUE(rlt::rl::environments::hyperdrone::tasks::target_frame::_bank_due(world));

    RNG_GPU rng;
    rlt::malloc(device_gpu, rng);
    rlt::init(device_gpu, rng, 1337);

    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM>>> observations;
    rlt::malloc(device_gpu, parameters);
    rlt::malloc(device_gpu, states);
    rlt::malloc(device_gpu, next_states);
    rlt::malloc(device_gpu, reset_mask);
    rlt::malloc(device_gpu, actions);
    rlt::malloc(device_gpu, observations);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, BASE_WORLD::OBSERVATION_DIM>>> raw_observations;
    rlt::malloc(device_gpu, raw_observations);
    rlt::set_all(device_gpu, world.target_frames, (T)-123);
    rlt::set_all(device_gpu, actions, (T)0);
    auto render_observation = [&](){
        rlt::request_render(device_gpu, world, reset_mask);
        if constexpr(RAW_FIRST){
            rlt::observe(device_gpu, world, parameters, states, typename BASE_WORLD::Observation{}, raw_observations, rng);
        }
        else{
            rlt::render(device_gpu, world, parameters, states, reset_mask);
        }
        const TI history_step = world.history_step;
        rlt::observe(device_gpu, world, parameters, states, typename WORLD::Observation{}, observations, rng);
        EXPECT_EQ(world.history_step, history_step);
    };

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM>>> observations_host;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states_host;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_host;
    DEVICE host_device;
    rlt::malloc(host_device, observations_host);
    rlt::malloc(host_device, states_host);
    rlt::malloc(host_device, reset_host);
    BankMirror mirror;
    mirror.allocate(host_device);

    // the first reset fills the bank and the resetting instances adopt its setups in order, so
    // a full reset gives every instance its own
    rlt::set_all(device_gpu, reset_mask, true);
    rlt::sample_initial_parameters(device_gpu, world, parameters, reset_mask, rng);
    rlt::sample_initial_state(device_gpu, world, parameters, states, reset_mask, rng);
    EXPECT_EQ(world.bank_scene_version, world.scene_version);
    mirror.download(device_gpu, host_device, world, parameters);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_EQ(rlt::get(host_device, mirror.setup_index, instance_i), instance_i);
        expect_adopted(host_device, mirror, instance_i);
    }
    EXPECT_EQ(rlt::get(host_device, mirror.bank_cursor, (TI)0), INSTANCES);
    EXPECT_NE(rlt::get_ref(host_device, mirror.setups, 0).target_roll, (T)0);

    render_observation();
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
                ASSERT_EQ(target_value, rlt::get(host_device, mirror.target_frames, instance_i, pixel_i * IMAGE_CHANNELS + channel_i));
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
    render_observation();
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

    // reset instance 0 only: it adopts a setup from the bank, the others keep their rows
    rlt::set_all(host_device, reset_host, false);
    rlt::set(host_device, reset_host, true, (TI)0);
    rlt::copy(host_device, device_gpu, reset_host, reset_mask);
    rlt::sample_initial_parameters(device_gpu, world, parameters, reset_mask, rng);
    rlt::sample_initial_state(device_gpu, world, parameters, states, reset_mask, rng);
    mirror.download(device_gpu, host_device, world, parameters);
    expect_adopted(host_device, mirror, 0);
    EXPECT_EQ(rlt::get(host_device, mirror.bank_cursor, (TI)0), INSTANCES + 1);
    render_observation();
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, host_device, observations, observations_host);
    for(TI instance_i = 1; instance_i < INSTANCES; instance_i++){
        for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
            for(TI channel_i = 0; channel_i < IMAGE_CHANNELS; channel_i++){
                ASSERT_EQ(channel_value(instance_i, pixel_i, TASK_SPEC::IMAGE_STACK_N * IMAGE_CHANNELS + channel_i), target_step0[(instance_i * CAM_PIXELS + pixel_i) * IMAGE_CHANNELS + channel_i]);
            }
        }
    }

    // the bank refreshes every TARGET_BANK_REFRESH_STEPS frames without touching running episodes
    std::vector<float> bank_before(INSTANCES * FRAME_DIM), rows_before(INSTANCES * FRAME_DIM);
    std::memcpy(bank_before.data(), rlt::data(mirror.bank_frames), bank_before.size() * sizeof(float));
    std::memcpy(rows_before.data(), rlt::data(mirror.target_frames), rows_before.size() * sizeof(float));
    rlt::set_all(device_gpu, reset_mask, false);
    for(TI step_i = 0; step_i < TASK_SPEC::TARGET_BANK_REFRESH_STEPS; step_i++){
        rlt::step(device_gpu, world, parameters, states, actions, next_states, rng);
        rlt::copy(device_gpu, device_gpu, next_states, states);
        rlt::render(device_gpu, world, parameters, states, reset_mask);
    }
    rlt::sample_initial_parameters(device_gpu, world, parameters, reset_mask, rng);
    EXPECT_EQ(world.bank_step, world.history_step);
    mirror.download(device_gpu, host_device, world, parameters);
    EXPECT_EQ(rlt::get(host_device, mirror.bank_cursor, (TI)0), (TI)0);
    EXPECT_NE(std::memcmp(bank_before.data(), rlt::data(mirror.bank_frames), bank_before.size() * sizeof(float)), 0) << "the refresh should draw new setups";
    for(TI instance_i = 1; instance_i < INSTANCES; instance_i++){
        for(TI element_i = 0; element_i < FRAME_DIM; element_i++){
            ASSERT_EQ(rlt::get(host_device, mirror.target_frames, instance_i, element_i), rows_before[instance_i * FRAME_DIM + element_i]) << "a refresh must leave running episodes' targets alone";
        }
    }

    // a scene change invalidates the bank, re-selecting the same scene included
    rlt::rotate_scene(device_gpu, world);
    EXPECT_EQ(world.active_slot, (TI)1);
    EXPECT_TRUE(rlt::rl::environments::hyperdrone::tasks::target_frame::_bank_due(world));
    rlt::sample_initial_parameters(device_gpu, world, parameters, reset_mask, rng);
    EXPECT_EQ(world.bank_scene_version, world.scene_version);
    rlt::select_scene(device_gpu, world, world.active_slot);
    EXPECT_TRUE(rlt::rl::environments::hyperdrone::tasks::target_frame::_bank_due(world));

    mirror.deallocate(host_device);
    rlt::free(host_device, reset_host);
    rlt::free(device_gpu, raw_observations);
    rlt::free(device_gpu, rng);
    rlt::free(device_gpu, parameters);
    rlt::free(device_gpu, states);
    rlt::free(device_gpu, next_states);
    rlt::free(device_gpu, reset_mask);
    rlt::free(device_gpu, actions);
    rlt::free(device_gpu, observations);
    rlt::free(host_device, observations_host);
    rlt::free(host_device, states_host);
    rlt::free(device_gpu, world);
    rlt::free(device, shared.library);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TARGET_FRAME_CUDA, BANK_AND_STACK_SEMANTICS){
    bank_and_stack_semantics<false>();
}
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TARGET_FRAME_CUDA, RAW_OBSERVATION_KEEPS_TARGET){
    bank_and_stack_semantics<true>();
}

// the CUDA bank render agrees with the host render of the same setups, up to gcc-vs-nvcc
// camera-producer ULPs
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_TARGET_FRAME_CUDA, CPU_CUDA_BANK_CONSISTENCY){
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
    typename BASE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    WORLD world_cpu, world_gpu;
    rlt::malloc(device, world_cpu);
    rlt::malloc(device_gpu, world_gpu);
    rlt::init(device, world_cpu, shared, dataset, corpus, 0, 1, 0);
    rlt::init(device_gpu, world_gpu, shared, dataset, corpus, 0, 1, 1);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    rlt::sample_target_setups(device, world_cpu, rng);
    rlt::copy(device, device_gpu, world_cpu.setups, world_gpu.setups);
    rlt::render_target_bank(device, world_cpu);
    rlt::render_target_bank(device_gpu, world_gpu);
    cudaDeviceSynchronize();
    rlt::Tensor<typename WORLD::FRAMES_SPEC> downloaded;
    rlt::malloc(device, downloaded);
    rlt::copy(device_gpu, device, world_gpu.bank_frames, downloaded);

    TI large_deltas = 0;
    T max_delta = 0;
    bool nonzero = false;
    for(TI setup_i = 0; setup_i < INSTANCES; setup_i++){
        for(TI element_i = 0; element_i < FRAME_DIM; element_i++){
            const T cpu_value = rlt::get(device, world_cpu.bank_frames, setup_i, element_i);
            const T gpu_value = rlt::get(device, downloaded, setup_i, element_i);
            const T delta = std::abs(cpu_value - gpu_value);
            max_delta = delta > max_delta ? delta : max_delta;
            if(delta > (T)0.05){
                large_deltas++;
            }
            nonzero = nonzero || cpu_value != (T)0;
        }
    }
    EXPECT_TRUE(nonzero);
    EXPECT_LE(large_deltas, INSTANCES * FRAME_DIM / 1000) << "max_delta " << max_delta;

    rlt::free(device, downloaded);
    rlt::free(device, rng);
    rlt::free(device, world_cpu);
    rlt::free(device_gpu, world_gpu);
    rlt::free(device, shared.library);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
