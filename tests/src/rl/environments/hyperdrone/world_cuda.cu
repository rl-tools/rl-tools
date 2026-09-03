#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cuda.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <filesystem>
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

namespace test_hyperdrone_world_cuda {
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
        using SHADING = rlt::rendering::raytracing::Low;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.25;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
}

using namespace test_hyperdrone_world_cuda;

template <typename T_DEVICE>
struct Tensors {
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES>>> rewards;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> terminated_flags;
    void allocate(T_DEVICE& device){
        rlt::malloc(device, parameters);
        rlt::malloc(device, states);
        rlt::malloc(device, next_states);
        rlt::malloc(device, reset_mask);
        rlt::malloc(device, actions);
        rlt::malloc(device, observations);
        rlt::malloc(device, rewards);
        rlt::malloc(device, terminated_flags);
    }
    void deallocate(T_DEVICE& device){
        rlt::free(device, parameters);
        rlt::free(device, states);
        rlt::free(device, next_states);
        rlt::free(device, reset_mask);
        rlt::free(device, actions);
        rlt::free(device, observations);
        rlt::free(device, rewards);
        rlt::free(device, terminated_flags);
    }
};

static bool cuda_available(){
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

constexpr TI ROLLOUT_STEPS = 5;
struct Rollout {
    typename WORLD::State states[ROLLOUT_STEPS][INSTANCES];
    T observations[ROLLOUT_STEPS][INSTANCES][WORLD::OBSERVATION_DIM];
    T rewards[ROLLOUT_STEPS][INSTANCES];
};

static void cuda_rollout(DEVICE& device, DEVICE_GPU& device_gpu, WORLD& world, Rollout& record, TI seed){
    RNG_GPU rng;
    rlt::malloc(device_gpu, rng);
    rlt::init(device_gpu, rng, seed);
    Tensors<DEVICE_GPU> tensors;
    tensors.allocate(device_gpu);
    Tensors<DEVICE> host;
    host.allocate(device);
    world.history_step = 0;

    rlt::set_all(device_gpu, tensors.reset_mask, true);
    rlt::set_all(device_gpu, tensors.actions, (T)0);
    rlt::sample_initial_parameters(device_gpu, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device_gpu, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    for(TI step_i = 0; step_i < ROLLOUT_STEPS; step_i++){
        rlt::render(device_gpu, world, tensors.parameters, tensors.states, tensors.reset_mask);
        rlt::observe(device_gpu, world, tensors.parameters, tensors.states, typename WORLD::Observation{}, tensors.observations, rng);
        rlt::step(device_gpu, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
        rlt::reward(device_gpu, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, tensors.rewards, rng);
        rlt::terminated(device_gpu, world, tensors.parameters, tensors.states, tensors.terminated_flags, rng);
        cudaDeviceSynchronize();
        rlt::copy(device_gpu, device, tensors.next_states, host.states);
        rlt::copy(device_gpu, device, tensors.observations, host.observations);
        rlt::copy(device_gpu, device, tensors.rewards, host.rewards);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            record.states[step_i][instance_i] = rlt::get(device, host.states, instance_i);
            record.rewards[step_i][instance_i] = rlt::get(device, host.rewards, instance_i);
            for(TI dim_i = 0; dim_i < WORLD::OBSERVATION_DIM; dim_i++){
                record.observations[step_i][instance_i][dim_i] = rlt::get(device, host.observations, instance_i, dim_i);
            }
        }
        rlt::copy(device_gpu, device_gpu, tensors.next_states, tensors.states);
        rlt::set_all(device_gpu, tensors.reset_mask, false);
    }
    tensors.deallocate(device_gpu);
    host.deallocate(device);
    rlt::free(device_gpu, rng);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD_CUDA, SEEDED_ROLLOUT_DETERMINISM){
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
    typename WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device_gpu, world);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device_gpu, world, shared, dataset, corpus, 0, 1, 0);

    auto* record_a = new Rollout;
    auto* record_b = new Rollout;
    cuda_rollout(device, device_gpu, world, *record_a, 1337);
    cuda_rollout(device, device_gpu, world, *record_b, 1337);
    bool nonzero_observation = false;
    for(TI step_i = 0; step_i < ROLLOUT_STEPS; step_i++){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            ASSERT_EQ(std::memcmp(&record_a->states[step_i][instance_i], &record_b->states[step_i][instance_i], sizeof(typename WORLD::State)), 0) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(record_a->rewards[step_i][instance_i], record_b->rewards[step_i][instance_i]);
            for(TI dim_i = 0; dim_i < WORLD::OBSERVATION_DIM; dim_i++){
                ASSERT_EQ(record_a->observations[step_i][instance_i][dim_i], record_b->observations[step_i][instance_i][dim_i]) << "step " << step_i << " instance " << instance_i << " dim " << dim_i;
                nonzero_observation = nonzero_observation || record_a->observations[step_i][instance_i][dim_i] != (T)0;
            }
        }
    }
    EXPECT_TRUE(nonzero_observation);
    delete record_a;
    delete record_b;
    rlt::free(device_gpu, world);
    rlt::free(device, shared.library);
}

// the CUDA per-step path renders the same frames as the CPU path for identical inputs, up to
// gcc-vs-nvcc pose-producer ULPs
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_WORLD_CUDA, CPU_CUDA_RENDER_CONSISTENCY){
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
    WORLD world_cpu, world_gpu;
    typename WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world_cpu);
    rlt::malloc(device_gpu, world_gpu);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world_cpu, shared, dataset, corpus, 0, 1, 0);
    rlt::init(device_gpu, world_gpu, shared, dataset, corpus, 0, 1, 0);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 7);
    Tensors<DEVICE> host;
    host.allocate(device);
    rlt::set_all(device, host.reset_mask, true);
    rlt::sample_initial_parameters(device, world_cpu, host.parameters, host.reset_mask, rng);
    rlt::sample_initial_state(device, world_cpu, host.parameters, host.states, host.reset_mask, rng);

    rlt::render(device, world_cpu, host.parameters, host.states, host.reset_mask);
    RNG rng_observe;
    rlt::malloc(device, rng_observe);
    rlt::init(device, rng_observe, 0);
    rlt::observe(device, world_cpu, host.parameters, host.states, typename WORLD::Observation{}, host.observations, rng_observe);

    Tensors<DEVICE_GPU> gpu;
    gpu.allocate(device_gpu);
    rlt::copy(device, device_gpu, host.parameters, gpu.parameters);
    rlt::copy(device, device_gpu, host.states, gpu.states);
    rlt::copy(device, device_gpu, host.reset_mask, gpu.reset_mask);
    RNG_GPU rng_gpu;
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_gpu, rng_gpu, 0);
    rlt::render(device_gpu, world_gpu, gpu.parameters, gpu.states, gpu.reset_mask);
    rlt::observe(device_gpu, world_gpu, gpu.parameters, gpu.states, typename WORLD::Observation{}, gpu.observations, rng_gpu);
    cudaDeviceSynchronize();
    Tensors<DEVICE> downloaded;
    downloaded.allocate(device);
    rlt::copy(device_gpu, device, gpu.observations, downloaded.observations);

    TI large_deltas = 0;
    T max_delta = 0;
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        for(TI dim_i = 0; dim_i < WORLD::OBSERVATION_DIM; dim_i++){
            T cpu_value = rlt::get(device, host.observations, instance_i, dim_i);
            T gpu_value = rlt::get(device, downloaded.observations, instance_i, dim_i);
            T delta = std::abs(cpu_value - gpu_value);
            max_delta = delta > max_delta ? delta : max_delta;
            if(delta > (T)0.05){
                large_deltas++;
            }
        }
    }
    EXPECT_LE(large_deltas, INSTANCES * WORLD::OBSERVATION_DIM / 1000) << "max_delta " << max_delta;

    host.deallocate(device);
    gpu.deallocate(device_gpu);
    downloaded.deallocate(device);
    rlt::free(device, world_cpu);
    rlt::free(device_gpu, world_gpu);
    rlt::free(device, shared.library);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
