#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cuda.h>
#include <rl_tools/rl/components/episodes/operations_cpu.h>
#include <rl_tools/rl/components/episodes/operations_cuda.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cuda.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <string>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;
namespace episodes = rlt::rl::components::episodes;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<rlt::devices::DefaultCUDASpecification>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
static const std::string SCENE_PATH = "";
#endif

namespace test_hyperdrone_episodes_cuda {
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

    struct DYNAMICS_STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;
        static constexpr TI EPISODE_STEP_LIMIT = test_hyperdrone_episodes_cuda::EPISODE_STEP_LIMIT;
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
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    struct EPISODES_SPEC: episodes::Specification<WORLD> {};
    constexpr TI STEPS = 7;
    constexpr TI STEP_LIMIT = 3;
    using END_REASON = episodes::EndReason;
}

using namespace test_hyperdrone_episodes_cuda;

static bool cuda_available(){
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

// per-step trace of the bookkeeping, compared exactly between the CPU and CUDA paths
struct Trace {
    bool reset[STEPS][INSTANCES];
    TI episode_step[STEPS][INSTANCES];
    bool terminated[STEPS][INSTANCES];
    bool truncated[STEPS][INSTANCES];
    END_REASON end_reason[STEPS][INSTANCES];
    bool finished[STEPS][INSTANCES];
    TI finished_length[STEPS][INSTANCES];
    T finished_return[STEPS][INSTANCES];
    END_REASON finished_reason[STEPS][INSTANCES];
    T dataset_terminated[STEPS][INSTANCES];
    T dataset_truncated[STEPS][INSTANCES];
    T dataset_reset[STEPS][INSTANCES];
};

using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<rlt::numeric_types::Policy<T>, TI, WORLD, bool, INSTANCES>;
using DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, STEPS>;
using DATASET = rlt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;

// the same verb sequence on either device; forced reset (instance 0) before step 2
template <typename COMPUTE_DEVICE, typename COMPUTE_RNG>
static void trace_rollout(DEVICE& device, COMPUTE_DEVICE& device_compute, WORLD& world, Trace& trace, TI seed){
    using EPISODES = episodes::Episodes<EPISODES_SPEC>;
    using LOG = episodes::Log<EPISODES_SPEC, STEPS>;
    COMPUTE_RNG rng;
    rlt::malloc(device_compute, rng);
    rlt::init(device_compute, rng, seed);
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES>>> rewards;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> mask;
    EPISODES episodes_compute, episodes_host;
    LOG log_compute, log_host;
    DATASET dataset_compute, dataset_host;
    rlt::malloc(device_compute, parameters);
    rlt::malloc(device_compute, states);
    rlt::malloc(device_compute, next_states);
    rlt::malloc(device_compute, actions);
    rlt::malloc(device_compute, rewards);
    rlt::malloc(device_compute, mask);
    rlt::malloc(device_compute, episodes_compute);
    rlt::malloc(device_compute, log_compute);
    rlt::malloc(device_compute, dataset_compute);
    rlt::malloc(device, episodes_host);
    rlt::malloc(device, log_host);
    rlt::malloc(device, dataset_host);
    rlt::set_all(device_compute, actions, (T)0);
    rlt::set_all(device_compute, mask, false);
    rlt::init(device_compute, episodes_compute);
    rlt::init(device_compute, log_compute);
    episodes_compute.step_limit = STEP_LIMIT;
    world.history_step = 0;
    for(TI step_i = 0; step_i < STEPS; step_i++){
        if(step_i == 2){
            rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> mask_host;
            rlt::malloc(device, mask_host);
            rlt::set_all(device, mask_host, false);
            rlt::set(device, mask_host, true, 0);
            rlt::copy(device, device_compute, mask_host, mask);
            rlt::free(device, mask_host);
            rlt::force_reset(device_compute, episodes_compute, mask);
        }
        rlt::begin_step(device_compute, world, episodes_compute, parameters, states, rng);
        rlt::record(device_compute, log_compute, episodes_compute, step_i);
        if(step_i == 0){
            rlt::record_reset(device_compute, dataset_compute, episodes_compute.reset);
        }
        rlt::render(device_compute, world, parameters, states, episodes_compute.reset);
        rlt::copy(device_compute, device, episodes_compute, episodes_host);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            trace.reset[step_i][instance_i] = rlt::get(device, episodes_host.reset, instance_i);
        }
        rlt::step(device_compute, world, parameters, states, actions, next_states, rng);
        rlt::reward(device_compute, world, parameters, states, actions, next_states, rewards, rng);
        rlt::copy(device_compute, device_compute, next_states, states);
        rlt::end_step(device_compute, world, episodes_compute, parameters, states, rewards, rng);
        rlt::record_step(device_compute, dataset_compute, step_i, rewards, episodes_compute.terminated, episodes_compute.truncated);
        rlt::copy(device_compute, device, episodes_compute, episodes_host);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            trace.episode_step[step_i][instance_i] = rlt::get(device, episodes_host.episode_step, instance_i);
            trace.terminated[step_i][instance_i] = rlt::get(device, episodes_host.terminated, instance_i);
            trace.truncated[step_i][instance_i] = rlt::get(device, episodes_host.truncated, instance_i);
            trace.end_reason[step_i][instance_i] = rlt::get(device, episodes_host.end_reason, instance_i);
        }
    }
    rlt::copy(device_compute, device, log_compute, log_host);
    rlt::copy(device_compute, device, dataset_compute, dataset_host);
    for(TI step_i = 0; step_i < STEPS; step_i++){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const TI pos = step_i * INSTANCES + instance_i;
            trace.finished[step_i][instance_i] = rlt::get(device, log_host.finished, step_i, instance_i);
            trace.finished_length[step_i][instance_i] = rlt::get(device, log_host.finished_length, step_i, instance_i);
            trace.finished_return[step_i][instance_i] = rlt::get(device, log_host.finished_return, step_i, instance_i);
            trace.finished_reason[step_i][instance_i] = rlt::get(device, log_host.finished_reason, step_i, instance_i);
            trace.dataset_terminated[step_i][instance_i] = rlt::get(dataset_host.terminated, pos, 0);
            trace.dataset_truncated[step_i][instance_i] = rlt::get(dataset_host.truncated, pos, 0);
            trace.dataset_reset[step_i][instance_i] = rlt::get(dataset_host.reset, pos, 0);
        }
    }
    rlt::free(device_compute, parameters);
    rlt::free(device_compute, states);
    rlt::free(device_compute, next_states);
    rlt::free(device_compute, actions);
    rlt::free(device_compute, rewards);
    rlt::free(device_compute, mask);
    rlt::free(device_compute, episodes_compute);
    rlt::free(device_compute, log_compute);
    rlt::free(device_compute, dataset_compute);
    rlt::free(device, episodes_host);
    rlt::free(device, log_host);
    rlt::free(device, dataset_host);
    rlt::free(device_compute, rng);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_CUDA, CPU_CUDA_PARITY){
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
    rlt::malloc(device, world);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world, shared, dataset, corpus, 0, 1, 0);

    auto* trace_cpu = new Trace;
    auto* trace_cuda = new Trace;
    trace_rollout<DEVICE, RNG>(device, device, world, *trace_cpu, 1337);
    trace_rollout<DEVICE_GPU, RNG_GPU>(device, device_gpu, world, *trace_cuda, 1337);
    TI truncations = 0;
    for(TI step_i = 0; step_i < STEPS; step_i++){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            ASSERT_EQ(trace_cpu->reset[step_i][instance_i], trace_cuda->reset[step_i][instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(trace_cpu->episode_step[step_i][instance_i], trace_cuda->episode_step[step_i][instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(trace_cpu->terminated[step_i][instance_i], trace_cuda->terminated[step_i][instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(trace_cpu->truncated[step_i][instance_i], trace_cuda->truncated[step_i][instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(trace_cpu->end_reason[step_i][instance_i], trace_cuda->end_reason[step_i][instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(trace_cpu->finished[step_i][instance_i], trace_cuda->finished[step_i][instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(trace_cpu->finished_length[step_i][instance_i], trace_cuda->finished_length[step_i][instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(trace_cpu->finished_reason[step_i][instance_i], trace_cuda->finished_reason[step_i][instance_i]) << "step " << step_i << " instance " << instance_i;
            ASSERT_NEAR(trace_cpu->finished_return[step_i][instance_i], trace_cuda->finished_return[step_i][instance_i], 1e-4) << "step " << step_i << " instance " << instance_i;
            ASSERT_EQ(trace_cpu->dataset_terminated[step_i][instance_i], trace_cuda->dataset_terminated[step_i][instance_i]);
            ASSERT_EQ(trace_cpu->dataset_truncated[step_i][instance_i], trace_cuda->dataset_truncated[step_i][instance_i]);
            ASSERT_EQ(trace_cpu->dataset_reset[step_i][instance_i], trace_cuda->dataset_reset[step_i][instance_i]);
            truncations += trace_cuda->truncated[step_i][instance_i] ? 1 : 0;
        }
    }
    EXPECT_GE(truncations, INSTANCES) << "the step limit must have truncated every instance at least once";
    EXPECT_TRUE(trace_cuda->reset[2][0]) << "the forced instance resets at the next begin_step";
    delete trace_cpu;
    delete trace_cuda;
    rlt::free(device, world);
    rlt::free(device, shared.library);
}
