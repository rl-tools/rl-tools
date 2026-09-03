#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cuda.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cuda.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <string>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;
namespace on_policy_runner = rlt::rl::components::on_policy_runner;

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
    constexpr TI STEPS = 7;
    constexpr TI STEP_LIMIT = 3;
    using END_REASON = on_policy_runner::EpisodeEndReason;
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

using POLICY_STATE = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1>>>;
using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<rlt::numeric_types::Policy<T>, WORLD, POLICY_STATE>;
using ON_POLICY_RUNNER = rlt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
using DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, STEPS>;
using DATASET = rlt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;

// the same verb sequence on either device; forced reset (instance 0) before step 2
template <typename COMPUTE_DEVICE, typename COMPUTE_RNG>
static void trace_rollout(DEVICE& device, COMPUTE_DEVICE& device_compute, WORLD& world, Trace& trace, TI seed){
    using LOG = on_policy_runner::EpisodeLog<ON_POLICY_RUNNER_SPEC, STEPS>;
    COMPUTE_RNG rng;
    rlt::malloc(device_compute, rng);
    rlt::init(device_compute, rng, seed);
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> next_states;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES>>> rewards;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> mask;
    ON_POLICY_RUNNER runner_compute, runner_host;
    LOG log_compute, log_host;
    DATASET dataset_compute, dataset_host;
    rlt::malloc(device_compute, next_states);
    rlt::malloc(device_compute, actions);
    rlt::malloc(device_compute, rewards);
    rlt::malloc(device_compute, mask);
    rlt::malloc(device_compute, runner_compute);
    rlt::malloc(device_compute, log_compute);
    rlt::malloc(device_compute, dataset_compute);
    rlt::malloc(device, runner_host.reset);
    rlt::malloc(device, runner_host.episode_step);
    rlt::malloc(device, runner_host.terminated);
    rlt::malloc(device, runner_host.truncated);
    rlt::malloc(device, runner_host.end_reason);
    rlt::malloc(device, log_host);
    rlt::malloc(device, dataset_host);
    rlt::set_all(device_compute, actions, (T)0);
    rlt::set_all(device_compute, mask, false);
    rlt::init(device_compute, log_compute);
    world.history_step = 0;
    rlt::init(device_compute, runner_compute, world, rng);
    runner_compute.episode_step_limit = STEP_LIMIT;
    for(TI step_i = 0; step_i < STEPS; step_i++){
        if(step_i == 2){
            rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> mask_host;
            rlt::malloc(device, mask_host);
            rlt::set_all(device, mask_host, false);
            rlt::set(device, mask_host, true, 0);
            rlt::copy(device, device_compute, mask_host, mask);
            rlt::free(device, mask_host);
            rlt::force_reset(device_compute, runner_compute, mask);
        }
        if(step_i > 0){
            rlt::begin_step(device_compute, runner_compute, world, rng);
        }
        rlt::record(device_compute, log_compute, runner_compute, step_i);
        if(step_i == 0){
            rlt::record_reset(device_compute, dataset_compute, runner_compute.reset);
        }
        rlt::rl::components::on_policy_runner::render(device_compute, runner_compute, world);
        rlt::copy(device_compute, device, runner_compute.reset, runner_host.reset);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            trace.reset[step_i][instance_i] = rlt::get(device, runner_host.reset, instance_i);
        }
        rlt::step(device_compute, world, runner_compute.env_parameters, runner_compute.states, actions, next_states, rng);
        rlt::reward(device_compute, world, runner_compute.env_parameters, runner_compute.states, actions, next_states, rewards, rng);
        rlt::copy(device_compute, device_compute, next_states, runner_compute.states);
        rlt::end_step(device_compute, runner_compute, world, rewards, rng);
        rlt::record_step(device_compute, dataset_compute, step_i, rewards, runner_compute.terminated, runner_compute.truncated);
        rlt::copy(device_compute, device, runner_compute.episode_step, runner_host.episode_step);
        rlt::copy(device_compute, device, runner_compute.terminated, runner_host.terminated);
        rlt::copy(device_compute, device, runner_compute.truncated, runner_host.truncated);
        rlt::copy(device_compute, device, runner_compute.end_reason, runner_host.end_reason);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            trace.episode_step[step_i][instance_i] = rlt::get(device, runner_host.episode_step, instance_i);
            trace.terminated[step_i][instance_i] = rlt::get(device, runner_host.terminated, instance_i);
            trace.truncated[step_i][instance_i] = rlt::get(device, runner_host.truncated, instance_i);
            trace.end_reason[step_i][instance_i] = rlt::get(device, runner_host.end_reason, instance_i);
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
    rlt::free(device_compute, next_states);
    rlt::free(device_compute, actions);
    rlt::free(device_compute, rewards);
    rlt::free(device_compute, mask);
    rlt::free(device_compute, runner_compute);
    rlt::free(device_compute, log_compute);
    rlt::free(device_compute, dataset_compute);
    rlt::free(device, runner_host.reset);
    rlt::free(device, runner_host.episode_step);
    rlt::free(device, runner_host.terminated);
    rlt::free(device, runner_host.truncated);
    rlt::free(device, runner_host.end_reason);
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
