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
    using POLICY_STATE = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1>>>;
    using RUNNER_SPEC = on_policy_runner::Specification<rlt::numeric_types::Policy<T>, WORLD, POLICY_STATE>;
    using RUNNER = rlt::rl::components::OnPolicyRunner<RUNNER_SPEC>;
    using BUFFER = on_policy_runner::Buffer<RUNNER_SPEC>;
    using DATASET_SPEC = on_policy_runner::DatasetSpecification<RUNNER_SPEC, STEPS>;
    using DATASET = on_policy_runner::Dataset<DATASET_SPEC>;
}

using namespace test_hyperdrone_episodes_cuda;

static bool cuda_available(){
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

struct Trace {
    bool reset[STEPS][INSTANCES];
    TI episode_step[STEPS][INSTANCES];
    bool terminated[STEPS][INSTANCES];
    T truncated[STEPS][INSTANCES];
    END_REASON reason[STEPS][INSTANCES];
    TI length[STEPS][INSTANCES];
    T episode_return[STEPS][INSTANCES];
    T dataset_reset[STEPS][INSTANCES];
    END_REASON forced_reason[INSTANCES];
};

template <typename COMPUTE_DEVICE, typename COMPUTE_RNG>
static void trace_rollout(DEVICE& device, COMPUTE_DEVICE& device_compute, WORLD& world, Trace& trace, TI seed){
    COMPUTE_RNG rng;
    RUNNER runner;
    BUFFER buffer;
    DATASET dataset_compute, dataset_host;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> mask, mask_host, terminated_host, reset_host;
    rlt::Tensor<rlt::tensor::Specification<TI, TI, rlt::tensor::Shape<TI, INSTANCES>>> episode_step_host;
    rlt::malloc(device_compute, rng);
    rlt::malloc(device_compute, runner);
    rlt::malloc(device_compute, buffer);
    rlt::malloc(device_compute, dataset_compute);
    rlt::malloc(device_compute, mask);
    rlt::malloc(device, mask_host);
    rlt::malloc(device, terminated_host);
    rlt::malloc(device, reset_host);
    rlt::malloc(device, episode_step_host);
    rlt::malloc(device, dataset_host);
    rlt::init(device_compute, rng, seed);
    rlt::set_all(device_compute, buffer.actions, (T)0);
    rlt::init(device_compute, runner, world, rng);
    runner.episode_step_limit = STEP_LIMIT;
    on_policy_runner::prologue(device_compute, dataset_compute, runner, world, rng);

    for(TI step_i = 0; step_i < STEPS; step_i++){
        if(step_i == 2){
            rlt::set_all(device, mask_host, false);
            rlt::set(device, mask_host, true, 0);
            rlt::copy(device, device_compute, mask_host, mask);
            on_policy_runner::reset(device_compute, runner, world, mask, rng);
            on_policy_runner::prologue(device_compute, dataset_compute, runner, world, rng);
        }
        rlt::copy(device_compute, device, runner.reset, reset_host);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            trace.reset[step_i][instance_i] = rlt::get(device, reset_host, instance_i);
        }
        on_policy_runner::epilogue(device_compute, dataset_compute, runner, buffer, world, rng, step_i);
        rlt::copy(device_compute, device, runner.episode_step, episode_step_host);
        rlt::copy(device_compute, device, buffer.terminated, terminated_host);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            trace.episode_step[step_i][instance_i] = rlt::get(device, episode_step_host, instance_i);
            trace.terminated[step_i][instance_i] = rlt::get(device, terminated_host, instance_i);
        }
    }

    rlt::copy(device_compute, device, dataset_compute.scalar_data, dataset_host.scalar_data);
    rlt::copy(device_compute, device, dataset_compute.episode_end_reason, dataset_host.episode_end_reason);
    rlt::copy(device_compute, device, dataset_compute.episode_length, dataset_host.episode_length);
    rlt::copy(device_compute, device, dataset_compute.episode_return, dataset_host.episode_return);
    for(TI step_i = 0; step_i < STEPS; step_i++){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const TI pos = step_i * INSTANCES + instance_i;
            trace.truncated[step_i][instance_i] = rlt::get(dataset_host.truncated, pos, 0);
            trace.reason[step_i][instance_i] = rlt::get(device, dataset_host.episode_end_reason, step_i + 1, instance_i);
            trace.length[step_i][instance_i] = rlt::get(device, dataset_host.episode_length, step_i + 1, instance_i);
            trace.episode_return[step_i][instance_i] = rlt::get(device, dataset_host.episode_return, step_i + 1, instance_i);
            trace.dataset_reset[step_i][instance_i] = rlt::get(dataset_host.reset, pos, 0);
        }
    }
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        trace.forced_reason[instance_i] = rlt::get(device, dataset_host.episode_end_reason, 0, instance_i);
    }

    rlt::free(device_compute, rng);
    rlt::free(device_compute, runner);
    rlt::free(device_compute, buffer);
    rlt::free(device_compute, dataset_compute);
    rlt::free(device_compute, mask);
    rlt::free(device, mask_host);
    rlt::free(device, terminated_host);
    rlt::free(device, reset_host);
    rlt::free(device, episode_step_host);
    rlt::free(device, dataset_host);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_CUDA, CPU_CUDA_PARITY){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    if(!cuda_available()){
        GTEST_SKIP() << "CUDA device unavailable";
    }
    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device);
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

    Trace trace_cpu{}, trace_cuda{};
    trace_rollout<DEVICE, RNG>(device, device, world_cpu, trace_cpu, 1337);
    trace_rollout<DEVICE_GPU, RNG_GPU>(device, device_gpu, world_gpu, trace_cuda, 1337);
    TI truncations = 0;
    for(TI step_i = 0; step_i < STEPS; step_i++){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            ASSERT_EQ(trace_cpu.reset[step_i][instance_i], trace_cuda.reset[step_i][instance_i]);
            ASSERT_EQ(trace_cpu.episode_step[step_i][instance_i], trace_cuda.episode_step[step_i][instance_i]);
            ASSERT_EQ(trace_cpu.terminated[step_i][instance_i], trace_cuda.terminated[step_i][instance_i]);
            ASSERT_EQ(trace_cpu.truncated[step_i][instance_i], trace_cuda.truncated[step_i][instance_i]);
            ASSERT_EQ(trace_cpu.reason[step_i][instance_i], trace_cuda.reason[step_i][instance_i]);
            ASSERT_EQ(trace_cpu.length[step_i][instance_i], trace_cuda.length[step_i][instance_i]);
            ASSERT_NEAR(trace_cpu.episode_return[step_i][instance_i], trace_cuda.episode_return[step_i][instance_i], 1e-4);
            ASSERT_EQ(trace_cpu.dataset_reset[step_i][instance_i], trace_cuda.dataset_reset[step_i][instance_i]);
            truncations += trace_cuda.truncated[step_i][instance_i] > (T)0.5 ? 1 : 0;
        }
    }
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        ASSERT_EQ(trace_cpu.forced_reason[instance_i], trace_cuda.forced_reason[instance_i]);
    }
    EXPECT_EQ(trace_cuda.forced_reason[0], END_REASON::FORCED);
    EXPECT_GE(truncations, INSTANCES);
    EXPECT_TRUE(trace_cuda.reset[2][0]);

    rlt::free(device, world_cpu);
    rlt::free(device_gpu, world_gpu);
    rlt::free(device, shared.library);
}
