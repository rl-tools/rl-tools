#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>

#include <backprop_tools/rl/environments/car/operations_cpu.h>
#if BACKPROP_TOOLS_ENABLE_GTK
#include <backprop_tools/rl/environments/car/ui.h>
#endif

#include <backprop_tools/nn_models/operations_generic.h>

#include <backprop_tools/rl/algorithms/td3/loop.h>

struct TrainingConfig{
    using DEV_SPEC = bpt::devices::DefaultCPUSpecification;
//    using DEVICE = bpt::devices::CPU<DEV_SPEC>;
    using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
    using T = float;
    using TI = typename DEVICE::index_t;

    using ENV_SPEC = bpt::rl::environments::car::SpecificationTrack<T, DEVICE::index_t, 100, 100, 20>;
    using ENVIRONMENT = bpt::rl::environments::CarTrack<ENV_SPEC>;
#if BACKPROP_TOOLS_ENABLE_GTK
    using UI = bpt::rl::environments::car::UI<bpt::rl::environments::car::ui::Specification<T, TI, ENVIRONMENT, 1000, 60>>;
#else
    using UI = bool;
#endif

    struct DEVICE_SPEC: bpt::devices::DefaultCPUSpecification {
        using LOGGING = bpt::devices::logging::CPU;
    };
    struct TD3PendulumParameters: bpt::rl::algorithms::td3::DefaultParameters<T, DEVICE::index_t>{
        constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
        constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
        constexpr static T GAMMA = 0.997;
    };

    using TD3_PARAMETERS = TD3PendulumParameters;

    using ActorStructureSpec = bpt::nn_models::mlp::StructureSpecification<T, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
    using CriticStructureSpec = bpt::nn_models::mlp::StructureSpecification<T, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;


    using OPTIMIZER_PARAMETERS = typename bpt::nn::optimizers::adam::DefaultParametersTorch<T, typename DEVICE::index_t>;
    using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    using ACTOR_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<ActorStructureSpec>;
    using ACTOR_NETWORK_TYPE = bpt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

    using ACTOR_TARGET_NETWORK_SPEC = bpt::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
    using ACTOR_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

    using CRITIC_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<CriticStructureSpec>;
    using CRITIC_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

    using CRITIC_TARGET_NETWORK_SPEC = backprop_tools::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
    using CRITIC_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

    using ACTOR_CRITIC_SPEC = bpt::rl::algorithms::td3::Specification<T, DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, OPTIMIZER, TD3_PARAMETERS>;
    using ACTOR_CRITIC_TYPE = bpt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;


    static constexpr int N_WARMUP_STEPS = ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
#ifndef BACKPROP_TOOLS_STEP_LIMIT
    static constexpr DEVICE::index_t STEP_LIMIT = 500000000; //2 * N_WARMUP_STEPS;
#else
    static constexpr DEVICE::index_t STEP_LIMIT = BACKPROP_TOOLS_STEP_LIMIT;
#endif
    static constexpr bool DETERMINISTIC_EVALUATION = true;
    static constexpr DEVICE::index_t EVALUATION_INTERVAL = 20000;
    static constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 1000000;
    static constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 1000;
    static constexpr bool COLLECT_EPISODE_STATS = true;
    static constexpr DEVICE::index_t EPISODE_STATS_BUFFER_SIZE = 1000;
    using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<
            T,
            DEVICE::index_t,
            ENVIRONMENT,
            1,
            false,
            REPLAY_BUFFER_CAP,
            ENVIRONMENT_STEP_LIMIT,
            bpt::rl::components::off_policy_runner::DefaultParameters<T>,
            COLLECT_EPISODE_STATS,
            EPISODE_STATS_BUFFER_SIZE
    >;
    const T STATE_TOLERANCE = 0.00001;
    static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
};

int main(){
    using TI = typename TrainingConfig::TI;
    TrainingState<TrainingConfig> ts;
    bpt::init(ts.device, ts.envs[0]);
    training_init(ts, 3);
//    ts.envs[0].parameters.dt = 0.01;
    for(TI step_i=0; step_i < TrainingConfig::STEP_LIMIT; step_i++){
        training_step(ts);
    }
    return 0;
}