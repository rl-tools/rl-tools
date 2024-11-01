#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/multi_agent_wrapper/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/td3_sampling/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
#endif

#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/standardize/persist_code.h>
#include <rl_tools/nn/layers/sample_and_squash/persist_code.h>
#include <rl_tools/nn/layers/td3_sampling/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist_code.h>

#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_helper_generic.h>
#include <rl_tools/rl/environments/l2f/parameters/default.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/crazyflie.h>

#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>


#include <rl_tools/utils/generic/typing.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = typename DEVICE::index_t;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
constexpr TI BASE_SEED = 0;


constexpr bool IDENT = true;

constexpr static auto MODEL = rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie;

constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;
static constexpr auto reward_function = IDENT ? rl_tools::rl::environments::l2f::parameters::reward_functions::squared<T> : rl_tools::rl::environments::l2f::parameters::reward_functions::constant<T>;
using REWARD_FUNCTION_CONST = typename rl_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
using REWARD_FUNCTION = typename rl_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

using PARAMETERS_SPEC = rl_tools::rl::environments::l2f::ParametersBaseSpecification<T, TI, 4, REWARD_FUNCTION, rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY, MODEL>;
using PARAMETERS_TYPE = rl_tools::rl::environments::l2f::ParametersDisturbances<T, TI, rl_tools::rl::environments::l2f::ParametersBase<PARAMETERS_SPEC>>;

static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = rl_tools::rl::environments::l2f::parameters::dynamics::registry<PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::Integration integration = {
    0.01 // integration dt
};
static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = IDENT ? rl_tools::rl::environments::l2f::parameters::init::init_90_deg<PARAMETERS_SPEC> : rl_tools::rl::environments::l2f::parameters::init::init_0_deg<PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::MDP::ObservationNoise observation_noise = {
    0.0, // position
    0.00, // orientation
    0.0, // linear_velocity
    0.0, // angular_velocity
    0.0, // imu acceleration
};
//        static constexpr typename PARAMETERS_TYPE::MDP::ObservationNoise observation_noise = {
//                0.0, // position
//                0.0, // orientation
//                0.0, // linear_velocity
//                0.0, // angular_velocity
//                0.0, // imu acceleration
//        };
static constexpr typename PARAMETERS_TYPE::MDP::ActionNoise action_noise = {
    0, // std of additive gaussian noise onto the normalized action (-1, 1)
};
static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = rl_tools::rl::environments::l2f::parameters::termination::fast_learning<PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::MDP mdp = {
    init,
    reward_function,
    observation_noise,
    action_noise,
    termination
};
static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
    // 1.5, // thrust_to_weight_min;
    // 4.0, // thrust_to_weight_max;
    // 0.027, // mass_min;
    // 5.0, // mass_max;
    // 1.0, // torque_to_inertia;
    // 0.0, // mass_size_deviation;
    // 0.0, // motor_time_constant;
    // 0.0 // rotor_torque_constant;
    0.0, // thrust_to_weight_min;
    0.0, // thrust_to_weight_max;
    IDENT ? 0.0 : 0.02, // mass_min;
    0.035, // mass_max;
    0.0, // mass_size_deviation;
    0.0, // motor_time_constant;
    0.0 // rotor_torque_constant;
};
static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
    typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0}, // random_force;
    typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_torque;
};
static constexpr PARAMETERS_TYPE nominal_parameters = {
    {
        dynamics,
        integration,
        mdp,
        domain_randomization
    },
    disturbances
};

namespace static_builder{
    using namespace rl_tools::rl::environments::l2f;
    struct ENVIRONMENT_STATIC_PARAMETERS{
        static constexpr TI ACTION_HISTORY_LENGTH = 16;
        static constexpr TI CLOSED_FORM = false;
        using STATE_BASE = StateBase<T, TI>;
        using STATE_TYPE = StateRotorsHistory<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<T, TI, STATE_BASE>>;
        using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                        observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                                observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                                        observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = observation::Position<observation::PositionSpecificationPrivileged<T, TI,
                observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                        observation::LinearVelocity<observation::LinearVelocitySpecificationPrivileged<T, TI,
                                observation::AngularVelocity<observation::AngularVelocitySpecificationPrivileged<T, TI,
                                        observation::RandomForce<observation::RandomForceSpecification<T, TI,
                                                observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>
                                        >
                                        >
                                >>
                        >>
                >>
        >>;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = PARAMETERS_TYPE;
        static constexpr auto PARAMETER_VALUES = nominal_parameters;
    };
}

using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, static_builder::ENVIRONMENT_STATIC_PARAMETERS>;
using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;

struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<T, TI>{
        static constexpr TI ACTOR_BATCH_SIZE = 256;
        static constexpr TI CRITIC_BATCH_SIZE = 256;
        static constexpr TI TRAINING_INTERVAL = 10;
        static constexpr TI CRITIC_TRAINING_INTERVAL = 1 * TRAINING_INTERVAL;
        static constexpr TI ACTOR_TRAINING_INTERVAL = 2 * TRAINING_INTERVAL;
        static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 1 * TRAINING_INTERVAL;
        static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.9;
        static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.3;
        static constexpr T GAMMA = 0.99;
        static constexpr bool IGNORE_TERMINATION = false;
        static constexpr T TARGET_ENTROPY = -((T)4);
    };
    static constexpr TI STEP_LIMIT = 10000000;
    static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
//            static constexpr bool SHARED_BATCH = false;
    struct ACTOR_OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T> {
        static constexpr T ALPHA = 0.0001;
    };
    struct CRITIC_OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T> {
        static constexpr T ALPHA = 0.0001;
    };
    struct ALPHA_OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<T> {
        static constexpr T ALPHA = 0.001;
    };
    static constexpr bool SAMPLE_ENVIRONMENT_PARAMETERS = IDENT;
};

using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsSequential>;

constexpr TI NUM_CHECKPOINTS = 10;
constexpr TI NUM_EVALUATIONS = 100;
constexpr TI NUM_SAVE_TRAJECTORIES = 10;
using LOOP_EXTRACK_CONFIG = rlt::rl::loop::steps::extrack::Config<LOOP_CORE_CONFIG>;
struct LOOP_CHECKPOINT_PARAMETERS: rlt::rl::loop::steps::checkpoint::Parameters<T, TI>{
    static constexpr TI CHECKPOINT_INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_CHECKPOINTS;
    static constexpr TI CHECKPOINT_INTERVAL = CHECKPOINT_INTERVAL_TEMP == 0 ? 1 : CHECKPOINT_INTERVAL_TEMP;
};
using LOOP_CHECKPOINT_CONFIG = rlt::rl::loop::steps::checkpoint::Config<LOOP_EXTRACK_CONFIG, LOOP_CHECKPOINT_PARAMETERS>;
struct LOOP_EVALUATION_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>{
    static constexpr TI EVALUATION_INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_EVALUATIONS;
    static constexpr TI EVALUATION_INTERVAL = EVALUATION_INTERVAL_TEMP == 0 ? 1 : EVALUATION_INTERVAL_TEMP;
    static constexpr TI NUM_EVALUATION_EPISODES = 10;
    static constexpr TI N_EVALUATIONS = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using LOOP_EVALUATION_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CHECKPOINT_CONFIG, LOOP_EVALUATION_PARAMETERS>;
struct LOOP_SAVE_TRAJECTORIES_PARAMETERS: rlt::rl::loop::steps::save_trajectories::Parameters<T, TI, LOOP_CHECKPOINT_CONFIG>{
    static constexpr TI INTERVAL_TEMP = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT / NUM_SAVE_TRAJECTORIES;
    static constexpr TI INTERVAL = INTERVAL_TEMP == 0 ? 1 : INTERVAL_TEMP;
    static constexpr TI NUM_EPISODES = 10;
};
using LOOP_SAVE_TRAJECTORIES_CONFIG = rlt::rl::loop::steps::save_trajectories::Config<LOOP_EVALUATION_CONFIG, LOOP_SAVE_TRAJECTORIES_PARAMETERS>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_SAVE_TRAJECTORIES_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;

using LOOP_STATE = typename LOOP_CONFIG::State<LOOP_CONFIG>;

int main() {
    TI seed = 11;
    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, seed);
    LOOP_STATE ts;
    ts.extrack_name = "zoo";
    ts.extrack_population_variates = "algorithm_environment";
    ts.extrack_population_values = "dr-sac_l2f";
    rlt::malloc(device);
    rlt::init(device);
    rlt::malloc(device, ts);
    rlt::init(device, ts, seed);
#ifdef RL_TOOLS_ENABLE_TENSORBOARD
    rlt::init(device, device.logger, ts.extrack_seed_path);
#endif
    LOOP_CONFIG::ENVIRONMENT env;
    LOOP_CONFIG::ENVIRONMENT::Parameters env_parameters, env_parameters_nominal;
    rlt::initial_parameters(device, env, env_parameters_nominal);
    rlt::sample_initial_parameters(device, env, env_parameters, rng);
    std::string parameters_json = rlt::json(device, env, env_parameters);
    rlt::compare_parameters(device, env_parameters_nominal, env_parameters);
    rlt::set_parameters(device, ts.off_policy_runner, env_parameters);
    while(!rlt::step(device, ts)){
    }
}