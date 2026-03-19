#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn/layers/unflatten/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>

#include <utility>
#include <iostream>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<float>;
using TI = typename DEVICE::index_t;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;

namespace l2f = rlt::rl::environments::l2f;
namespace obs = l2f::observation;

// =========================================================================
// L2F dynamics configuration (no trajectory, simple state)
// =========================================================================
using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
static constexpr TI SIMULATION_FREQUENCY = 100;
static constexpr TI EPISODE_STEP_LIMIT = 500;
using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::crazyflie;

static constexpr REWARD_FUNCTION reward_function = {
    false, // non_negative
    0.10,  // scale
    1.00,  // constant
    0.00,  // termination_penalty
    10.00, // position
    0.00,  // position_clip
    2.50,  // orientation
    0.05,  // linear_velocity
    0.00,  // angular_velocity
    0.00,  // linear_acceleration
    0.00,  // angular_acceleration
    0.10,  // action
    0.00,  // d_action
    0.00   // position_error_integral
};

static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
    0.2,  // guidance (20% start exactly at target)
    1.0,  // max_position (1m radius)
    0.3,  // max_angle (~17 deg tilt)
    1.0,  // max_linear_velocity
    1.0,  // max_angular_velocity
    true, // relative_rpm
    -1,   // min_rpm
    +1,   // max_rpm
};

static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
    true,  // enabled
    1.5,   // position (wider than spawn to avoid instant term)
    10,    // linear_velocity
    35,    // angular_velocity
    10000, // position_integral
    50000, // orientation_integral
};

static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::Integration integration = {
    static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY)
};
static constexpr typename PARAMETERS_TYPE::MDP mdp = {
    init,
    reward_function,
    {}, // observation_noise (zeros)
    {}, // action_noise (zeros)
    termination
};
static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
    {0, 0}, // random_force
    {0, 0}  // random_torque
};
static constexpr PARAMETERS_TYPE nominal_parameters = {
    {dynamics, integration, mdp},
    disturbances
};

// =========================================================================
// Environment static parameters
// =========================================================================
static constexpr TI ACTION_HISTORY_LENGTH = 1;

struct STATIC_PARAMETERS {
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI CLOSED_FORM = false;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;

    using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;

    using OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI,
            obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
            obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
            obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI,
            obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;

    using OBSERVATION_TYPE_PRIVILEGED = obs::Position<obs::PositionSpecificationPrivileged<T, TI,
            obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecificationPrivileged<T, TI,
            obs::LinearVelocity<obs::LinearVelocitySpecificationPrivileged<T, TI,
            obs::AngularVelocity<obs::AngularVelocitySpecificationPrivileged<T, TI,
            obs::ActionHistory<obs::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;

    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
    using PARAMETERS = PARAMETERS_TYPE;
    static constexpr auto PARAMETER_VALUES = nominal_parameters;
    static constexpr T STATE_LIMIT_POSITION = 100000;
    static constexpr T STATE_LIMIT_VELOCITY = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
};

// =========================================================================
// Visual environment specification
// =========================================================================
static constexpr TI NUM_ENVS = 64;
static constexpr TI CAM_WIDTH = 64;
static constexpr TI CAM_HEIGHT = 64;
static constexpr TI NUM_PROBES = 64;

using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, STATIC_PARAMETERS, NUM_ENVS, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES>;
using ENVIRONMENT = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;

// =========================================================================
// PPO loop configuration
// =========================================================================
struct ADAM_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
    static constexpr T ALPHA = 3e-4;
    static constexpr T EPSILON = 1e-5;
    static constexpr T EPSILON_SQRT = 1e-5;
};

struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
    static constexpr TI BATCH_SIZE = 512;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 64;
    static constexpr TI N_ENVIRONMENTS = NUM_ENVS;
    static constexpr TI TOTAL_STEP_LIMIT = 10000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT;
    static constexpr TI EPISODE_STEP_LIMIT = ::EPISODE_STEP_LIMIT;
    using ACTOR_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
    using CRITIC_OPTIMIZER_PARAMETERS = ADAM_PARAMETERS;
    static constexpr bool NORMALIZE_OBSERVATIONS = true;
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01;
        static constexpr TI N_EPOCHS = 4;
        static constexpr T GAMMA = 0.99;
        static constexpr T LAMBDA = 0.95;
        static constexpr T EPSILON_CLIP = 0.2;
        static constexpr T INITIAL_ACTION_STD = 0.5;
    };
};

// CNN actor + dense critic (asymmetric observations)
template<typename T_TYPE_POLICY, typename T_TI, typename T_ENVIRONMENT, typename PARAMETERS, bool T_DYNAMIC_ALLOCATION = true>
struct ConfigApproximatorsCNN{
    static constexpr T_TI STEPS = 1;
    static constexpr T_TI FORWARD_BATCH_SIZE = PARAMETERS::BATCH_SIZE;
    template <typename CAPABILITY>
    struct Actor{
        using OBS_SHAPE = typename T_ENVIRONMENT::Observation::SHAPE;
        using INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
        static constexpr T_TI IMG_H = T_ENVIRONMENT::Observation::HEIGHT;
        static constexpr T_TI IMG_W = T_ENVIRONMENT::Observation::WIDTH;
        static constexpr T_TI IMG_C = T_ENVIRONMENT::Observation::CHANNELS;
        // Flatten(H,W,C -> H*W*C) -> Standardize -> Unflatten(H*W*C -> H,W,C) -> Conv1 -> Conv2 -> Flatten -> MLP
        using INPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, T_TI>;
        using INPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<INPUT_FLATTEN_CONFIG>;
        using STANDARDIZATION_LAYER_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
        using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
        using UNFLATTEN_CONFIG = rlt::nn::layers::unflatten::Configuration<T_TYPE_POLICY, T_TI, IMG_H, IMG_W, IMG_C>;
        using UNFLATTEN = rlt::nn::layers::unflatten::BindConfiguration<UNFLATTEN_CONFIG>;
        using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 16, 3, 3, 2, 2, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
        using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<T_TYPE_POLICY, T_TI, 32, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
        using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
        using OUTPUT_FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<T_TYPE_POLICY, T_TI>;
        using OUTPUT_FLATTEN = rlt::nn::layers::flatten::BindConfiguration<OUTPUT_FLATTEN_CONFIG>;
        using MLP_CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, T_ENVIRONMENT::ACTION_DIM, 2, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
        using MLP = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_CONFIG>;
        using MODULE_CHAIN = rlt::nn_models::sequential::Module<INPUT_FLATTEN, STANDARDIZATION_LAYER, UNFLATTEN, CONV1, CONV2, OUTPUT_FLATTEN, MLP>;
        using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
    };
    template <typename CAPABILITY>
    struct Critic{
        using OBS_PRIV_SHAPE = typename T_ENVIRONMENT::ObservationPrivileged::SHAPE;
        using INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, FORWARD_BATCH_SIZE>, STEPS>;
        using STANDARDIZATION_LAYER_CONFIG = rlt::nn::layers::standardize::Configuration<T_TYPE_POLICY, T_TI>;
        using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
        using CONFIG = rlt::nn_models::mlp::Configuration<T_TYPE_POLICY, T_TI, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY>;
        using TYPE = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<CONFIG>;
        using MODULE_CHAIN = rlt::nn_models::sequential::Module<STANDARDIZATION_LAYER, rlt::nn_models::sequential::Module<TYPE>>;
        using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
    };
    using ACTOR_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T_TYPE_POLICY, T_TI, typename PARAMETERS::ACTOR_OPTIMIZER_PARAMETERS, T_DYNAMIC_ALLOCATION>;
    using CRITIC_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T_TYPE_POLICY, T_TI, typename PARAMETERS::CRITIC_OPTIMIZER_PARAMETERS, T_DYNAMIC_ALLOCATION>;
    using ACTOR_OPTIMIZER = rlt::nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
    using CRITIC_OPTIMIZER = rlt::nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
    using CAPABILITY_ADAM = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, T_DYNAMIC_ALLOCATION>;
    using ACTOR_TYPE = typename Actor<CAPABILITY_ADAM>::MODEL;
    using CRITIC_TYPE = typename Critic<CAPABILITY_ADAM>::MODEL;
};

using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximatorsCNN>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_CORE_CONFIG>;
using LOOP_EXTRACK_CONFIG = rlt::rl::loop::steps::extrack::Config<LOOP_TIMING_CONFIG>;
using LOOP_CONFIG = LOOP_EXTRACK_CONFIG;
using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

// =========================================================================
// Main
// =========================================================================
int main(int argc, char** argv){
    const char* scene_path = nullptr;
    if (argc > 1) {
        scene_path = argv[1];
    }

    DEVICE device;
    LOOP_STATE ts;
    TI seed = 0;

    // 1. Allocate (each env gets its own renderer+scene)
    rlt::malloc(device, ts);

    // 2. Post-fixup: share env[0]'s renderer+scene with env[1..N-1]
    auto& env0 = rlt::get_ref(device, ts.envs, static_cast<TI>(0));
    for (TI env_i = 1; env_i < NUM_ENVS; env_i++) {
        auto& env = rlt::get_ref(device, ts.envs, env_i);
        // Free independently allocated renderer+scene
        if (env.owns_renderer && env.renderer != nullptr) {
            rlt::free(device, *env.renderer);
            delete env.renderer;
        }
        if (env.scene != nullptr) {
            delete env.scene;
        }
        // Point to env[0]'s shared instances
        env.renderer = env0.renderer;
        env.scene = env0.scene;
        env.owns_renderer = false;
    }

    // 3. Configure all envs
    for (TI env_i = 0; env_i < NUM_ENVS; env_i++) {
        auto& env = rlt::get_ref(device, ts.envs, env_i);
        env.use_target_mode = true;
        env.scene_path = scene_path;
        if (env_i > 0) {
            env.renderer_initialized = true; // only env[0] will do full init
        }
    }

    // 4. Init (env[0] loads scene, builds pipeline, precomputes indoor positions; others skip)
    rlt::init(device, ts, seed);

    // 5. Pick target position from precomputed indoor positions
    if (env0.scene->num_indoor_positions > 0) {
        auto& target = env0.scene->indoor_positions[0];
        T target_translation[3] = {
            target.position[0],
            target.position[1] + env0.eye_height,
            target.position[2]
        };
        rlt::log(device, device.logger, "Target scene position: [",
            target_translation[0], ", ", target_translation[1], ", ", target_translation[2], "]");
        for (TI env_i = 0; env_i < NUM_ENVS; env_i++) {
            auto& env = rlt::get_ref(device, ts.envs, env_i);
            for (TI j = 0; j < 3; j++) {
                env.target_scene_translation[j] = target_translation[j];
            }
        }
    }

    // 6. Training loop
    rlt::log(device, device.logger, "Starting PPO training (visual L2F hover)");
    rlt::log(device, device.logger, "  N_ENVIRONMENTS: ", NUM_ENVS);
    rlt::log(device, device.logger, "  STEP_LIMIT: ", LOOP_CORE_PARAMETERS::STEP_LIMIT);
    rlt::log(device, device.logger, "  OBSERVATION_DIM (image): ", ENVIRONMENT::OBSERVATION_DIM);
    rlt::log(device, device.logger, "  OBSERVATION_DIM_PRIVILEGED: ", ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED);

    while (!rlt::step(device, ts)) {
    }

    rlt::log(device, device.logger, "Training finished at step ", ts.step);

    // 7. Cleanup: detach shared renderer+scene from env[1..N-1] before free
    for (TI env_i = 1; env_i < NUM_ENVS; env_i++) {
        auto& env = rlt::get_ref(device, ts.envs, env_i);
        env.renderer = nullptr;
        env.scene = nullptr;
    }
    rlt::free(device, ts);

    return 0;
}
