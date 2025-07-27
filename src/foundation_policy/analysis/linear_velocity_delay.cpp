#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/td3_sampling/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/multi_agent_wrapper/operations_generic.h>

// #include <rl_tools/containers/tensor/persist.h>
// #include <rl_tools/nn/layers/sample_and_squash/persist.h>
// #include <rl_tools/nn/layers/dense/persist.h>
// #include <rl_tools/nn/layers/standardize/persist.h>
// #include <rl_tools/nn/layers/gru/persist.h>
// #include <rl_tools/nn/layers/td3_sampling/persist.h>
// #include <rl_tools/nn_models/mlp/persist.h>
// #include <rl_tools/nn_models/sequential/persist.h>
// #include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
// #include <rl_tools/rl/components/replay_buffer/persist.h>

#include <rl_tools/rl/environments/l2f/operations_generic.h>

#include <rl_tools/rl/utils/evaluation/operations_cpu.h>

#include "rl_tools/rl/environments/multi_agent/environments.h"

namespace rlt = rl_tools;

#include "../blob/checkpoint.h"
#include "../post_training/environment.h"


using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_PARAMS_DEVICE = rlt::devices::random::Generic<DEVICE::SPEC::MATH>;
using RNG_PARAMS = RNG_PARAMS_DEVICE::ENGINE<>;
using TI = DEVICE::index_t;
using T = float;

struct OPTIONS {
    static constexpr bool RANDOMIZE_MOTOR_MAPPING = false;
    static constexpr bool RANDOMIZE_THRUST_CURVES = false;
    static constexpr bool OBSERVE_THRASH_MARKOV = false;
    static constexpr bool MOTOR_DELAY = true;
    static constexpr bool ACTION_HISTORY = true;
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr bool OBSERVATION_NOISE = true;
};







namespace builder{
    // copied from ../post_training/environment.h
    // added linear velocity delay
    using SUPER = typename builder::ENVIRONMENT_FACTORY_POST_TRAINING<DEVICE, T, TI, OPTIONS>;
    using BASE_ENV = typename SUPER::ENVIRONMENT;

    using PARAMETERS_TYPE = ParametersObservationDelay<ParametersObservationDelaySpecification<T, TI, BASE_ENV::Parameters>>;

    struct ENVIRONMENT_STATIC_PARAMETERS{
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI EPISODE_STEP_LIMIT = 5 * SUPER::BASE_ENV::SIMULATION_FREQUENCY;
        static constexpr TI CLOSED_FORM = false;
        static constexpr bool RANDOMIZE_THRUST_CURVES = OPTIONS::RANDOMIZE_THRUST_CURVES;
        static constexpr bool RANDOMIZE_MOTOR_MAPPING = OPTIONS::RANDOMIZE_MOTOR_MAPPING;
        static constexpr bool OBSERVE_THRUST_CURVES = OPTIONS::RANDOMIZE_THRUST_CURVES && OPTIONS::OBSERVE_THRASH_MARKOV;
        static constexpr bool OBSERVE_MOTOR_POSITIONS = OPTIONS::RANDOMIZE_MOTOR_MAPPING && OPTIONS::OBSERVE_THRASH_MARKOV;
        static_assert(OPTIONS::ACTION_HISTORY_LENGTH >= 1);
        static constexpr TI ANGULAR_VELOCITY_DELAY = 0; // one step at 100hz = 10ms ~ delay from IMU to input to the policy: 1.3ms time constant of the IIR in the IMU (bw ~110Hz) + synchronization delay (2ms) + (negligible SPI transfer latency due to it being interrupt-based) + 1ms sensor.c RTOS loop @ 1khz + 2ms for the RLtools loop
        static constexpr TI LINEAR_VELOCITY_DELAY = 10; // one step at 100hz = 10ms ~ delay from IMU to input to the policy: 1.3ms time constant of the IIR in the IMU (bw ~110Hz) + synchronization delay (2ms) + (negligible SPI transfer latency due to it being interrupt-based) + 1ms sensor.c RTOS loop @ 1khz + 2ms for the RLtools loop
        using STATE_BASE = StateLinearVelocityDelay<StateLinearVelocityDelaySpecification<T, TI, LINEAR_VELOCITY_DELAY, StateAngularVelocityDelay<StateAngularVelocityDelaySpecification<T, TI, ANGULAR_VELOCITY_DELAY, StateLastAction<StateSpecification<T, TI, StateBase<StateSpecification<T, TI>>>>>>>>;
        using STATE_TYPE_MOTOR_DELAY = StateTrajectory<StateSpecification<T, TI, StateRotorsHistory<StateRotorsHistorySpecification<T, TI, OPTIONS::ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>>>;
        using STATE_TYPE_NO_MOTOR_DELAY = StateRandomForce<StateSpecification<T, TI, STATE_BASE>>;
        using STATE_TYPE = rl_tools::utils::typing::conditional_t<OPTIONS::MOTOR_DELAY, STATE_TYPE_MOTOR_DELAY, STATE_TYPE_NO_MOTOR_DELAY>;
        static_assert(!OPTIONS::ACTION_HISTORY || OPTIONS::MOTOR_DELAY, "Action history implies motor delay");
        using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                observation::LinearVelocityDelayed<observation::LinearVelocityDelayedSpecification<T, TI, LINEAR_VELOCITY_DELAY,
                observation::AngularVelocityDelayed<observation::AngularVelocityDelayedSpecification<T, TI, ANGULAR_VELOCITY_DELAY,
                observation::ActionHistory<observation::ActionHistorySpecification<T, TI, OPTIONS::ACTION_HISTORY ? OPTIONS::ACTION_HISTORY_LENGTH : 1 // one-step action history to Markovify the d_action regularization
        >>>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = PARAMETERS_TYPE;
        static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {
            SUPER::BASE_ENV::nominal_parameters,
            (TI)0,
            (TI)0,
        };
        static constexpr T STATE_LIMIT_POSITION = 100000;
        static constexpr T STATE_LIMIT_VELOCITY = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
    };

    using ENVIRONMENT_SPEC = Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
    using ENVIRONMENT = rl::environments::Multirotor<ENVIRONMENT_SPEC>;
}

using ENVIRONMENT = builder::ENVIRONMENT;

constexpr TI NUM_EPISODES_EVAL = 100;


using EVAL_ACTOR = rlt::checkpoint::actor::TYPE;

int main(int argc, char** argv){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    TI seed = 0;
    rlt::init(device, rng, seed);
    ENVIRONMENT env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    // env.parameters.mdp.init.max_angular_velocity = 0;
    // env.parameters.mdp.init.max_angle = 0;
    env.parameters.mdp.init.max_linear_velocity = 0.2;

    using EVAL_SPEC = rlt::rl::utils::evaluation::Specification<T, TI, ENVIRONMENT, NUM_EPISODES_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>;
    rlt::rl::utils::evaluation::Result<EVAL_SPEC> result;
    rlt::rl::utils::evaluation::Data<rlt::rl::utils::evaluation::DataSpecification<EVAL_SPEC>> data;
    RNG rng_copy = rng;
    rlt::malloc(device, data);
    rlt::Mode<rlt::mode::Evaluation<rlt::nn::layers::sample_and_squash::mode::DisableEntropy<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Final>>>> mode;
    rlt::rl::environments::DummyUI ui;

    T original_thrust_to_weight_ratio = 0;
    for (TI motor_i = 0; motor_i < 4; ++motor_i) {
        T thrust = 0;
        for (TI coeff = 0; coeff < 3; ++coeff) {
            thrust += env.parameters.dynamics.rotor_thrust_coefficients[motor_i][coeff] * std::pow(env.parameters.dynamics.action_limit.max, coeff);
        }
        original_thrust_to_weight_ratio += thrust;
    }
    original_thrust_to_weight_ratio /= env.parameters.dynamics.mass * 9.81;
    T original_mass = env.parameters.dynamics.mass;

    std::vector<T> test_t2w(19);
    std::generate(test_t2w.begin(), test_t2w.end(), [n = 0]() mutable { return 1.25 + 0.25 * n++; });

    static constexpr TI MAX_LINEAR_VELOCITY_DELAY = 20;

    for (auto t2w : test_t2w){
        T ratio = t2w / original_thrust_to_weight_ratio;
        env.parameters.dynamics.mass = original_mass / ratio;
        for (TI linear_velocity_delay = 0; linear_velocity_delay < MAX_LINEAR_VELOCITY_DELAY; ++linear_velocity_delay){
            env.parameters.linear_velocity_observation_delay = linear_velocity_delay;
            static constexpr bool DYNAMIC_ALLOCATION = true;
            using ADJUSTED_POLICY = typename EVAL_ACTOR::template CHANGE_BATCH_SIZE<TI, EVAL_SPEC::N_EPISODES>;
            ADJUSTED_POLICY::template State<DYNAMIC_ALLOCATION> policy_state;
            ADJUSTED_POLICY::template Buffer<DYNAMIC_ALLOCATION> policy_evaluation_buffers;
            rlt::rl::utils::evaluation::Buffer<rlt::rl::utils::evaluation::BufferSpecification<EVAL_SPEC, DYNAMIC_ALLOCATION>> evaluation_buffers;
            rlt::malloc(device, policy_state);
            rlt::malloc(device, policy_evaluation_buffers);
            rlt::malloc(device, evaluation_buffers);
            rlt::evaluate(device, env, ui, rlt::checkpoint::actor::module, policy_state, policy_evaluation_buffers, evaluation_buffers, result, data, rng, mode);
            T action_mean = 0;
            T action_std = 0;
            TI num_action_values = 0;
            T speed_mean = 0;
            TI num_speed_values = 0;
            for (TI episode_i = 0; episode_i < EVAL_SPEC::N_EPISODES; ++episode_i) {
                if (result.episode_length[episode_i] == EVAL_SPEC::STEP_LIMIT){
                    for (TI step_i = EVAL_SPEC::STEP_LIMIT/2; step_i < EVAL_SPEC::STEP_LIMIT; ++step_i){
                        for (TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; ++action_i){
                            T action = rlt::get(device, data.actions, episode_i, step_i, action_i);
                            action_mean += action;
                            action_std += action * action;
                            num_action_values += 1;
                        }
                        const auto& state = rlt::get(device, data.states, episode_i, step_i);
                        T speed = 0;
                        for (TI axis_i=0; axis_i < 3; axis_i++){
                            speed += state.linear_velocity[axis_i] * state.linear_velocity[axis_i];
                        }
                        speed = std::sqrt(speed);
                        speed_mean += speed;
                        num_speed_values += 1;
                    }
                }
            }
            if (num_action_values > 0) {
                action_mean /= num_action_values;
                action_std = std::sqrt(std::max((T)0, action_std / num_action_values - action_mean * action_mean));
            } else {
                action_mean = 0;
                action_std = 0;
            }
            if (num_speed_values > 0) {
                speed_mean /= num_speed_values;
            } else {
                speed_mean = 0;
            }
            std::cout << "T2W: " << t2w << " LV delay: " << linear_velocity_delay << " Episode lengths mean: " << result.episode_length_mean << ", Returns mean: " << result.returns_mean << " action std: " << action_std << " speed mean: " << speed_mean << std::endl;
            rlt::free(device, policy_state);
            rlt::free(device, policy_evaluation_buffers);
            rlt::free(device, evaluation_buffers);
        }
    }
}
