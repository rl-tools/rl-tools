// this is a test to check if everything compiles without any dependencies (by replacing the dependency based math functions with dummy implementations)
#ifdef LAYER_IN_C_OPERATIONS_CPU
#include <layer_in_c/operations/cpu.h>
#else
#include <layer_in_c/operations/dummy.h>
#endif

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/rl/environments/operations_generic.h>
#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/rl/rl.h>
#include <layer_in_c/rl/operations_generic.h>



#include <layer_in_c/rl/utils/evaluation.h>


#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
#include <layer_in_c/rl/environments/pendulum/ui.h>
#include <layer_in_c/rl/utils/evaluation_visual.h>
#endif


#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
#include "plot_policy_and_value_function.h"
#endif


namespace lic = layer_in_c;
using DTYPE = float;

#ifdef LAYER_IN_C_OPERATIONS_CPU
using DEVICE = lic::devices::DefaultCPU;
using NN_DEVICE = lic::devices::DefaultCPU;
using AC_DEVICE = lic::devices::DefaultCPU;
#else
using DEVICE = lic::devices::DefaultDummy;
using NN_DEVICE = lic::devices::DefaultDummy;
using AC_DEVICE = lic::devices::DefaultDummy;
#endif
typedef lic::rl::environments::pendulum::Specification<DTYPE, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::Pendulum<DEVICE, PENDULUM_SPEC> ENVIRONMENT;
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
typedef lic::rl::environments::pendulum::UI<DTYPE> UI;
#endif
ENVIRONMENT env;

//struct ActorStructureSpec{
//    using T = DTYPE;
//    static constexpr typename DEVICE::index_t INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM;
//    static constexpr typename DEVICE::index_t OUTPUT_DIM = ENVIRONMENT::ACTION_DIM;
//    static constexpr int NUM_LAYERS = 3;
//    static constexpr int HIDDEN_DIM = 64;
//    static constexpr lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = lic::nn::activation_functions::RELU;
//    static constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = lic::nn::activation_functions::TANH;
//};

//struct CriticStructureSpec{
//    using T = DTYPE;
//    static constexpr typename DEVICE::index_t INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM;
//    static constexpr typename DEVICE::index_t OUTPUT_DIM = 1;
//    static constexpr int NUM_LAYERS = 3;
//    static constexpr int HIDDEN_DIM = 64;
//    static constexpr lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = lic::nn::activation_functions::RELU;
//    static constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = lic::nn::activation_functions::IDENTITY;
//};

using ActorStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH>;
using CriticStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY>;

template <typename T>
struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<T, AC_DEVICE::index_t>{
    constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3PendulumParameters<DTYPE>>;
using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<TD3_SPEC>;


constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 500000;
constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 200;
AC_DEVICE::SPEC::LOGGING logger;
AC_DEVICE device(logger);
NN_DEVICE nn_device(logger);
lic::rl::components::OffPolicyRunner<
        lic::rl::components::off_policy_runner::Specification<
                DTYPE,
                AC_DEVICE::index_t,
                ENVIRONMENT,
                REPLAY_BUFFER_CAP,
                ENVIRONMENT_STEP_LIMIT,
                lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>
        >
> off_policy_runner;
ActorCriticType actor_critic;
const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

int main() {
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
    UI ui;
#endif
    lic::malloc(device, actor_critic);
    auto rng = lic::random::default_engine(decltype(device)::SPEC::RANDOM());
    lic::init(device, actor_critic, rng);

    for(int step_i = 0; step_i < 15000; step_i++){
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
        if(step_i % 20 == 0){
            plot_policy_and_value_function<DTYPE, ENVIRONMENT, decltype(actor_critic.actor), decltype(actor_critic.critic_1)>(actor_critic.actor, actor_critic.critic_1, std::string("full_training"), step_i);
        }
#endif
        if(step_i > REPLAY_BUFFER_CAP){
            lic::logging::text(device.logger, "warning: replay buffer is rolling over");
        }
        lic::step(device, off_policy_runner, actor_critic.actor, rng);
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
        lic::set_state(ui, off_policy_runner.state);
#endif

        if(off_policy_runner.replay_buffer.full || off_policy_runner.replay_buffer.position > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                lic::logging::text(device.logger, "step_i: ", step_i);
            }
            DTYPE critic_1_loss = lic::train_critic(device, actor_critic, actor_critic.critic_1, off_policy_runner.replay_buffer, rng);
            lic::train_critic(device, actor_critic, actor_critic.critic_2, off_policy_runner.replay_buffer, rng);
//            std::cout << "Critic 1 loss: " << critic_1_loss << std::endl;
            if(step_i % 2 == 0){
                lic::train_actor(device, actor_critic, off_policy_runner.replay_buffer, rng);
                lic::update_targets(device, actor_critic);
            }
        }
        if(step_i % 1000 == 0){
            DTYPE mean_return = lic::evaluate<AC_DEVICE, ENVIRONMENT, decltype(actor_critic.actor), typeof(rng), ENVIRONMENT_STEP_LIMIT, true>(device, env, actor_critic.actor, 1, rng);
            lic::logging::text(logger, "Mean return: ", mean_return);
#ifdef LAYER_IN_C_CONTEXT_CPU
            if(mean_return > -400){
                return 0;
            }
#else
            if(mean_return > -200000){
                return 0;
            }
#endif
//            if(step_i >= 6000){
//                ASSERT_GT(mean_return, -1000);
//            }
//            if(step_i >= 14000){
//                ASSERT_GT(mean_return, -400);
//            }

//#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
//            plot_policy_and_value_function<DTYPE, ENVIRONMENT, ActorCriticType::ACTOR_NETWORK_TYPE, ActorCriticType::CRITIC_NETWORK_TYPE>(actor_critic.actor, actor_critic.critic_1, std::string("full_training"), step_i);
//#endif
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
            //            for(int evaluation_i = 0; evaluation_i < 10; evaluation_i++){
//                ENVIRONMENT::State initial_state;
//                lic::sample_initial_state(env, initial_state, rng);
//                lic::evaluate_visual<ENVIRONMENT, UI, ActorCriticType::ACTOR_NETWORK_TYPE, ENVIRONMENT_STEP_LIMIT, 5>(env, ui, actor_critic.actor, initial_state);
//            }
#endif
        }
    }
    return -1;

}
