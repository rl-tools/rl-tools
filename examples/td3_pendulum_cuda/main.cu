#include <layer_in_c/operations/cpu.h>


#define FUNCTION_PLACEMENT __device__ __host__

#include <layer_in_c/operations/cuda.h>

#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/rl/environments/pendulum/operations_generic.h>
#include <layer_in_c/rl/components/off_policy_runner/operations_generic.h>
#include <layer_in_c/rl/algorithms/td3/operations_generic.h>

#include <layer_in_c/rl/utils/evaluation.h>

namespace lic = layer_in_c;
using DTYPE = float;

using DEVICE = lic::devices::DefaultCPU;
typedef lic::rl::environments::pendulum::Specification<DTYPE, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::Pendulum<DEVICE, PENDULUM_SPEC> ENVIRONMENT;
ENVIRONMENT env;

struct ActorStructureSpec{
    using T = DTYPE;
    static constexpr size_t INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM;
    static constexpr size_t OUTPUT_DIM = ENVIRONMENT::ACTION_DIM;
    static constexpr int NUM_LAYERS = 3;
    static constexpr int HIDDEN_DIM = 64;
    static constexpr lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = lic::nn::activation_functions::RELU;
    static constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = lic::nn::activation_functions::TANH;
};

struct CriticStructureSpec{
    using T = DTYPE;
    static constexpr size_t INPUT_DIM = ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM;
    static constexpr size_t OUTPUT_DIM = 1;
    static constexpr int NUM_LAYERS = 3;
    static constexpr int HIDDEN_DIM = 64;
    static constexpr lic::nn::activation_functions::ActivationFunction HIDDEN_ACTIVATION_FUNCTION = lic::nn::activation_functions::RELU;
    static constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_ACTIVATION_FUNCTION = lic::nn::activation_functions::IDENTITY;
};

using AC_DEVICE = lic::devices::DefaultCPU;
using AC_DEVICE_CUDA = lic::devices::DefaultCUDA;
template <typename T>
struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<AC_DEVICE, T>{
    constexpr static size_t CRITIC_BATCH_SIZE = 100;
    constexpr static size_t ACTOR_BATCH_SIZE = 100;
};

using NN_DEVICE = lic::devices::DefaultCPU;
using NN_DEVICE_CUDA = lic::devices::DefaultCUDA;

template <typename DEVICE>
struct NetworkTypes{
    using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<DEVICE, ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
    using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<DEVICE, ACTOR_NETWORK_SPEC>;

    using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<DEVICE, ActorStructureSpec>;
    using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<DEVICE , ACTOR_TARGET_NETWORK_SPEC>;

    using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<DEVICE, CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
    using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<DEVICE, CRITIC_NETWORK_SPEC>;

    using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<DEVICE, CriticStructureSpec>;
    using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<DEVICE, CRITIC_TARGET_NETWORK_SPEC>;
};

using NetworkTypesCPU = NetworkTypes<NN_DEVICE>;
using NetworkTypesCUDA = NetworkTypes<NN_DEVICE_CUDA>;
using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, ENVIRONMENT, NN_DEVICE, NetworkTypesCPU::ACTOR_NETWORK_TYPE, NetworkTypesCPU::ACTOR_TARGET_NETWORK_TYPE, NetworkTypesCPU::CRITIC_NETWORK_TYPE, NetworkTypesCPU::CRITIC_TARGET_NETWORK_TYPE, TD3PendulumParameters<DTYPE>>;
using TD3_SPEC_CUDA = lic::rl::algorithms::td3::Specification<DTYPE, ENVIRONMENT, NN_DEVICE, NetworkTypesCUDA::ACTOR_NETWORK_TYPE, NetworkTypesCUDA::ACTOR_TARGET_NETWORK_TYPE, NetworkTypesCUDA::CRITIC_NETWORK_TYPE, NetworkTypesCUDA::CRITIC_TARGET_NETWORK_TYPE, TD3PendulumParameters<DTYPE>>;
using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<AC_DEVICE, TD3_SPEC>;
using ActorCriticTypeCUDA = lic::rl::algorithms::td3::ActorCritic<AC_DEVICE_CUDA, TD3_SPEC_CUDA>;



constexpr size_t REPLAY_BUFFER_CAP = 500000;
constexpr size_t ENVIRONMENT_STEP_LIMIT = 200;
AC_DEVICE::SPEC::LOGGING logger;
AC_DEVICE device(logger);
NN_DEVICE nn_device(logger);
lic::rl::components::OffPolicyRunner<
        AC_DEVICE,
        lic::rl::components::off_policy_runner::Specification<
                AC_DEVICE,
                DTYPE,
                ENVIRONMENT,
                REPLAY_BUFFER_CAP,
                ENVIRONMENT_STEP_LIMIT,
                lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>
        >
> off_policy_runner(device);
ActorCriticType actor_critic(device, nn_device);
const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

#include <iostream>

template <typename C, typename RB, typename RNG>
__global__ void
train_critic_kernel(const ActorCriticTypeCUDA* pac, C* pc, const RB* prb, RNG* prng, bool init){
    if(blockIdx.x == 0 && threadIdx.x == 0){
        printf("train_critic_kernel\n");
        if(init){
            curand_init(0, 0, 0, prng);
        }
        ActorCriticTypeCUDA ac = *pac;
        C c = *pc;
        RB rb = *prb;
        RNG rng = *prng;
        lic::train_critic(ac, c, rb, rng);
        *prng = rng;
        lic::copy(pc, &c);
    }
}

int main() {
    std::mt19937 rng(2);
    lic::init(actor_critic, rng);
    ActorCriticTypeCUDA                            * actor_critic_gpu;
    ActorCriticTypeCUDA::SPEC::CRITIC_NETWORK_TYPE * critic_2_gpu;
    decltype(off_policy_runner.replay_buffer)      * rb_gpu;
    curandState* rng_gpu;
    cudaMalloc(&actor_critic_gpu, sizeof(actor_critic));
    cudaMalloc(&    critic_2_gpu, sizeof(actor_critic.critic_2));
    cudaMalloc(&          rb_gpu, sizeof(off_policy_runner.replay_buffer));
    cudaMalloc(&         rng_gpu, sizeof(curandState));
    cudaDeviceSynchronize();

    for(int step_i = 0; step_i < 15000; step_i++){
        if(step_i > REPLAY_BUFFER_CAP){
            std::cout << "warning: replay buffer is rolling over" << std::endl;
        }
        lic::step(off_policy_runner, actor_critic.actor, rng);

        if(off_policy_runner.replay_buffer.full || off_policy_runner.replay_buffer.position > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                std::cout << "step_i: " << step_i << std::endl;
            }
//            DTYPE critic_1_loss = lic::train_critic(actor_critic, actor_critic.critic_1, off_policy_runner.replay_buffer, rng);
//            lic::train_critic(actor_critic, actor_critic.critic_2, off_policy_runner.replay_buffer, rng);
            cudaMemcpy(actor_critic_gpu, &actor_critic                   , sizeof(actor_critic                   ), cudaMemcpyHostToDevice);
            cudaMemcpy(    critic_2_gpu, &actor_critic.critic_2          , sizeof(actor_critic.critic_2          ), cudaMemcpyHostToDevice);
            cudaMemcpy(          rb_gpu, &off_policy_runner.replay_buffer, sizeof(off_policy_runner.replay_buffer), cudaMemcpyHostToDevice);

            dim3 grid(1);
            dim3 block(1);
            cudaDeviceSynchronize();
            train_critic_kernel<<<grid, block>>>(actor_critic_gpu, critic_2_gpu, rb_gpu, rng_gpu, off_policy_runner.replay_buffer.position == N_WARMUP_STEPS + 1);
            cudaDeviceSynchronize();

            std::cout << "Critic 1 loss: " << std::endl;
            if(step_i % 2 == 0){
                lic::train_actor(actor_critic, off_policy_runner.replay_buffer, rng);
                lic::update_targets(actor_critic);
            }
        }
        if(step_i % 1000 == 0){
            DTYPE mean_return = lic::evaluate<DEVICE, ENVIRONMENT, decltype(actor_critic.actor), typeof(rng), ENVIRONMENT_STEP_LIMIT, true>(device, env, actor_critic.actor, 1, rng);
            std::cout << "Mean return: " << mean_return << std::endl;
        }
    }
    return 0;
}
