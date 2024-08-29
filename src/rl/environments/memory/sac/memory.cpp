#define RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD
#ifdef RL_TOOLS_ENABLE_TRACY
#include "Tracy.hpp"
#endif

#define MUX
#ifdef MUX
#include <rl_tools/operations/cpu_mux.h>
#else
#include <rl_tools/operations/cpu.h>
#endif
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#ifdef MUX
#include <rl_tools/nn/operations_cpu_mux.h>
#else
#include <rl_tools/nn/operations_cpu.h>
#endif
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/rl/environments/memory/operations_cpu.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential_v2/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/containers/matrix/persist.h>
#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/optimizers/adam/instance/persist.h>
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn_models/sequential_v2/persist.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
#endif

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/extrack/operations_cpu.h>
#include <rl_tools/rl/loop/steps/checkpoint/operations_cpu.h>
#include <rl_tools/rl/loop/steps/save_trajectories/operations_cpu.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

namespace rlt = rl_tools;

#include "approximators.h"


#ifdef MUX
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
#else
using DEVICE = rlt::devices::DefaultCPU;
#endif
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

#include "parameters.h"


using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;


int main(){
    TI seed = 1;
    DEVICE device;
    LOOP_STATE ts;
    ts.extrack_name = "sequential";
    ts.extrack_population_variates = "algorithm_environment";
    ts.extrack_population_values = "sac_memory";
    rlt::malloc(device);
    rlt::init(device);
    rlt::malloc(device, ts);
    rlt::init(device, ts, seed);
#ifdef RL_TOOLS_ENABLE_TENSORBOARD
    rlt::init(device, device.logger, ts.extrack_seed_path);
#endif
    while(!rlt::step(device, ts)){
#ifdef RL_TOOLS_ENABLE_TRACY
        FrameMark;
#endif
//        rlt::set_all(device, ts.actor_critic.critic_1.content.b_iz, -100);
//        rlt::set_all(device, ts.actor_critic.critic_target_1.content.b_iz, -100);
//        rlt::set_all(device, ts.actor_critic.critic_2.content.b_iz, -100);
//        rlt::set_all(device, ts.actor_critic.critic_target_2.content.b_iz, -100);
//        rlt::set_all(device, ts.actor_critic.actor.content.b_iz, -100);
//        if(ts.step % 1000 == 0){
//
//            constexpr TI TEST_SEQUENCE_LENGTH = SEQUENCE_LENGTH;
//            T biz_mean = rlt::sum(device, ts.actor_critic.critic_1.content.b_iz)/ decltype(ts.actor_critic.critic_1.content.b_iz)::SPEC::SIZE;
//            T bhz_mean = rlt::sum(device, ts.actor_critic.critic_1.content.b_hz)/ decltype(ts.actor_critic.critic_1.content.b_hz)::SPEC::SIZE;
//            std::cout << "b_iz mean: " << biz_mean << " b_hz: " << bhz_mean << std::endl;
//            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TEST_SEQUENCE_LENGTH, 1, 2>>> test_input;
//            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TEST_SEQUENCE_LENGTH, 1, 1>>> test_output;
//            rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, TEST_SEQUENCE_LENGTH, 1, 1>>> test_actor_output;
//            decltype(ts.actor_critic.critic_1)::Buffer<1> critic_buffer;
//            decltype(ts.actor_critic.actor)::Buffer<1> actor_buffer;
//            rlt::malloc(device, test_input);
//            rlt::malloc(device, test_output);
//            rlt::malloc(device, test_actor_output);
//            rlt::malloc(device, actor_buffer);
//            rlt::malloc(device, critic_buffer);
//            auto actor_input = rlt::view_range(device, test_input, 0, rlt::tensor::ViewSpec<2, 1>{});
//            if(TEST_SEQUENCE_LENGTH >= 2){
//                for(TI seq_i = 0; seq_i < TEST_SEQUENCE_LENGTH-1; seq_i++){
//                    T value = rlt::random::uniform_int_distribution(device.random, 0, 1, ts.rng);
//                    T action = rlt::random::uniform_int_distribution(device.random, 0, 1, ts.rng);
//                    rlt::set(device, test_input, value, seq_i, 0, 0);
//                    rlt::set(device, test_input, action, seq_i, 0, 1);
//                }
//            }
////            rlt::nn::Mode<rlt::nn::layers::gru::StepByStepMode<TI, rlt::nn::mode::Inference>> mode;
////            mode.reset = true;
//            rlt::nn::Mode<rlt::nn::mode::Inference> mode;
//            for(TI input_i = 0; input_i < 2; input_i++){
//                rlt::set(device, test_input, (T)input_i, TEST_SEQUENCE_LENGTH-1, 0, 0);
//                for(TI action_i = 0; action_i < 2; action_i++){
//                    T action = ((T)action_i)/10;
//                    rlt::set(device, test_input, action, TEST_SEQUENCE_LENGTH-1, 0, 1);
//                    rlt::evaluate(device, ts.actor_critic.critic_1, test_input, test_output, critic_buffer, ts.rng, mode);
//                    std::cout << "Input " << input_i << " action " << action << " value: " << rlt::get(device, test_output, TEST_SEQUENCE_LENGTH-1, 0, 0) << std::endl;
//                }
//                rlt::evaluate(device, ts.actor_critic.actor, actor_input, test_actor_output, actor_buffer, ts.rng, mode);
//                std::cout << "Input " << input_i << " actor_action " << rlt::get(device, test_actor_output, TEST_SEQUENCE_LENGTH-1, 0, 0) << std::endl;
//            }
//            rlt::free(device, test_input);
//            rlt::free(device, test_output);
//            rlt::free(device, actor_buffer);
//            rlt::free(device, critic_buffer);
//        }

    }
    return 0;
}
