#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>

#include "td3/parameters.h"

#include <backprop_tools/nn/layers/dense/operations_cpu.h>
#include <backprop_tools/nn_models/sequential/operations_generic.h>


#include <backprop_tools/rl/utils/validation.h>

#include <backprop_tools/rl/environments/multirotor/ui.h>

#include <backprop_tools/nn/optimizers/adam/adam.h>

//#include "../../../../checkpoints/multirotor_td3/multirotor_td3_2023_10_04_19_28_21/actor_000000003000000.h"
#include "../../../../checkpoints/multirotor_td3/multirotor_td3_2023_10_04_19_20_42/actor_000000003000000.h"

#include <thread>

namespace bpt = backprop_tools;

//template <typename DEVICE, typename INPUT_SPEC, typename OUTPUT_SPEC>
//void evaluate(DEVICE& device, Policy& p, bpt::Matrix<INPUT_SPEC>& input, bpt::Matrix<OUTPUT_SPEC>& output, PolicyBuffers& buffers){
//    using TI = typename DEVICE::index_t;
//    for(TI row_i = 0; row_i < OUTPUT_SPEC::ROWS; row_i++){
//        for(TI col_i = 0; col_i < OUTPUT_SPEC::COLS; col_i++){
//            set(output, row_i, col_i, 1337);
//        }
//    }
//}

using DEVICE = bpt::devices::DefaultCPU;

using T = float;
using TI = typename DEVICE::index_t;


using ENVIRONMENT = parameters_0::environment<T, TI>::ENVIRONMENT;


using VALIDATION_SPEC = bpt::rl::utils::validation::Specification<T, TI, ENVIRONMENT>;
constexpr TI N_EPISODES = 10;
constexpr TI MAX_EPISODE_LENGTH = 1000;
using TASK_SPEC = bpt::rl::utils::validation::TaskSpecification<VALIDATION_SPEC, N_EPISODES, MAX_EPISODE_LENGTH>;

int main(){

    auto& p = backprop_tools::checkpoint::actor::model;
    bpt::utils::typing::remove_reference<decltype(p)>::type::DoubleBuffer<N_EPISODES> buffers;
    bpt::rl::utils::validation::Task<TASK_SPEC> task;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM{}, 3);
    DEVICE device;
    ENVIRONMENT envs[N_EPISODES];
    for(ENVIRONMENT& env: envs){
        env.parameters = parameters_0::environment<T, TI>::parameters;
        env.parameters.mdp.init.max_angle *= 0.2;
    }
    bpt::init(device, task, envs, rng);
    bpt::malloc(device, buffers);

    // UI
    using UI = bpt::rl::environments::multirotor::UI<ENVIRONMENT>;
    UI ui[N_EPISODES];
    for(TI i = 0; i < N_EPISODES; i++){
        ui[i].host = "localhost";
        ui[i].port = "8080";
        ui[i].id = i;
        bpt::init(device, envs[i], ui[i]);
    }

    while(true){
        bpt::reset(device, task, rng);
        for(TI step_i=0; step_i < MAX_EPISODE_LENGTH; step_i++){
            bpt::step(device, task, p, buffers, rng);
            for(TI i = 0; i < N_EPISODES; i++){
                if(!task.terminated[i]){
                    auto action = bpt::row(device, task.episode_buffer[i].actions, task.step-1);
                    bpt::set_state(device, ui[i], task.state[i], action);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    return 0;
}