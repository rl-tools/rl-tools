#include <backprop_tools/operations/cpu_mux.h>

#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>
#include <backprop_tools/rl/environments/multirotor/metrics.h>

#include "td3/parameters.h"

#include <backprop_tools/nn/layers/dense/operations_cpu.h>
#include <backprop_tools/nn_models/sequential/operations_generic.h>



#include <backprop_tools/rl/utils/validation_analysis.h>

#ifdef ENABLE_UI
#include <backprop_tools/rl/environments/multirotor/ui.h>
#endif

#include <backprop_tools/nn/optimizers/adam/adam.h>

//#include "../../../../checkpoints/multirotor_td3/multirotor_td3_2023_10_04_19_28_21/actor_000000003000000.h"
#include "../../../../tests/data/actor_checkpoint.h"

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
using DEV_SPEC = backprop_tools::devices::cpu::Specification<backprop_tools::devices::math::CPU, backprop_tools::devices::random::CPU, backprop_tools::devices::logging::CPU_TENSORBOARD>;
using DEVICE = backprop_tools::DEVICE_FACTORY<DEV_SPEC>;

using T = float;
using TI = typename DEVICE::index_t;


using ABLATION_SPEC = parameters::DefaultAblationSpec;
using ENVIRONMENT = parameters_0::environment<T, TI, ABLATION_SPEC>::ENVIRONMENT;


using VALIDATION_SPEC = bpt::rl::utils::validation::Specification<T, TI, ENVIRONMENT>;
constexpr TI N_EPISODES = 10;
constexpr TI MAX_EPISODE_LENGTH = parameters_0::rl<T, TI, ENVIRONMENT>::ENVIRONMENT_STEP_LIMIT;
using TASK_SPEC = bpt::rl::utils::validation::TaskSpecification<VALIDATION_SPEC, N_EPISODES, MAX_EPISODE_LENGTH>;


using ADDITIONAL_METRICS = bpt::rl::utils::validation::set::Component<bpt::rl::utils::validation::metrics::SettlingFractionPosition<TI, 200>, bpt::rl::utils::validation::set::FinalComponent>;
using METRICS = bpt::rl::utils::validation::DefaultMetrics<ADDITIONAL_METRICS>;

int main(){

    auto& p = backprop_tools::checkpoint::actor::model;
    bpt::utils::typing::remove_reference<decltype(p)>::type::DoubleBuffer<N_EPISODES> buffers;
    bpt::rl::utils::validation::Task<TASK_SPEC> task;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM{}, 3);
    DEVICE device;
    bpt::construct(device, device.logger);
    ENVIRONMENT envs[N_EPISODES];
    for(ENVIRONMENT& env: envs){
        env.parameters = parameters_0::environment<T, TI, ABLATION_SPEC>::parameters;
//        env.parameters.mdp.init.max_angle *= 0.2;
    }
    bpt::init(device, task, envs, rng);
    bpt::malloc(device, buffers);

    // UI
#ifdef ENABLE_UI
    using UI = bpt::rl::environments::multirotor::UI<ENVIRONMENT>;
    UI ui[N_EPISODES];
    for(TI i = 0; i < N_EPISODES; i++){
        ui[i].host = "localhost";
        ui[i].port = "8080";
        ui[i].id = i;
        bpt::init(device, envs[i], ui[i]);
    }
#endif

    std::cout << "Validating actor: " << bpt::checkpoint::meta::name << std::endl;

    while(true){
        bpt::reset(device, task, rng);
        bool completed = false;
        while(!completed){
            completed = bpt::step(device, task, p, buffers, rng);
#ifdef ENABLE_UI
            for(TI i = 0; i < N_EPISODES; i++){
                if(!task.terminated[i]){
                    auto action = bpt::row(device, task.episode_buffer[i].actions, task.step-1);
                    bpt::set_state(device, ui[i], task.state[i], action);
                }
            }
#endif
//            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
//        for(TI metric_i = 0; metric_i < bpt::length(device, METRICS{}); metric_i++){

        constexpr TI index = 0;
//        T value = bpt::evaluate(device, bpt::get(device, METRICS{}, bpt::Constant<0>{}), task);
//        }
        bpt::analyse_log(device, task, METRICS{});
//        T return_mean = bpt::evaluate(device, bpt::rl::utils::validation::metrics::ReturnMean{}, task);
//        T return_std = bpt::evaluate(device, bpt::rl::utils::validation::metrics::ReturnStd{}, task);
//        T terminated_fraction = bpt::evaluate(device, bpt::rl::utils::validation::metrics::TerminatedFraction{}, task);
//        T episode_length_mean = bpt::evaluate(device, bpt::rl::utils::validation::metrics::EpisodeLengthMean{}, task);
//        T episode_length_std = bpt::evaluate(device, bpt::rl::utils::validation::metrics::EpisodeLengthStd{}, task);
//        T settled_fraction_position_20cm = bpt::evaluate(device, bpt::rl::utils::validation::metrics::SettlingFractionPosition<TI, TI(200)>{}, task);
//        std::ostringstream oss;
//        oss << std::setprecision(2) << std::fixed
//            << std::left << std::setw(10) << "Return mean:"          << std::right << std::setw(10) << return_mean
//            << std::left << std::setw(10) << " | return std:"          << std::right << std::setw(10) << return_std
//            << std::left << std::setw(10) << " | terminated fraction:" << std::right << std::setw(10) << terminated_fraction
//            << std::left << std::setw(10) << " | episode length mean:" << std::right << std::setw(10) << episode_length_mean
//            << std::left << std::setw(10) << " | episode length std:"  << std::right << std::setw(10) << episode_length_std
//            << std::left << std::setw(10) << " | settled fraction:"    << std::right << std::setw(10) << settled_fraction_position_20cm
//        ;
//        std::cout << oss.str() << std::endl;
    }

    return 0;
}