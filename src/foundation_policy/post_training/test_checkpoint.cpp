#include <rl_tools/operations/cpu.h>

#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
// #include <rl_tools/rl/loop/steps/nn_analytics/operations_cpu.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/utils/evaluation/operations_generic.h>

#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/layers/sample_and_squash/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/td3_sampling/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/multi_agent_wrapper/persist.h>
#include <rl_tools/rl/components/replay_buffer/persist.h>

#include "../../../logs/2025-03-26_11-06-24/checkpoints/86/checkpoint.h"


namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#include "environment.h"
#include "../pre_training/options.h"
#include "config.h"

int test(){
    DEVICE device;
    RNG rng;
    rlt::init(device, rng, 100);

    using ACTOR_TYPE = rl_tools::checkpoint::actor::TYPE::CHANGE_BATCH_SIZE<TI, 1>;
    ACTOR_TYPE::Buffer<false> buffer;
    // test
    rl_tools::checkpoint::actor::TYPE::Buffer<> test_buffer;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rl_tools::checkpoint::example::output::SHAPE>> output;
    rlt::malloc(device, test_buffer);
    rlt::malloc(device, output);
    rlt::Mode<rlt::mode::Evaluation<>> mode;
    rlt::evaluate(device, rl_tools::checkpoint::actor::module, rl_tools::checkpoint::example::input::container, output, test_buffer, rng, mode);
    T abs_diff = rlt::abs_diff(device, rl_tools::checkpoint::example::output::container, output) / rl_tools::checkpoint::example::output::SPEC::SIZE;
    auto last_step_output = rlt::view(device, output, rlt::get<0>(rl_tools::checkpoint::example::output::SHAPE{}) - 1);
    auto last_step_expected = rlt::view(device, rl_tools::checkpoint::example::output::container, rlt::get<0>(rl_tools::checkpoint::example::output::SHAPE{}) - 1);
    std::cout << "last_step_output: " << std::endl;
    rlt::print(device, last_step_output);
    std::cout << "last_step_expected: " << std::endl;
    rlt::print(device, last_step_expected);

    std::cout << "abs_diff to checkpoint example: " << abs_diff << std::endl;
    rlt::free(device, test_buffer);
    rlt::free(device, output);
    return abs_diff < 1e-5 ? 0 : 1;
}


int main(){
    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::malloc(device, rng);
    TI seed = 6;
    rlt::init(device, rng, seed);
    using EVALUATION_ACTOR_TYPE_BATCH_SIZE = typename ACTOR::template CHANGE_BATCH_SIZE<TI, NUM_EPISODES_EVAL>;
    using EVALUATION_ACTOR_TYPE = typename EVALUATION_ACTOR_TYPE_BATCH_SIZE::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<DYNAMIC_ALLOCATION>>;
    rlt::rl::environments::DummyUI ui;
    EVALUATION_ACTOR_TYPE evaluation_actor;
    EVALUATION_ACTOR_TYPE::Buffer<DYNAMIC_ALLOCATION> eval_buffer;
    rlt::malloc(device, evaluation_actor);
    rlt::malloc(device, eval_buffer);
    auto file = HighFive::File("logs/2025-03-26_11-06-24/checkpoints/86/checkpoint.h5", HighFive::File::ReadOnly);
    auto actor_group = file.getGroup("actor");
    rlt::load(device, evaluation_actor, actor_group);
    // rlt::copy(device, device, rl_tools::checkpoint::actor::module, evaluation_actor);


    ENVIRONMENT env_eval;
    ENVIRONMENT::Parameters env_eval_parameters;
    rlt::init(device, env_eval);
    rlt::sample_initial_parameters(device, env_eval, env_eval_parameters, rng);
    rlt::Mode<rlt::mode::Evaluation<>> mode;
    RESULT_EVAL result_eval;
    DATA_EVAL data_eval;
    rlt::evaluate(device, env_eval, ui, evaluation_actor, result_eval, data_eval, rng, mode);
    rlt::add_scalar(device, device.logger, "evaluation/return/mean", result_eval.returns_mean);
    rlt::add_scalar(device, device.logger, "evaluation/return/std", result_eval.returns_std);
    rlt::add_scalar(device, device.logger, "evaluation/episode_length/mean", result_eval.episode_length_mean);
    rlt::add_scalar(device, device.logger, "evaluation/episode_length/std", result_eval.episode_length_std);
    rlt::add_scalar(device, device.logger, "evaluation/share_terminated", result_eval.share_terminated);
    rlt::log(device, device.logger, "Mean return: ", result_eval.returns_mean, " Mean episode length: ", result_eval.episode_length_mean, " Share terminated: ", result_eval.share_terminated * 100, "%");

    rlt::free(device, evaluation_actor);
    rlt::free(device, eval_buffer);
}
