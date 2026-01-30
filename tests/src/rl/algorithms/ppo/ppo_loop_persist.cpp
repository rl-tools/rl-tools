#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/persist/backends/hdf5/hdf5.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/persist.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/persist.h>
#include <gtest/gtest.h>
#include <filesystem>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;
using ENVIRONMENT_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
    static constexpr TI N_ENVIRONMENTS = 4;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 32;
    static constexpr TI BATCH_SIZE = 64;
    static constexpr TI TOTAL_STEP_LIMIT = 1000 * 1000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI ACTOR_HIDDEN_DIM = 16;
    static constexpr TI CRITIC_HIDDEN_DIM = 16;
    static constexpr bool NORMALIZE_OBSERVATIONS = true;
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.1;
        static constexpr TI N_EPOCHS = 1;
    };
};
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, true>;
using LOOP_STATE = LOOP_CORE_CONFIG::template State<LOOP_CORE_CONFIG>;
template <typename DEVICE, typename CONFIG>
T abs_diff_state(DEVICE& device, typename CONFIG::template State<CONFIG>& s1, typename CONFIG::template State<CONFIG>& s2){
    T diff = 0;
    diff += rlt::abs_diff(device, s1.ppo.actor, s2.ppo.actor);
    diff += rlt::abs_diff(device, s1.ppo.critic, s2.ppo.critic);
    diff += rlt::abs_diff(device, s1.on_policy_runner_dataset.data, s2.on_policy_runner_dataset.data);
    diff += rlt::abs_diff(device, s1.observation_normalizer.mean, s2.observation_normalizer.mean);
    diff += rlt::abs_diff(device, s1.observation_normalizer.std, s2.observation_normalizer.std);
    diff += rlt::abs_diff(device, s1.observation_privileged_normalizer.mean, s2.observation_privileged_normalizer.mean);
    diff += rlt::abs_diff(device, s1.observation_privileged_normalizer.std, s2.observation_privileged_normalizer.std);
    return diff;
}
TEST(RL_TOOLS_RL_ALGORITHMS_PPO_LOOP, PERSIST_DETERMINISM) {
    DEVICE device;
    rlt::init(device);
    constexpr TI SEED = 42;
    constexpr TI TOTAL_STEPS = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT;
    constexpr TI CHECKPOINT_STEP = TOTAL_STEPS/2;
    static_assert(CHECKPOINT_STEP > 0);
    LOOP_STATE ts_continuous, ts_paused, ts_reloaded;
    rlt::malloc(device, ts_continuous);
    rlt::malloc(device, ts_paused);
    rlt::malloc(device, ts_reloaded);
    rlt::init(device, ts_continuous, SEED);
    rlt::init(device, ts_paused, SEED);
    for(TI step_i = 0; step_i < CHECKPOINT_STEP; step_i++){
        rlt::step(device, ts_continuous);
        rlt::step(device, ts_paused);
    }
    T diff_before_checkpoint = abs_diff_state<DEVICE, LOOP_CORE_CONFIG>(device, ts_continuous, ts_paused);
    EXPECT_EQ(diff_before_checkpoint, 0);
    std::string checkpoint_path = "test_ppo_loop_persist_checkpoint.h5";
    {
        std::lock_guard<std::mutex> lock(rlt::persist::backends::hdf5::global_mutex());
        auto file = HighFive::File(checkpoint_path, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
        auto group = rlt::create_group(device, file, "loop_state");
        rlt::save(device, ts_paused, group);
    }
    {
        std::lock_guard<std::mutex> lock(rlt::persist::backends::hdf5::global_mutex());
        auto file = HighFive::File(checkpoint_path, HighFive::File::ReadOnly);
        auto group = rlt::get_group(device, file, "loop_state");
        bool success = rlt::load(device, ts_reloaded, group);
        EXPECT_TRUE(success);
    }
    T diff_after_load = abs_diff_state<DEVICE, LOOP_CORE_CONFIG>(device, ts_paused, ts_reloaded);
    EXPECT_EQ(diff_after_load, 0);
    for(TI step_i = CHECKPOINT_STEP; step_i < TOTAL_STEPS; step_i++){
        rlt::step(device, ts_continuous);
        rlt::step(device, ts_reloaded);
    }
    T diff_after_training = abs_diff_state<DEVICE, LOOP_CORE_CONFIG>(device, ts_continuous, ts_reloaded);
    EXPECT_EQ(diff_after_training, 0);
    rlt::free(device, ts_continuous);
    rlt::free(device, ts_paused);
    rlt::free(device, ts_reloaded);
    std::filesystem::remove(checkpoint_path);
}
TEST(RL_TOOLS_RL_ALGORITHMS_PPO_LOOP, PERSIST_SAVE_LOAD) {
    DEVICE device;
    rlt::init(device);
    LOOP_STATE ts, ts_loaded;
    rlt::malloc(device, ts);
    rlt::malloc(device, ts_loaded);
    rlt::init(device, ts, 123);
    for(TI step_i = 0; step_i < 5; step_i++){
        rlt::step(device, ts);
    }
    std::string checkpoint_path = "test_ppo_loop_persist_save_load.h5";
    {
        std::lock_guard<std::mutex> lock(rlt::persist::backends::hdf5::global_mutex());
        auto file = HighFive::File(checkpoint_path, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
        auto group = rlt::create_group(device, file, "loop_state");
        rlt::save(device, ts, group);
    }
    {
        std::lock_guard<std::mutex> lock(rlt::persist::backends::hdf5::global_mutex());
        auto file = HighFive::File(checkpoint_path, HighFive::File::ReadOnly);
        auto group = rlt::get_group(device, file, "loop_state");
        bool success = rlt::load(device, ts_loaded, group);
        EXPECT_TRUE(success);
    }
    T diff = abs_diff_state<DEVICE, LOOP_CORE_CONFIG>(device, ts, ts_loaded);
    EXPECT_EQ(diff, 0);
    EXPECT_EQ(ts.step, ts_loaded.step);
    EXPECT_EQ(ts.next_checkpoint_id, ts_loaded.next_checkpoint_id);
    EXPECT_EQ(ts.next_evaluation_id, ts_loaded.next_evaluation_id);
    EXPECT_EQ(ts.observation_normalizer.age, ts_loaded.observation_normalizer.age);
    EXPECT_EQ(ts.observation_privileged_normalizer.age, ts_loaded.observation_privileged_normalizer.age);
    rlt::free(device, ts);
    rlt::free(device, ts_loaded);
    std::filesystem::remove(checkpoint_path);
}
