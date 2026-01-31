#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/utils/evaluation/operations_generic.h>
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
#include <iostream>
#include <sstream>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = typename DEVICE::index_t;
using ENVIRONMENT_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
    static constexpr TI N_ENVIRONMENTS = 16;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = ENVIRONMENT::EPISODE_STEP_LIMIT;
    static constexpr TI BATCH_SIZE = N_ENVIRONMENTS*ON_POLICY_RUNNER_STEPS_PER_ENV;
    static constexpr TI TOTAL_STEP_LIMIT = 1000 * 1000 * 10;
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
struct StateComparison {
    using STATE = typename CONFIG::template State<CONFIG>;
    T actor_diff = 0;
    T critic_diff = 0;
    T ppo_diff = 0;
    T actor_optimizer_diff = 0;
    T critic_optimizer_diff = 0;
    T on_policy_runner_diff = 0;
    T dataset_diff = 0;
    T obs_normalizer_diff = 0;
    T obs_priv_normalizer_diff = 0;
    TI step_diff = 0;
    TI next_checkpoint_id_diff = 0;
    TI next_evaluation_id_diff = 0;
    bool rng_match = true;
    void compare(DEVICE& device, STATE& s1, STATE& s2) {
        actor_diff = rlt::abs_diff(device, s1.ppo.actor, s2.ppo.actor);
        critic_diff = rlt::abs_diff(device, s1.ppo.critic, s2.ppo.critic);
        ppo_diff = rlt::abs_diff(device, s1.ppo, s2.ppo);
        actor_optimizer_diff = rlt::abs_diff(device, s1.actor_optimizer, s2.actor_optimizer);
        critic_optimizer_diff = rlt::abs_diff(device, s1.critic_optimizer, s2.critic_optimizer);
        on_policy_runner_diff = rlt::abs_diff(device, s1.on_policy_runner, s2.on_policy_runner);
        dataset_diff = rlt::abs_diff(device, s1.on_policy_runner_dataset, s2.on_policy_runner_dataset);
        obs_normalizer_diff = rlt::abs_diff(device, s1.observation_normalizer, s2.observation_normalizer);
        obs_priv_normalizer_diff = rlt::abs_diff(device, s1.observation_privileged_normalizer, s2.observation_privileged_normalizer);
        step_diff = (s1.step > s2.step) ? (s1.step - s2.step) : (s2.step - s1.step);
        next_checkpoint_id_diff = (s1.next_checkpoint_id > s2.next_checkpoint_id) ? (s1.next_checkpoint_id - s2.next_checkpoint_id) : (s2.next_checkpoint_id - s1.next_checkpoint_id);
        next_evaluation_id_diff = (s1.next_evaluation_id > s2.next_evaluation_id) ? (s1.next_evaluation_id - s2.next_evaluation_id) : (s2.next_evaluation_id - s1.next_evaluation_id);
        std::stringstream ss1, ss2;
        ss1 << s1.rng.engine;
        ss2 << s2.rng.engine;
        rng_match = (ss1.str() == ss2.str());
    }
    T total_diff() const {
        return ppo_diff + actor_optimizer_diff + critic_optimizer_diff + on_policy_runner_diff + dataset_diff + obs_normalizer_diff + obs_priv_normalizer_diff + (T)step_diff + (T)next_checkpoint_id_diff + (T)next_evaluation_id_diff + (rng_match ? 0 : 1);
    }
    void print(const std::string& label) const {
        std::cout << "=== " << label << " ===" << std::endl;
        std::cout << "  actor_diff: " << actor_diff << std::endl;
        std::cout << "  critic_diff: " << critic_diff << std::endl;
        std::cout << "  ppo_diff: " << ppo_diff << std::endl;
        std::cout << "  actor_optimizer_diff: " << actor_optimizer_diff << std::endl;
        std::cout << "  critic_optimizer_diff: " << critic_optimizer_diff << std::endl;
        std::cout << "  on_policy_runner_diff: " << on_policy_runner_diff << std::endl;
        std::cout << "  dataset_diff: " << dataset_diff << std::endl;
        std::cout << "  obs_normalizer_diff: " << obs_normalizer_diff << std::endl;
        std::cout << "  obs_priv_normalizer_diff: " << obs_priv_normalizer_diff << std::endl;
        std::cout << "  step_diff: " << step_diff << std::endl;
        std::cout << "  next_checkpoint_id_diff: " << next_checkpoint_id_diff << std::endl;
        std::cout << "  next_evaluation_id_diff: " << next_evaluation_id_diff << std::endl;
        std::cout << "  rng_match: " << (rng_match ? "YES" : "NO") << std::endl;
        std::cout << "  TOTAL: " << total_diff() << std::endl;
    }
};

constexpr TI EVAL_N_EPISODES = 10;
constexpr TI EVAL_STEP_LIMIT = 200;
using EVAL_SPEC = rlt::rl::utils::evaluation::Specification<TYPE_POLICY, TI, ENVIRONMENT, EVAL_N_EPISODES, EVAL_STEP_LIMIT>;
using EVAL_BUFFER_SPEC = rlt::rl::utils::evaluation::BufferSpecification<EVAL_SPEC>;
using EVAL_BUFFER = rlt::rl::utils::evaluation::Buffer<EVAL_BUFFER_SPEC>;
using EVAL_RESULT = rlt::rl::utils::evaluation::Result<EVAL_SPEC>;

template <typename DEVICE, typename CONFIG, typename RNG>
T evaluate_policy(DEVICE& device, typename CONFIG::template State<CONFIG>& ts, RNG& rng){
    EVAL_BUFFER eval_buffer;
    EVAL_RESULT eval_result;
    ENVIRONMENT env;
    rlt::rl::environments::DummyUI ui;
    auto& actor = rlt::get_actor(ts);
    using ACTOR_TYPE = rlt::utils::typing::remove_reference_t<decltype(actor)>;
    typename ACTOR_TYPE::template State<true> actor_state;
    typename ACTOR_TYPE::template Buffer<true> actor_buffer;
    rlt::malloc(device, eval_buffer);
    rlt::malloc(device, actor_state);
    rlt::malloc(device, actor_buffer);
    rlt::reset(device, actor, actor_state, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
    rlt::evaluate(device, env, ui, actor, actor_state, actor_buffer, eval_buffer, eval_result, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
    rlt::free(device, eval_buffer);
    rlt::free(device, actor_state);
    rlt::free(device, actor_buffer);
    return eval_result.returns_mean;
}

TEST(RL_TOOLS_RL_ALGORITHMS_PPO_LOOP, PERSIST_CHECKPOINT_RIGOROUS) {
    DEVICE device;
    rlt::init(device);
    constexpr TI SEED = 42;
    constexpr TI TOTAL_STEPS = LOOP_CORE_CONFIG::CORE_PARAMETERS::STEP_LIMIT;
    constexpr TI CHECKPOINT_STEP = TOTAL_STEPS / 2;
    
    LOOP_STATE ts, ts_loaded;
    rlt::malloc(device, ts);
    rlt::malloc(device, ts_loaded);
    rlt::init(device, ts, SEED);
    rlt::init(device, ts_loaded, SEED);
    
    // Train to checkpoint
    for(TI step_i = 0; step_i < CHECKPOINT_STEP; step_i++){
        rlt::step(device, ts);
    }
    
    // Save checkpoint
    std::string checkpoint_path = "test_ppo_loop_persist_checkpoint.h5";
    {
        std::lock_guard<std::mutex> lock(rlt::persist::backends::hdf5::global_mutex());
        auto file = HighFive::File(checkpoint_path, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
        auto group = rlt::create_group(device, file, "loop_state");
        rlt::save(device, ts, group);
    }
    
    // Load checkpoint
    {
        std::lock_guard<std::mutex> lock(rlt::persist::backends::hdf5::global_mutex());
        auto file = HighFive::File(checkpoint_path, HighFive::File::ReadOnly);
        auto group = rlt::get_group(device, file, "loop_state");
        bool success = rlt::load(device, ts_loaded, group);
        ASSERT_TRUE(success);
    }
    
    // Rigorous comparison after load
    StateComparison<DEVICE, LOOP_CORE_CONFIG> comp_after_load;
    comp_after_load.compare(device, ts, ts_loaded);
    std::cout << "Step: " << CHECKPOINT_STEP << std::endl;
    comp_after_load.print("After Load (ts vs ts_loaded)");
    ASSERT_EQ(comp_after_load.total_diff(), 0) << "State mismatch after loading";
    
    // Check for NaN
    ASSERT_FALSE(rlt::is_nan(device, rlt::get_actor(ts)));
    ASSERT_FALSE(rlt::is_nan(device, rlt::get_actor(ts_loaded)));
    
    // Continue training both and compare after each step for first few steps
    std::cout << "\n=== Step-by-step comparison ===" << std::endl;
    for(TI step_i = CHECKPOINT_STEP; step_i < CHECKPOINT_STEP + 5; step_i++){
        std::cout << "step_i: " << step_i << std::endl;


        auto& env_orig = rlt::get(ts.on_policy_runner.environments, 0, 0);
        auto& params_orig = rlt::get(ts.on_policy_runner.env_parameters, 0, 0);
        auto& state_orig = rlt::get(ts.on_policy_runner.states, 0, 0);
        auto& env_loaded = rlt::get(ts_loaded.on_policy_runner.environments, 0, 0);
        auto& params_loaded = rlt::get(ts_loaded.on_policy_runner.env_parameters, 0, 0);
        auto& state_loaded = rlt::get(ts_loaded.on_policy_runner.states, 0, 0);
        ASSERT_FLOAT_EQ(state_orig.theta, state_loaded.theta);
        ASSERT_FLOAT_EQ(state_orig.theta_dot, state_loaded.theta_dot);
        ASSERT_EQ(params_orig.g, params_loaded.g);
        ASSERT_EQ(params_orig.max_speed, params_loaded.max_speed);
        ASSERT_EQ(params_orig.max_torque, params_loaded.max_torque);
        ASSERT_EQ(params_orig.dt, params_loaded.dt);
        ASSERT_EQ(params_orig.m, params_loaded.m);
        ASSERT_EQ(params_orig.l, params_loaded.l);
        ASSERT_EQ(params_orig.initial_state_min_angle, params_loaded.initial_state_min_angle);
        ASSERT_EQ(params_orig.initial_state_max_angle, params_loaded.initial_state_max_angle);
        ASSERT_EQ(params_orig.initial_state_min_speed, params_loaded.initial_state_min_speed);
        ASSERT_EQ(params_orig.initial_state_max_speed, params_loaded.initial_state_max_speed);

        ASSERT_EQ(ts.rng.engine, ts_loaded.rng.engine);

        rlt::step(device, ts);
        rlt::step(device, ts_loaded);

        StateComparison<DEVICE, LOOP_CORE_CONFIG> comp_step;
        comp_step.compare(device, ts, ts_loaded);
        std::cout << "After step " << step_i << ": total_diff = " << comp_step.total_diff();
        if(comp_step.total_diff() > 0) {
            std::cout << " [DIVERGED]" << std::endl;
            comp_step.print("Step " + std::to_string(step_i));
        } else {
            std::cout << " [OK]" << std::endl;
        }
        
        // Check for NaN
        bool ts_nan = rlt::is_nan(device, rlt::get_actor(ts));
        bool ts_loaded_nan = rlt::is_nan(device, rlt::get_actor(ts_loaded));
        if(ts_nan || ts_loaded_nan) {
            std::cout << "NaN detected! ts: " << ts_nan << ", ts_loaded: " << ts_loaded_nan << std::endl;
        }
        ASSERT_FALSE(ts_nan);
        ASSERT_FALSE(ts_loaded_nan);
        ASSERT_EQ(comp_step.total_diff(), 0) << "State diverged at step " << step_i;
    }
    
    // Continue training to completion
    for(TI step_i = CHECKPOINT_STEP + 5; step_i < TOTAL_STEPS; step_i++){
        rlt::step(device, ts);
        rlt::step(device, ts_loaded);
    }
    
    // Final comparison
    StateComparison<DEVICE, LOOP_CORE_CONFIG> comp_final;
    comp_final.compare(device, ts, ts_loaded);
    comp_final.print("Final (after full training)");
    EXPECT_EQ(comp_final.total_diff(), 0) << "State mismatch after full training";
    
    // Evaluate both
    RNG eval_rng;
    rlt::init(device, eval_rng, SEED);
    T return_original = evaluate_policy<DEVICE, LOOP_CORE_CONFIG>(device, ts, eval_rng);
    rlt::init(device, eval_rng, SEED);
    T return_loaded = evaluate_policy<DEVICE, LOOP_CORE_CONFIG>(device, ts_loaded, eval_rng);
    std::cout << "\nEvaluation return (original):  " << return_original << std::endl;
    std::cout << "Evaluation return (loaded):    " << return_loaded << std::endl;
    ASSERT_FLOAT_EQ(return_original, return_loaded) << "Evaluation returns should match";

    rlt::free(device, ts);
    rlt::free(device, ts_loaded);
}

TEST(RL_TOOLS_RL_ALGORITHMS_PPO_LOOP, PERSIST_SAVE_LOAD) {
    DEVICE device;
    rlt::init(device);
    LOOP_STATE ts, ts_loaded;
    rlt::malloc(device, ts);
    rlt::malloc(device, ts_loaded);
    rlt::init(device, ts, 123);
    rlt::init(device, ts_loaded, 123);
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
        ASSERT_TRUE(success);
    }
    
    StateComparison<DEVICE, LOOP_CORE_CONFIG> comp;
    comp.compare(device, ts, ts_loaded);
    comp.print("After Load");
    ASSERT_EQ(comp.total_diff(), 0);
    ASSERT_EQ(ts.step, ts_loaded.step);
    ASSERT_EQ(ts.next_checkpoint_id, ts_loaded.next_checkpoint_id);
    ASSERT_EQ(ts.next_evaluation_id, ts_loaded.next_evaluation_id);
    ASSERT_EQ(ts.observation_normalizer.age, ts_loaded.observation_normalizer.age);
    ASSERT_EQ(ts.observation_privileged_normalizer.age, ts_loaded.observation_privileged_normalizer.age);
    
    rlt::free(device, ts);
    rlt::free(device, ts_loaded);
    std::filesystem::remove(checkpoint_path);
}
