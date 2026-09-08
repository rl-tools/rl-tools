#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/persist/backends/tar/operations_cpu.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/standardize/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <gtest/gtest.h>
#include <metra/metra.h>

namespace rl_tools::test_environment_persist{
    using T = float;
    using TI = devices::DefaultCPU::index_t;
    using TYPE_POLICY = numeric_types::Policy<T>;
    struct EnvironmentParameters: rl::environments::pendulum::DefaultParameters<T>{ T reset_offset = 0; };
    using Base = rl::environments::Pendulum<rl::environments::pendulum::Specification<T, TI, EnvironmentParameters>>;
    struct Environment: Base{
        Parameters parameters;
        TI resets;
        int* resource;
    };
}
namespace rl_tools{
    template <typename DEVICE> void malloc(DEVICE&, test_environment_persist::Environment& env){ env.resource = new int(42); }
    template <typename DEVICE> void free(DEVICE&, test_environment_persist::Environment& env){ delete env.resource; }
    template <typename DEVICE> void init(DEVICE&, test_environment_persist::Environment& env){ env.parameters = {}; env.resets = 0; }
    template <typename DEVICE, typename RNG>
    void sample_initial_parameters(DEVICE&, test_environment_persist::Environment& env, test_environment_persist::Environment::Parameters& parameters, RNG&){ parameters = env.parameters; }
    template <typename DEVICE, typename RNG>
    void sample_initial_state(DEVICE& device, test_environment_persist::Environment& env, test_environment_persist::Environment::Parameters& parameters, test_environment_persist::Environment::State& state, RNG& rng){
        sample_initial_state(device, static_cast<test_environment_persist::Base&>(env), parameters, state, rng);
        state.theta += parameters.reset_offset + 0.001f * ++env.resets;
    }
    template <typename DEVICE, typename GROUP>
    void save(DEVICE& device, test_environment_persist::Environment& env, GROUP& group){
        save_binary(device, &env.parameters, 1, group, "parameters");
        save_binary(device, &env.resets, 1, group, "resets");
    }
    template <typename DEVICE, typename GROUP>
    bool load(DEVICE& device, test_environment_persist::Environment& env, GROUP& group){
        bool success = load_binary(device, &env.parameters, 1, group, "parameters");
        success &= load_binary(device, &env.resets, 1, group, "resets");
        return success;
    }
}
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/persist.h>

namespace rl_tools::test_environment_persist{
    struct Parameters: rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, Environment>{
        static constexpr TI N_ENVIRONMENTS = 2, ON_POLICY_RUNNER_STEPS_PER_ENV = 4, BATCH_SIZE = 8;
        static constexpr TI EPISODE_STEP_LIMIT = 2, ACTOR_HIDDEN_DIM = 4, CRITIC_HIDDEN_DIM = 4;
        struct PPO_PARAMETERS: rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{ static constexpr TI N_EPOCHS = 1; };
    };
    using RNG = devices::generic::random::ArrayENGINE<devices::generic::random::ArraySpecification<TI, 2>>;
    using Config = rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, Environment, Parameters, rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, true>;
    using State = Config::State<Config>;
}
TEST(PPO_ENVIRONMENT_PERSIST, MUTABLE_DEFAULTS_AND_RESUME){
    using namespace rl_tools;
    using namespace test_environment_persist;
    devices::DefaultCPU device;
    State original, hdf5, tar;
    malloc(device, original); malloc(device, hdf5); malloc(device, tar);
    init(device, original, 42); init(device, hdf5, 3); init(device, tar, 5);
    int* resources[2][2];
    for(TI i = 0; i < 2; i++){
        auto& env = get_ref(device, original.environment.environments, i);
        env.parameters.reset_offset = 5 + i;
        env.resets = 20 + i;
        resources[0][i] = get_ref(device, hdf5.environment.environments, i).resource;
        resources[1][i] = get_ref(device, tar.environment.environments, i).resource;
    }
    reset(device, original.on_policy_runner, original.environment, original.rng);
    for(TI i = 0; i < 2; i++) EXPECT_EQ(get_ref(device, original.on_policy_runner.env_parameters, i).reset_offset, 5 + i);
    step(device, original);
    const char* path = "test_ppo_mutable_environment.h5";
    {
        persist::backends::hdf5::File file(path, persist::backends::hdf5::Mode::WRITE);
        auto group = create_group(device, file, "loop_state");
        save(device, original, group);
    }
    {
        persist::backends::hdf5::File file(path, persist::backends::hdf5::Mode::READ);
        auto group = get_group(device, file, "loop_state");
        ASSERT_TRUE(load(device, hdf5, group));
    }
    persist::backends::tar::Writer writer;
    persist::backends::tar::WriterGroup<persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> write_group{"", &writer};
    save(device, original, write_group);
    persist::backends::tar::finalize(device, writer);
    persist::backends::tar::ReaderGroup<persist::backends::tar::ReaderGroupSpecification<TI>> read_group;
    read_group.data = {writer.buffer.data(), static_cast<TI>(writer.buffer.size())};
    ASSERT_TRUE(load(device, tar, read_group));
    for(TI i = 0; i < 2; i++){
        auto& source = get_ref(device, original.environment.environments, i);
        auto& restored_hdf5 = get_ref(device, hdf5.environment.environments, i);
        auto& restored_tar = get_ref(device, tar.environment.environments, i);
        EXPECT_EQ(restored_hdf5.parameters.reset_offset, source.parameters.reset_offset);
        EXPECT_EQ(restored_tar.parameters.reset_offset, source.parameters.reset_offset);
        EXPECT_EQ(restored_hdf5.resets, source.resets);
        EXPECT_EQ(restored_tar.resets, source.resets);
        EXPECT_EQ(restored_hdf5.resource, resources[0][i]);
        EXPECT_EQ(restored_tar.resource, resources[1][i]);
    }
    reset(device, original.on_policy_runner, original.environment, original.rng);
    reset(device, hdf5.on_policy_runner, hdf5.environment, hdf5.rng);
    reset(device, tar.on_policy_runner, tar.environment, tar.rng);
    for(TI i = 0; i < 3; i++){
        step(device, original); step(device, hdf5); step(device, tar);
        EXPECT_EQ(abs_diff(device, original.on_policy_runner_dataset, hdf5.on_policy_runner_dataset), 0);
        EXPECT_EQ(abs_diff(device, original.on_policy_runner_dataset, tar.on_policy_runner_dataset), 0);
        EXPECT_EQ(abs_diff(device, original.ppo, hdf5.ppo), 0);
        EXPECT_EQ(abs_diff(device, original.ppo, tar.ppo), 0);
        EXPECT_EQ(abs_diff(device, original.rng, hdf5.rng), 0);
        EXPECT_EQ(abs_diff(device, original.rng, tar.rng), 0);
    }
    free(device, tar); free(device, hdf5); free(device, original);
    std::remove(path);
    metra::log("ppo/persistence/mutable_environment_failures", ::testing::Test::HasFailure() ? 1.0 : 0.0);
}
