#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/numeric_types/policy.h>
#include <gtest/gtest.h>
#include <metra/metra.h>

namespace rl_tools::test_runner_dispatch{
    using TI = unsigned int;
    struct Device{
        using index_t = TI;
        devices::math::CPU math;
    };
    struct Environment{
        using TI = test_runner_dispatch::TI;
        using State = float;
        using Parameters = float;
        struct Observation{
            static constexpr TI DIM = 1;
            using SHAPE = tensor::Shape<TI, DIM>;
        };
        using ObservationPrivileged = Observation;
        static constexpr TI INSTANCES = 3, N_AGENTS = 1, ACTION_DIM = 1, EPISODE_STEP_LIMIT = 2;
    };
}
namespace rl_tools::utils{
    void assert_exit(test_runner_dispatch::Device&, bool condition, const char* message){
        EXPECT_TRUE(condition) << message;
    }
}
namespace rl_tools{
    template <typename PARAMETERS, typename MASK>
    void sample_initial_parameters(test_runner_dispatch::Device& device, test_runner_dispatch::Environment&, PARAMETERS& parameters, const MASK& mask, unsigned int&){
        for(unsigned int i = 0; i < 3; i++) if(get(device, mask, i)) set(device, parameters, 10.0f, i);
    }
    template <typename PARAMETERS, typename STATES, typename MASK>
    void sample_initial_state(test_runner_dispatch::Device& device, test_runner_dispatch::Environment&, PARAMETERS& parameters, STATES& states, const MASK& mask, unsigned int&){
        for(unsigned int i = 0; i < 3; i++) if(get(device, mask, i)) set(device, states, get(device, parameters, i) + i, i);
    }
}

#include <rl_tools/rl/components/on_policy_runner/operations_generic.h>

TEST(RL_TOOLS_ON_POLICY_RUNNER_DISPATCH, DEVICE_WITHOUT_ID){
    using namespace rl_tools;
    using TI = test_runner_dispatch::TI;
    using Environment = test_runner_dispatch::Environment;
    using PolicyState = Tensor<tensor::Specification<float, TI, tensor::Shape<TI, 1>>>;
    using RS = rl::components::on_policy_runner::Specification<numeric_types::Policy<float>, Environment, PolicyState>;
    using DS = rl::components::on_policy_runner::DatasetSpecification<RS, 2>;
    devices::DefaultCPU cpu;
    test_runner_dispatch::Device device;
    Environment environment;
    unsigned int rng = 0;
    rl::components::OnPolicyRunner<RS> runner;
    rl::components::on_policy_runner::Buffer<RS> buffer;
    rl::components::on_policy_runner::Dataset<DS> dataset;
    Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, 3>>> mask;
    malloc(cpu, runner); malloc(cpu, buffer); malloc(cpu, dataset); malloc(cpu, mask);
    init(cpu, runner);
    set_all(cpu, runner.reset, false);
    set_all(cpu, runner.states, -1.0f);
    set_all(cpu, mask, false);
    set(cpu, mask, true, 1);
    reset(device, runner, environment, mask, rng);
    EXPECT_EQ(get(cpu, runner.states, 0), -1.0f);
    EXPECT_EQ(get(cpu, runner.states, 1), 11.0f);
    EXPECT_FALSE(get(cpu, runner.reset, 0));
    EXPECT_TRUE(get(cpu, runner.reset, 1));
    set_all(cpu, buffer.rewards, 3.0f);
    set_all(cpu, buffer.terminated, false);
    set(cpu, buffer.terminated, true, 2);
    record_transition(device, dataset, runner, buffer, 0);
    record_transition(device, dataset, runner, buffer, 1);
    for(TI i = 0; i < 3; i++){
        EXPECT_EQ(get(dataset.rewards, i, 0), 3.0f);
        EXPECT_EQ(get(dataset.truncated, i, 0), i == 2);
        EXPECT_EQ(get(dataset.truncated, 3 + i, 0), 1.0f);
        EXPECT_EQ(get(cpu, runner.episode_step, i), 0u);
    }
    reset_mask(device, runner, mask);
    free(cpu, mask); free(cpu, dataset); free(cpu, buffer); free(cpu, runner);
    metra::log("on_policy_runner/dispatch/failures", ::testing::Test::HasFailure() ? 1.0 : 0.0);
}
