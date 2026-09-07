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

namespace rl_tools::test_runner_dispatch{
    struct Actor{
        struct { Tensor<tensor::Specification<float, TI, tensor::Shape<TI, 1>>> parameters; } log_std;
        double seen[3]{};
    };
    struct ActorBuffer{};
}
namespace rl_tools{
    test_runner_dispatch::Actor& get_last_layer(test_runner_dispatch::Actor& actor){ return actor; }
    template <typename DEVICE, typename INPUT_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE>
    void evaluate(DEVICE& device, test_runner_dispatch::Actor& actor, const Tensor<INPUT_SPEC>& input, Tensor<OUTPUT_SPEC>& output, test_runner_dispatch::ActorBuffer&, RNG&, const Mode<MODE>&){
        static_assert(utils::typing::is_same_v<typename INPUT_SPEC::T, double>);
        static_assert(length(typename INPUT_SPEC::SHAPE{}) == 3 && get<0>(typename INPUT_SPEC::SHAPE{}) == 1 && get<1>(typename INPUT_SPEC::SHAPE{}) == 3);
        static_assert(mode::is<MODE, mode::Rollout>);
        for(unsigned int i = 0; i < 3; i++){
            actor.seen[i] = get(device, input, 0, i, 0);
            set(device, output, static_cast<float>(actor.seen[i]), 0, i, 0);
        }
    }
}
#include <rl_tools/rl/components/on_policy_runner/operations_generic_extensions.h>

TEST(RL_TOOLS_ON_POLICY_RUNNER_DISPATCH, HYBRID_PRESERVES_OBSERVATION_PRECISION){
    using namespace rl_tools;
    using TI = test_runner_dispatch::TI;
    using Environment = test_runner_dispatch::Environment;
    using PolicyState = Tensor<tensor::Specification<float, TI, tensor::Shape<TI, 1>>>;
    using RS = rl::components::on_policy_runner::Specification<numeric_types::Policy<float>, Environment, PolicyState, Environment::Observation, Environment::ObservationPrivileged, double>;
    using DS = rl::components::on_policy_runner::DatasetSpecification<RS, 2>;
    devices::DefaultCPU cpu, evaluation_device;
    devices::DefaultCPU::SPEC::RANDOM::ENGINE<> rng;
    malloc(cpu, rng); init(cpu, rng, 1);
    test_runner_dispatch::Actor actor, evaluation_actor;
    test_runner_dispatch::ActorBuffer actor_buffer;
    malloc(cpu, actor.log_std.parameters); set_all(cpu, actor.log_std.parameters, 0);
    rl::components::on_policy_runner::Dataset<DS> dataset;
    rl::components::on_policy_runner::Buffer<RS> buffer;
    rl::components::on_policy_runner::CollectionEvaluationBuffer<RS> transfer, evaluation_transfer;
    malloc(cpu, dataset); malloc(cpu, buffer); malloc(cpu, transfer); malloc(evaluation_device, evaluation_transfer);
    for(TI i = 0; i < 3; i++) set(cpu, dataset.all_observations, 1.0 + 1e-12 * (i + 1), i, 0);
    interlude(cpu, evaluation_device, dataset, buffer, actor, evaluation_actor, actor_buffer, transfer, evaluation_transfer, rng, rng, 0);
    for(TI i = 0; i < 3; i++) EXPECT_EQ(evaluation_actor.seen[i], 1.0 + 1e-12 * (i + 1));
    free(evaluation_device, evaluation_transfer); free(cpu, transfer); free(cpu, buffer); free(cpu, dataset);
    free(cpu, actor.log_std.parameters); free(cpu, rng);
}
