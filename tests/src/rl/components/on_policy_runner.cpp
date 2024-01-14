#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/pendulum/pendulum.h>
#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/on_policy_runner.h>
#include <rl_tools/rl/components/on_policy_runner/operations_generic.h>
#include <rl_tools/rl/components/on_policy_runner/persist.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;


#include <gtest/gtest.h>


TEST(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER, TEST){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = rlt::rl::environments::pendulum::Specification<T, TI>;
    using ENVIRONMENT = rlt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;

    constexpr TI N_ENVIRONMENTS = 3;
    using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS>;
    using ON_POLICY_RUNNER = rlt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;


    DEVICE device;
    ON_POLICY_RUNNER runner;
    rlt::malloc(device, runner);
    ENVIRONMENT envs[ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS];
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 199);
    rlt::init(device, runner, envs, rng);

    using ACTOR_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::TANH>;
    using ACTOR_SPEC = rlt::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;
    using ACTOR_BUFFERS = typename ACTOR_TYPE::template Buffer<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;


    constexpr TI STEPS_PER_ENV = 1000;
    using DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, STEPS_PER_ENV>;
    using DATASET = rlt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;

    ACTOR_TYPE actor;
    ACTOR_BUFFERS actor_buffers;
    DATASET dataset;
    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffers);
    rlt::malloc(device, dataset);
    rlt::init_weights(device, actor, rng);
    rlt::set_all(device, dataset.data, 0);


    rlt::collect(device, dataset, runner, actor, actor_buffers, rng);
    rlt::print(device, dataset.data);
    rlt::collect(device, dataset, runner, actor, actor_buffers, rng);
    rlt::print(device, dataset.data);
    rlt::collect(device, dataset, runner, actor, actor_buffers, rng);
    rlt::print(device, dataset.data);
    ENVIRONMENT::State states[ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS];
    for(TI env_i = 0; env_i < ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; env_i++){
        states[env_i] = get(runner.states, 0, env_i);
    }
    rlt::collect(device, dataset, runner, actor, actor_buffers, rng);
    for(TI env_i = 0; env_i < ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; env_i++){
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            TI pos = step_i * ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS + env_i;
            {
                rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
                rlt::malloc(device, observation);
                rlt::observe(device, get(runner.environments, 0, env_i), states[env_i], observation, rng);
                auto observation_runner = rlt::view<DEVICE, decltype(dataset.observations)::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM>(device, dataset.observations, pos, 0);
                auto abs_diff = rlt::abs_diff(device, observation, observation_runner);
                if(!get(dataset.truncated, pos, 0)){
//                    ASSERT_FLOAT_EQ(abs_diff, 0);
                }
                rlt::free(device, observation);
            }
            typename ENVIRONMENT::State next_state;
            auto action = rlt::view<DEVICE, decltype(dataset.actions)::SPEC, 1, ENVIRONMENT::ACTION_DIM>(device, dataset.actions, pos, 0);
            step(device, get(runner.environments, 0, env_i), states[env_i], action, next_state, rng);
            states[env_i] = next_state;
        }
    }
    std::string FILE_PATH = "test_rl_components_on_policy_runner_dataset.h5";
    {
        auto file = HighFive::File(FILE_PATH, HighFive::File::Overwrite);
        rlt::save(device, dataset, file.createGroup("dataset"));
    }

    {
        auto file = HighFive::File(FILE_PATH, HighFive::File::ReadOnly);
        DATASET loaded;
        rlt::malloc(device, loaded);
        rlt::load(device, loaded, file.getGroup("dataset"));
        auto abs_diff = rlt::abs_diff(device, loaded.data, dataset.data);
        ASSERT_FLOAT_EQ(0, abs_diff);
    }



}