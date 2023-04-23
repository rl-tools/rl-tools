#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/rl/environments/pendulum/pendulum.h>
#include <backprop_tools/rl/environments/pendulum/operations_generic.h>
#include <backprop_tools/nn_models/mlp_unconditional_stddev/operations_cpu.h>
#include <backprop_tools/rl/components/on_policy_runner/on_policy_runner.h>
#include <backprop_tools/rl/components/on_policy_runner/operations_generic.h>
#include <backprop_tools/rl/components/on_policy_runner/persist.h>

namespace bpt = backprop_tools;


#include <gtest/gtest.h>


TEST(BACKPROP_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER, TEST){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = bpt::rl::environments::pendulum::Specification<T, TI>;
    using ENVIRONMENT = bpt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;

    constexpr TI N_ENVIRONMENTS = 3;
    using ON_POLICY_RUNNER_SPEC = bpt::rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS>;
    using ON_POLICY_RUNNER = bpt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;


    DEVICE device;
    ON_POLICY_RUNNER runner;
    bpt::malloc(device, runner);
    ENVIRONMENT envs[ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS];
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 199);
    bpt::init(device, runner, envs, rng);

    using ACTOR_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::ActivationFunction::RELU, bpt::nn::activation_functions::TANH>;
    using ACTOR_SPEC = bpt::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TYPE = bpt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;
    using ACTOR_BUFFERS = typename ACTOR_TYPE::template Buffers<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;


    constexpr TI STEPS_PER_ENV = 1000;
    using DATASET_SPEC = bpt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, STEPS_PER_ENV>;
    using DATASET = bpt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;

    ACTOR_TYPE actor;
    ACTOR_BUFFERS actor_buffers;
    DATASET dataset;
    bpt::malloc(device, actor);
    bpt::malloc(device, actor_buffers);
    bpt::malloc(device, dataset);
    bpt::init_weights(device, actor, rng);
    bpt::set_all(device, dataset.data, 0);


    bpt::collect(device, dataset, runner, actor, actor_buffers, rng);
    bpt::print(device, dataset.data);
    bpt::collect(device, dataset, runner, actor, actor_buffers, rng);
    bpt::print(device, dataset.data);
    bpt::collect(device, dataset, runner, actor, actor_buffers, rng);
    bpt::print(device, dataset.data);
    ENVIRONMENT::State states[ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS];
    for(TI env_i = 0; env_i < ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; env_i++){
        states[env_i] = get(runner.states, 0, env_i);
    }
    bpt::collect(device, dataset, runner, actor, actor_buffers, rng);
    for(TI env_i = 0; env_i < ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; env_i++){
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            TI pos = step_i * ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS + env_i;
            {
                bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
                bpt::malloc(device, observation);
                bpt::observe(device, get(runner.environments, 0, env_i), states[env_i], observation);
                auto observation_runner = bpt::view<DEVICE, decltype(dataset.observations)::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM>(device, dataset.observations, pos, 0);
                auto abs_diff = bpt::abs_diff(device, observation, observation_runner);
                if(!get(dataset.truncated, pos, 0)){
//                    ASSERT_FLOAT_EQ(abs_diff, 0);
                }
                bpt::free(device, observation);
            }
            typename ENVIRONMENT::State next_state;
            auto action = bpt::view<DEVICE, decltype(dataset.actions)::SPEC, 1, ENVIRONMENT::ACTION_DIM>(device, dataset.actions, pos, 0);
            step(device, get(runner.environments, 0, env_i), states[env_i], action, next_state);
            states[env_i] = next_state;
        }
    }
    std::string FILE_PATH = "test_rl_components_on_policy_runner_dataset.h5";
    {
        auto file = HighFive::File(FILE_PATH, HighFive::File::Overwrite);
        bpt::save(device, dataset, file.createGroup("dataset"));
    }

    {
        auto file = HighFive::File(FILE_PATH, HighFive::File::ReadOnly);
        DATASET loaded;
        bpt::malloc(device, loaded);
        bpt::load(device, loaded, file.getGroup("dataset"));
        auto abs_diff = bpt::abs_diff(device, loaded.data, dataset.data);
        ASSERT_FLOAT_EQ(0, abs_diff);
    }



}