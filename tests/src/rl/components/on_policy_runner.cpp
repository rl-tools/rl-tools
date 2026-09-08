#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/pendulum/pendulum.h>
#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/rl/environments/batch/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/components/on_policy_runner/on_policy_runner.h>
#include <rl_tools/rl/components/on_policy_runner/operations_generic.h>
#include <rl_tools/random/operations_generic_array.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/persist.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
#include <gtest/gtest.h>

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE::index_t;
using ENVIRONMENT_SPEC = rlt::rl::environments::pendulum::Specification<T, TI>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
constexpr TI BATCH_SIZE = 1;

template <typename CAPABILITY>
struct Actor{
    using ACTOR_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
    using ACTOR_SPEC = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::ActivationFunction::TANH, rlt::nn::activation_functions::IDENTITY>;
    using ACTOR = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<ACTOR_SPEC>;
    using MODULE_CHAIN = rlt::nn_models::sequential::Module<ACTOR>;

    using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, ACTOR_INPUT_SHAPE>;
};

TEST(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER, TEST){
    using ACTOR_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using ACTOR_TYPE = typename Actor<ACTOR_CAPABILITY>::MODEL;

    constexpr TI N_ENVIRONMENTS = 3;
    using BATCH_SPEC = rlt::rl::environments::batch::Specification<ENVIRONMENT, N_ENVIRONMENTS>;
    using BATCH = rlt::rl::environments::batch::Independent<BATCH_SPEC>;
    using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<TYPE_POLICY, BATCH, ACTOR_TYPE::State<>>;
    using ON_POLICY_RUNNER = rlt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
    using ON_POLICY_RUNNER_BUFFER = rlt::rl::components::on_policy_runner::Buffer<ON_POLICY_RUNNER_SPEC>;


    DEVICE device;
    BATCH environment;
    ON_POLICY_RUNNER runner;
    ON_POLICY_RUNNER_BUFFER runner_buffer;
    rlt::malloc(device, environment);
    rlt::malloc(device, runner);
    rlt::malloc(device, runner_buffer);
    rlt::devices::generic::random::ArrayENGINE<rlt::devices::generic::random::ArraySpecification<TI, 1024>> rng;

    using ACTOR_ROLLOUT_TYPE = typename ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;
    using ACTOR_BUFFERS = typename ACTOR_ROLLOUT_TYPE::template Buffer<>;


    constexpr TI STEPS_PER_ENV = 1000;
    using DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, STEPS_PER_ENV>;
    using DATASET = rlt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;

    ACTOR_TYPE actor;
    ACTOR_BUFFERS actor_buffers;
    DATASET dataset;
    rlt::malloc(device, rng);
    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffers);
    rlt::malloc(device, dataset);
    rlt::init_weights(device, actor, rng);
    rlt::set_all(device, dataset.scalar_data, 0);
    rlt::init(device, rng, 199);
    rlt::init(device, environment);
    rlt::init(device, runner, environment, rng);


    rlt::collect(device, dataset, runner, runner_buffer, environment, actor, actor_buffers, rng);
    rlt::print(device, dataset.scalar_data);
    rlt::collect(device, dataset, runner, runner_buffer, environment, actor, actor_buffers, rng);
    rlt::print(device, dataset.scalar_data);
    rlt::collect(device, dataset, runner, runner_buffer, environment, actor, actor_buffers, rng);
    rlt::print(device, dataset.scalar_data);
    ENVIRONMENT::State states[ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS];
    ENVIRONMENT::Parameters env_parameters[ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS];
    for(TI env_i = 0; env_i < ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; env_i++){
        states[env_i] = get(device, runner.states, env_i);
        env_parameters[env_i] = get(device, runner.env_parameters, env_i);
    }
    rlt::collect(device, dataset, runner, runner_buffer, environment, actor, actor_buffers, rng);
    for(TI env_i = 0; env_i < ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; env_i++){
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            TI pos = step_i * ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS + env_i;
            {
                rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::Observation::DIM>> observation;
                rlt::malloc(device, observation);
                auto& rng_state = get(rng.states, 0, env_i);
                rlt::observe(device, get_ref(device, environment.environments, env_i), env_parameters[env_i], states[env_i], typename ENVIRONMENT::Observation{}, observation, rng_state);
                auto all_obs_matrix = rlt::matrix_view(device, dataset.all_observations);
                auto observation_runner = rlt::view<DEVICE, decltype(all_obs_matrix)::SPEC, 1, ENVIRONMENT::Observation::DIM>(device, all_obs_matrix, pos, 0);
                auto abs_diff = rlt::abs_diff(device, observation, observation_runner);
                if(!get(dataset.truncated, pos, 0)){
//                    ASSERT_FLOAT_EQ(abs_diff, 0);
                }
                rlt::free(device, observation);
            }
            typename ENVIRONMENT::State next_state;
            auto action = rlt::view<DEVICE, decltype(dataset.actions)::SPEC, 1, ENVIRONMENT::ACTION_DIM>(device, dataset.actions, pos, 0);
            auto& rng_state = get(rng.states, 0, env_i);
            step(device, get_ref(device, environment.environments, env_i), env_parameters[env_i], states[env_i], action, next_state, rng_state);
            states[env_i] = next_state;
        }
    }
    std::string FILE_PATH = "test_rl_components_on_policy_runner_dataset.h5";
    {
        auto file = rl_tools::persist::backends::hdf5::File(FILE_PATH, rl_tools::persist::backends::hdf5::Mode::WRITE);
        auto dataset_group = rlt::create_group(device, file, "dataset");
        rlt::save(device, dataset, dataset_group);
    }

    {
        auto file = rl_tools::persist::backends::hdf5::File(FILE_PATH, rl_tools::persist::backends::hdf5::Mode::READ);
        DATASET loaded;
        rlt::malloc(device, loaded);
        auto dataset_group = rlt::get_group(device, file, "dataset");
        rlt::load(device, loaded, dataset_group);
        auto abs_diff = rlt::abs_diff(device, loaded.scalar_data, dataset.scalar_data);
        ASSERT_FLOAT_EQ(0, abs_diff);
    }



}
