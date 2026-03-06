#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/rl/environments/reacher/operations_generic.h>

// Dense reacher checkpoint
#include "../../../../experiments/2026-03-06_09-24-11/ac6f68c_zoo_environment_algorithm/reacher-v0_ppo/0000/steps/000000000000240/checkpoint.h"

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = typename DEVICE::index_t;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;

struct OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>{
    static constexpr T ALPHA = 1e-3;
};

#include <gtest/gtest.h>

TEST(RL_TOOLS_RL_ENVIRONMENTS_REACHER, DISTILLATION) {
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    constexpr TI BATCH = 32;
    constexpr TI H = 84, W = 84, C = 3;
    constexpr TI ACTION_DIM = 2;

    // Dense teacher from checkpoint (change batch size to BATCH)
    using DENSE_TYPE = rl_tools::checkpoint::actor::TYPE::CHANGE_BATCH_SIZE<TI, BATCH>;
    constexpr auto dense_model = rl_tools::checkpoint::actor::factory_function<DENSE_TYPE>();
    typename DENSE_TYPE::template Buffer<> dense_buffer;
    rlt::malloc(device, dense_buffer);

    // Visual student model (NatureCNN)
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH, H, W, C>;
    using CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 32, 8, 8, 4, 4, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV1 = rlt::nn::layers::conv2d::BindConfiguration<CONV1_CONFIG>;
    using CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 64, 4, 4, 2, 2, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV2 = rlt::nn::layers::conv2d::BindConfiguration<CONV2_CONFIG>;
    using CONV3_CONFIG = rlt::nn::layers::conv2d::Configuration<TYPE_POLICY, TI, 64, 3, 3, 1, 1, 0, 0, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CONV3 = rlt::nn::layers::conv2d::BindConfiguration<CONV3_CONFIG>;
    using FLATTEN_CONFIG = rlt::nn::layers::flatten::Configuration<TYPE_POLICY, TI>;
    using FLATTEN = rlt::nn::layers::flatten::BindConfiguration<FLATTEN_CONFIG>;
    using MLP_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, ACTION_DIM, 2, 512, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using MLP = rlt::nn_models::mlp::BindConfiguration<MLP_CONFIG>;

    using MODULE_CHAIN = rlt::nn_models::sequential::Module<CONV1, rlt::nn_models::sequential::Module<CONV2, rlt::nn_models::sequential::Module<CONV3, rlt::nn_models::sequential::Module<FLATTEN, rlt::nn_models::sequential::Module<MLP>>>>>;
    using CAPA = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
    using VISUAL_MODEL = rlt::nn_models::sequential::Build<CAPA, MODULE_CHAIN, INPUT_SHAPE>;

    VISUAL_MODEL visual_model;
    typename VISUAL_MODEL::template Buffer<true> visual_buffer;

    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, OPTIMIZER_PARAMETERS>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
    OPTIMIZER optimizer;

    rlt::malloc(device, visual_model);
    rlt::malloc(device, visual_buffer);
    rlt::malloc(device, optimizer);
    rlt::init(device, optimizer);
    rlt::init_weights(device, visual_model, rng);
    rlt::reset_optimizer_state(device, optimizer, visual_model);

    // Environment
    using ENV_SPEC = rlt::rl::environments::reacher::Specification<T, TI>;
    using ENV = rlt::rl::environments::Reacher<ENV_SPEC>;
    ENV env;
    ENV::Parameters params;

    // Tensors
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> image_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH, 4>>> dense_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH, ACTION_DIM>>> target_actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, BATCH, ACTION_DIM>>> d_output;

    rlt::malloc(device, image_input);
    rlt::malloc(device, dense_input);
    rlt::malloc(device, target_actions);
    rlt::malloc(device, d_output);

    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;

    // Closed-loop evaluation helpers
    using VISUAL_ENV_SPEC = rlt::rl::environments::reacher::Specification<T, TI>;
    using VISUAL_ENV = rlt::rl::environments::ReacherVisual<VISUAL_ENV_SPEC>;
    VISUAL_ENV visual_env;
    VISUAL_ENV::Parameters visual_params;

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, 4>>> dense_obs_single;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ACTION_DIM>>> dense_action_single;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, H, W, C>>> visual_obs_single;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, ACTION_DIM>>> visual_action_single;
    rlt::malloc(device, dense_obs_single);
    rlt::malloc(device, dense_action_single);
    rlt::malloc(device, visual_obs_single);
    rlt::malloc(device, visual_action_single);

    constexpr TI NUM_EVAL_EPISODES = 20;

    auto evaluate_closed_loop = [&](){
        // Dense teacher
        T dense_total_return = 0;
        for(TI ep = 0; ep < NUM_EVAL_EPISODES; ep++){
            ENV::State state;
            rlt::sample_initial_state(device, env, params, state, rng);
            T episode_return = 0;
            for(TI step = 0; step < ENV::EPISODE_STEP_LIMIT; step++){
                auto obs_flat = rlt::view_memory<rlt::tensor::Shape<TI, 4>>(device, dense_obs_single);
                auto obs_matrix = rlt::matrix_view(device, obs_flat);
                rlt::observe(device, env, params, state, rlt::rl::environments::reacher::ObservationDense<TI>{}, obs_matrix, rng);
                rlt::evaluate(device, dense_model, dense_obs_single, dense_action_single, dense_buffer, rng, eval_mode);
                auto action_flat = rlt::view_memory<rlt::tensor::Shape<TI, ACTION_DIM>>(device, dense_action_single);
                auto action_matrix = rlt::matrix_view(device, action_flat);
                ENV::State next_state;
                rlt::step(device, env, params, state, action_matrix, next_state, rng);
                T r = rlt::reward(device, env, params, state, action_matrix, next_state, rng);
                episode_return += r;
                state = next_state;
                if(rlt::terminated(device, env, params, state, rng)){ break; }
            }
            dense_total_return += episode_return;
        }

        // Visual student
        T visual_total_return = 0;
        for(TI ep = 0; ep < NUM_EVAL_EPISODES; ep++){
            VISUAL_ENV::State state;
            rlt::sample_initial_state(device, visual_env, visual_params, state, rng);
            T episode_return = 0;
            for(TI step = 0; step < VISUAL_ENV::EPISODE_STEP_LIMIT; step++){
                auto obs_flat = rlt::view_memory<rlt::tensor::Shape<TI, H * W * C>>(device, visual_obs_single);
                auto obs_matrix = rlt::matrix_view(device, obs_flat);
                rlt::observe(device, visual_env, visual_params, state, rlt::rl::environments::reacher::ObservationImage<TI, H, W>{}, obs_matrix, rng);
                rlt::evaluate(device, visual_model, visual_obs_single, visual_action_single, visual_buffer, rng, eval_mode);
                auto action_flat = rlt::view_memory<rlt::tensor::Shape<TI, ACTION_DIM>>(device, visual_action_single);
                auto action_matrix = rlt::matrix_view(device, action_flat);
                VISUAL_ENV::State next_state;
                rlt::step(device, visual_env, visual_params, state, action_matrix, next_state, rng);
                T r = rlt::reward(device, visual_env, visual_params, state, action_matrix, next_state, rng);
                episode_return += r;
                state = next_state;
                if(rlt::terminated(device, visual_env, visual_params, state, rng)){ break; }
            }
            visual_total_return += episode_return;
        }
        return std::make_pair(dense_total_return / NUM_EVAL_EPISODES, visual_total_return / NUM_EVAL_EPISODES);
    };

    constexpr TI NUM_ITERATIONS = 10000;

    for(TI iter = 0; iter < NUM_ITERATIONS; iter++){
        // Sample random states and generate observations
        for(TI b = 0; b < BATCH; b++){
            ENV::State state;
            rlt::sample_initial_state(device, env, params, state, rng);

            auto dense_step = rlt::view(device, dense_input, (TI)0);
            auto dense_slice = rlt::view(device, dense_step, b);
            auto dense_flat = rlt::view_memory<rlt::tensor::Shape<TI, 4>>(device, dense_slice);
            auto dense_matrix = rlt::matrix_view(device, dense_flat);
            rlt::observe(device, env, params, state, rlt::rl::environments::reacher::ObservationDense<TI>{}, dense_matrix, rng);

            auto image_step = rlt::view(device, image_input, (TI)0);
            auto image_batch = rlt::view(device, image_step, b);
            auto image_flat = rlt::view_memory<rlt::tensor::Shape<TI, H * W * C>>(device, image_batch);
            auto image_matrix = rlt::matrix_view(device, image_flat);
            rlt::observe(device, env, params, state, rlt::rl::environments::reacher::ObservationImage<TI, H, W>{}, image_matrix, rng);
        }

        rlt::evaluate(device, dense_model, dense_input, target_actions, dense_buffer, rng, eval_mode);

        rlt::zero_gradient(device, visual_model);
        rlt::forward(device, visual_model, image_input, visual_buffer, rng);

        auto visual_output = rlt::output(device, visual_model);
        T loss = 0;
        for(TI b = 0; b < BATCH; b++){
            for(TI a = 0; a < ACTION_DIM; a++){
                T pred = rlt::get(device, visual_output, 0, b, a);
                T target = rlt::get(device, target_actions, 0, b, a);
                T diff = pred - target;
                loss += diff * diff;
                rlt::set(device, d_output, (T)(2.0 * diff / (BATCH * ACTION_DIM)), 0, b, a);
            }
        }
        loss /= (BATCH * ACTION_DIM);

        rlt::backward(device, visual_model, image_input, d_output, visual_buffer);
        rlt::step(device, optimizer, visual_model);

        if(iter % 50 == 0){
            auto [dense_return, visual_return] = evaluate_closed_loop();
            std::cout << "Iteration " << iter << ", MSE: " << loss << ", Dense return: " << dense_return << ", Visual return: " << visual_return << std::endl;
        }
    }

    auto [final_dense_return, final_visual_return] = evaluate_closed_loop();
    std::cout << "Final: Dense return = " << final_dense_return << ", Visual return = " << final_visual_return << std::endl;
    EXPECT_GT(final_visual_return, final_dense_return * 0.5);

    rlt::free(device, dense_obs_single);
    rlt::free(device, dense_action_single);
    rlt::free(device, visual_obs_single);
    rlt::free(device, visual_action_single);
}
