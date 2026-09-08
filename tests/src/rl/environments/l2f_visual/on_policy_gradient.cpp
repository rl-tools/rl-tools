#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include "../../algorithms/ppo/collection.h"
#include "training_hyperdrone_loss.h"
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/loss_functions/mse/operations_generic.h>

namespace rl_tools::test_on_policy_gradient{
    using T = double;
    using TI = unsigned int;
    using TP = numeric_types::Policy<T>;
    constexpr TI BATCH = 4;
    constexpr TI SAMPLES = 8;
    template<bool NORMALIZE>
    struct Parameters: rl::algorithms::ppo::DefaultParameters<TP, TI, BATCH>{
        static constexpr bool NORMALIZE_ADVANTAGE = NORMALIZE;
    };

    struct AdamParameters: nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TP>{
        static constexpr T ALPHA = 1e-4;
    };

    template<bool NORMALIZE>
    void check(){
        devices::DefaultCPU device;
        using MLP = nn_models::mlp_unconditional_stddev::BindConfiguration<nn_models::mlp::Configuration<TP, TI, 1, 2, 2, nn::activation_functions::TANH, nn::activation_functions::IDENTITY>>;
        using SHAPE = tensor::Shape<TI, 1, BATCH, 1>;
        using MODEL = nn_models::sequential::Build<nn::capability::Gradient<nn::parameters::Adam>, nn_models::sequential::Module<MLP>, SHAPE>;
        using PS = rl::algorithms::ppo::Specification<TP, TI, test_ppo_collection::Environment, MODEL, MODEL, Parameters<NORMALIZE>>;
        rl::algorithms::PPO<PS> ppo;
        rl::algorithms::ppo::Buffers<rl::algorithms::ppo::BufferSpecification<PS>> buffers;
        Matrix<matrix::Specification<T, TI, BATCH, 1>> actions, log_probs, advantages;
        nn::optimizers::Adam<nn::optimizers::adam::Specification<TP, TI, AdamParameters>> optimizer;
        malloc(device, optimizer);
        init(device, optimizer);
        malloc(device, ppo);
        typename decltype(device)::SPEC::RANDOM::ENGINE<> rng;
        malloc(device, rng);
        init(device, rng, 0);
        init_weights(device, ppo.actor, rng);
        zero_gradient(device, ppo.actor);
        reset_optimizer_state(device, optimizer, ppo.actor);
        malloc(device, buffers);
        malloc(device, actions);
        malloc(device, log_probs);
        malloc(device, advantages);
        T means[SAMPLES] = {-0.2, 0.4, 0.1, -0.5, 0.7, 0.2, -0.3, 0.6};
        const T samples[SAMPLES] = {0.1, -0.4, 0.8, 0.3, -0.2, 0.9, 0.1, -0.1};
        T weights[SAMPLES] = {-2, 0.5, 1, 3, 0.4, -1, 2, 0.3};
        if constexpr(NORMALIZE){
            for(TI batch = 0; batch < SAMPLES / BATCH; batch++){
                T mean = 0;
                for(TI i = 0; i < BATCH; i++){
                    mean += weights[batch * BATCH + i] / BATCH;
                }
                T variance = 0;
                for(TI i = 0; i < BATCH; i++){
                    T difference = weights[batch * BATCH + i] - mean;
                    variance += difference * difference / BATCH;
                }
                for(TI i = 0; i < BATCH; i++){
                    weights[batch * BATCH + i] = (weights[batch * BATCH + i] - mean) / (std::sqrt(variance) + Parameters<NORMALIZE>::ADVANTAGE_EPSILON);
                }
            }
        }
        const T raw_advantages[SAMPLES] = {-2, 0.5, 1, 3, 0.4, -1, 2, 0.3};
        T log_std = -0.7;
        auto objective = [&](){
            T total = 0;
            for(TI i = 0; i < SAMPLES; i++){
                const T z = (samples[i] - means[i]) / std::exp(log_std);
                const T log_probability = -0.5 * z * z - log_std - 0.5 * std::log(2 * math::PI<T>);
                const T entropy = log_std + 0.5 * std::log(2 * math::PI<T>) + 0.5;
                total -= (weights[i] * log_probability + Parameters<NORMALIZE>::ACTION_ENTROPY_COEFFICIENT * entropy) / SAMPLES;
            }
            return total;
        };
        auto& last = get_last_layer(ppo.actor);
        set_all(device, last.log_std.parameters, log_std);
        set_all(device, last.log_std.gradient, (T)0.25);
        T derivatives[SAMPLES];
        TI total_samples = 0;
        for(TI batch = 0; batch < SAMPLES / BATCH; batch++){
            for(TI i = 0; i < BATCH; i++){
                const TI row = batch * BATCH + i;
                set(buffers.current_batch_actions, i, 0, means[row]);
                set(actions, i, 0, samples[row]);
                set(log_probs, i, 0, (T)20);
                set(advantages, i, 0, raw_advantages[row]);
            }
            auto statistics = training_hyperdrone_actor_loss_gradient(device, ppo, buffers, actions, log_probs, advantages, utils::typing::integral_constant<TI, SAMPLES>{});
            total_samples += statistics.samples;
            EXPECT_EQ(statistics.clipped_samples, BATCH);
            for(TI i = 0; i < BATCH; i++){
                derivatives[batch * BATCH + i] = get(buffers.d_action_log_prob_d_action, i, 0);
            }
            EXPECT_EQ(get(device, last.log_std.parameters, 0), log_std);
            EXPECT_EQ(get(device, optimizer.age, 0), 1);
        }
        EXPECT_EQ(total_samples, SAMPLES);
        constexpr T h = 1e-6;
        for(TI i = 0; i < SAMPLES; i++){
            const T original = means[i];
            means[i] = original + h;
            const T plus = objective();
            means[i] = original - h;
            const T minus = objective();
            means[i] = original;
            EXPECT_NEAR(derivatives[i], (plus - minus) / (2 * h), 1e-8);
        }
        log_std += h;
        const T plus = objective();
        log_std -= 2 * h;
        const T minus = objective();
        EXPECT_NEAR(get(device, last.log_std.gradient, 0) - 0.25, (plus - minus) / (2 * h), 1e-8);
        const T gradient = get(device, last.log_std.gradient, 0) - 0.25;
        set(device, last.log_std.gradient, gradient, 0);
        const T expected = -0.7 - AdamParameters::ALPHA * gradient / (std::sqrt(std::max(gradient * gradient, AdamParameters::EPSILON_SQRT)) + AdamParameters::EPSILON);
        step(device, optimizer, ppo.actor);
        EXPECT_EQ(get(device, optimizer.age, 0), 2);
        EXPECT_NEAR(get(device, last.log_std.parameters, 0), expected, 1e-12);
        free(device, rng);
        free(device, optimizer);
        free(device, ppo);
        free(device, buffers);
        free(device, actions);
        free(device, log_probs);
        free(device, advantages);
    }
}

TEST(L2F_VISUAL_HYPERDRONE_ON_POLICY_GRADIENT, ACCUMULATION_FINITE_DIFFERENCE){
    rl_tools::test_on_policy_gradient::check<false>();
}
TEST(L2F_VISUAL_HYPERDRONE_ON_POLICY_GRADIENT, MINIBATCH_NORMALIZATION_FINITE_DIFFERENCE){
    rl_tools::test_on_policy_gradient::check<true>();
}

TEST(L2F_VISUAL_HYPERDRONE_ON_POLICY_GRADIENT, CRITIC_ACCUMULATION_FINITE_DIFFERENCE){
    using namespace rl_tools;
    using namespace test_on_policy_gradient;
    devices::DefaultCPU device;
    Matrix<matrix::Specification<T, TI, BATCH, 1>> values, targets, derivatives;
    malloc(device, values);
    malloc(device, targets);
    malloc(device, derivatives);
    const T bias = 0.4;
    T accumulated = 0;
    auto objective = [&](T prediction){
        T loss = 0;
        for(TI i = 0; i < SAMPLES; i++){
            const T difference = prediction - (T)i / SAMPLES;
            loss += 0.5 * difference * difference / SAMPLES;
        }
        return loss;
    };
    for(TI batch = 0; batch < SAMPLES / BATCH; batch++){
        set_all(device, values, bias);
        for(TI row = 0; row < BATCH; row++){
            set(targets, row, 0, (T)(batch * BATCH + row) / SAMPLES);
        }
        nn::loss_functions::mse::gradient(device, values, targets, derivatives, (T)0.5 * BATCH / SAMPLES);
        for(TI row = 0; row < BATCH; row++){
            accumulated += get(derivatives, row, 0);
        }
    }
    constexpr T h = 1e-6;
    EXPECT_NEAR(accumulated, (objective(bias + h) - objective(bias - h)) / (2 * h), 1e-10);
    free(device, values);
    free(device, targets);
    free(device, derivatives);
}
