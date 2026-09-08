#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/operations_cpu.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/operations_generic_mixed_imitation.h>
#include <gtest/gtest.h>
#include <limits>
#include <metra/metra.h>

namespace{
    namespace rlt = rl_tools;
    namespace mixed = rlt::rl::algorithms::ppo::mixed_imitation;
    using DEVICE = rlt::devices::DefaultCPU;
    using TI = DEVICE::index_t;
    using T = double;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    struct Parameters: mixed::DefaultParameters<TYPE_POLICY, TI>{
        static constexpr T IMITATION_WEIGHT = 2.5;
        static constexpr T OUTPUT_GRADIENT_NORM_RATIO = 0.3;
        static constexpr T MAX_IMITATION_WEIGHT = 20;
    };
    template <TI IL> using Spec = mixed::Specification<TYPE_POLICY, TI, 8, 4, IL, Parameters>;
    using Matrix = rlt::Matrix<rlt::matrix::Specification<T, TI, 8, 2, false>>;
    constexpr T NAN_VALUE = std::numeric_limits<T>::quiet_NaN();

    template <typename SPEC>
    T loss(DEVICE& device, const Matrix& predictions, const Matrix& targets, T scale){
        T result = 0;
        for(TI row = 0; row < 8; row++){
            for(TI action = 0; action < 2; action++){
                T prediction = rlt::get(predictions, row, action);
                if(rlt::is_imitation_sample(device, SPEC{}, row)){
                    T error = prediction - rlt::get(targets, row, action);
                    result += Parameters::IMITATION_WEIGHT * error * error / (2 * SPEC::IMITATION_BATCH_SIZE * 2);
                }
                else{
                    T advantage = row % 2 == 0 ? -0.7 : 1.3;
                    T sampled_action = 0.1 * (row + action);
                    T error = prediction - sampled_action;
                    result += advantage * error * error / (2 * SPEC::RL_BATCH_SIZE);
                }
            }
        }
        return scale * result;
    }

    template <typename SPEC>
    void rl_gradient(DEVICE& device, const Matrix& predictions, Matrix& gradient, T scale){
        for(TI row = 0; row < 8; row++){
            for(TI action = 0; action < 2; action++){
                T advantage = row % 2 == 0 ? -0.7 : 1.3;
                T derivative = advantage * (rlt::get(predictions, row, action) - 0.1 * (row + action)) * scale / SPEC::RL_BATCH_SIZE;
                rlt::set(gradient, row, action, rlt::is_imitation_sample(device, SPEC{}, row) ? NAN_VALUE : derivative);
            }
        }
    }

    template <TI IL>
    void check_derivatives(){
        DEVICE device;
        using SPEC = Spec<IL>;
        Matrix predictions, targets, gradient;
        for(TI row = 0; row < 8; row++){
            for(TI action = 0; action < 2; action++){
                rlt::set(predictions, row, action, 0.17 * (row + 1) - 0.3 * action);
                rlt::set(targets, row, action, rlt::is_imitation_sample(device, SPEC{}, row) ? 0.2 * action : NAN_VALUE);
            }
        }
        constexpr T SCALE = 0.25;
        rl_gradient<SPEC>(device, predictions, gradient, SCALE);
        mixed::Metrics<SPEC> metrics;
        rlt::mix_policy_output_gradients(device, SPEC{}, predictions, targets, gradient, metrics, SCALE, rlt::Mode<mixed::FixedWeight<>>{});
        T il_norm_squared = 0;
        T rl_norm_squared = 0;
        for(TI row = 0; row < 8; row++){
            for(TI action = 0; action < 2; action++){
                T prediction = rlt::get(predictions, row, action);
                constexpr T EPS = 1e-5;
                rlt::set(predictions, row, action, prediction + EPS);
                T plus = loss<SPEC>(device, predictions, targets, SCALE);
                rlt::set(predictions, row, action, prediction - EPS);
                T minus = loss<SPEC>(device, predictions, targets, SCALE);
                rlt::set(predictions, row, action, prediction);
                T derivative = rlt::get(gradient, row, action);
                EXPECT_NEAR(derivative, (plus - minus) / (2 * EPS), 1e-10);
                if(rlt::is_imitation_sample(device, SPEC{}, row)){
                    il_norm_squared += derivative * derivative;
                }
                else{
                    rl_norm_squared += derivative * derivative;
                }
            }
        }
        EXPECT_NEAR(metrics.weighted_imitation_output_gradient_norm, std::sqrt(il_norm_squared), 1e-12);
        EXPECT_NEAR(metrics.rl_output_gradient_norm, std::sqrt(rl_norm_squared), 1e-12);
        EXPECT_DOUBLE_EQ(metrics.imitation_weight, Parameters::IMITATION_WEIGHT);
    }
}

TEST(PPO_MIXED_IMITATION, SUBSET_MEANS_FINITE_DIFFERENCES){
    check_derivatives<1>();
    check_derivatives<2>();
    check_derivatives<3>();
}

TEST(PPO_MIXED_IMITATION, EVERY_BATCH_AND_WARMUP_BOUNDARY){
    DEVICE device;
    for(TI batch = 0; batch < 5; batch++){
        TI il = 0;
        TI rl = 0;
        for(TI row = batch * 8; row < (batch + 1) * 8; row++){
            bool imitation = rlt::is_imitation_sample(device, Spec<1>{}, row);
            il += imitation ? 1 : 0;
            rl += imitation ? 0 : 1;
            EXPECT_EQ(rlt::use_teacher_action(device, Spec<1>{}, row, Parameters::TEACHER_FORCING_UPDATES - 1), imitation);
            EXPECT_FALSE(rlt::use_teacher_action(device, Spec<1>{}, row, Parameters::TEACHER_FORCING_UPDATES));
            EXPECT_FALSE(rlt::use_teacher_action(device, Spec<0>{}, row, 0));
        }
        EXPECT_EQ(il, 2);
        EXPECT_EQ(rl, 6);
    }
}

TEST(PPO_MIXED_IMITATION, FIXED_NORM_RATIO_AND_DEGENERATE_GRADIENTS){
    DEVICE device;
    using SPEC = Spec<1>;
    Matrix predictions, targets, gradient;
    rlt::set_all(device, predictions, (T)1);
    rlt::set_all(device, targets, (T)0);
    mixed::Metrics<SPEC> metrics;
    for(T scale: {0.1, 1.0}){
        rlt::set_all(device, gradient, scale);
        rlt::mix_policy_output_gradients(device, SPEC{}, predictions, targets, gradient, metrics, scale, rlt::Mode<mixed::FixedNormRatio<>>{});
        EXPECT_NEAR(metrics.output_gradient_norm_ratio, Parameters::OUTPUT_GRADIENT_NORM_RATIO, 1e-12);
        EXPECT_TRUE(metrics.ratio_valid);
        EXPECT_FALSE(metrics.balancing_fallback);
        EXPECT_FALSE(metrics.weight_clamped);
        EXPECT_DOUBLE_EQ(rlt::get(gradient, 1, 0), scale);
    }
    rlt::set_all(device, gradient, (T)1e4);
    rlt::mix_policy_output_gradients(device, SPEC{}, predictions, targets, gradient, metrics, 1, rlt::Mode<mixed::FixedNormRatio<>>{});
    EXPECT_TRUE(metrics.weight_clamped);
    EXPECT_DOUBLE_EQ(metrics.imitation_weight, Parameters::MAX_IMITATION_WEIGHT);
    rlt::set_all(device, gradient, (T)0);
    rlt::mix_policy_output_gradients(device, SPEC{}, predictions, targets, gradient, metrics, 1, rlt::Mode<mixed::FixedNormRatio<>>{});
    EXPECT_TRUE(metrics.balancing_fallback);
    EXPECT_FALSE(metrics.ratio_valid);
    EXPECT_DOUBLE_EQ(metrics.imitation_weight, Parameters::IMITATION_WEIGHT);
    EXPECT_GT(rlt::get(gradient, 0, 0), 0);
    rlt::set_all(device, predictions, (T)0);
    rlt::set_all(device, gradient, (T)1);
    rlt::mix_policy_output_gradients(device, SPEC{}, predictions, targets, gradient, metrics, 1, rlt::Mode<mixed::FixedNormRatio<>>{});
    EXPECT_TRUE(metrics.balancing_fallback);
    EXPECT_DOUBLE_EQ(metrics.weighted_imitation_output_gradient_norm, 0);
    EXPECT_DOUBLE_EQ(metrics.output_gradient_norm_ratio, 0);
    rlt::set_all(device, gradient, (T)0);
    rlt::mix_policy_output_gradients(device, SPEC{}, predictions, targets, gradient, metrics, 1, rlt::Mode<mixed::FixedNormRatio<>>{});
    EXPECT_TRUE(metrics.balancing_fallback);
    EXPECT_FALSE(metrics.ratio_valid);
    EXPECT_DOUBLE_EQ(metrics.output_gradient_norm_ratio, 0);
    metra::log("ppo_mixed_imitation/finite_difference_checks", 48.0);
}

TEST(PPO_MIXED_IMITATION, COMBINED_BACKWARD_ACCUMULATES_BOTH_LOSSES){
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    using CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 2, rlt::nn::activation_functions::IDENTITY>;
    using LAYER = rlt::nn::layers::dense::Layer<CONFIG, rlt::nn::capability::Gradient<rlt::nn::parameters::Gradient>, rlt::tensor::Shape<TI, 1, 8, 2>>;
    LAYER layer;
    LAYER::Buffer<true> buffer;
    rlt::malloc(device, layer);
    rlt::malloc(device, buffer);
    rlt::set_all(device, layer.weights.parameters, (T)0.3);
    rlt::set_all(device, layer.biases.parameters, (T)0.1);
    rlt::zero_gradient(device, layer);
    Matrix inputs[2], targets[2], predictions, gradient;
    for(TI batch = 0; batch < 2; batch++){
        for(TI row = 0; row < 8; row++){
            for(TI action = 0; action < 2; action++){
                rlt::set(inputs[batch], row, action, 0.13 * (row + action + batch));
                rlt::set(targets[batch], row, action, 0.2 * batch - 0.1 * action);
            }
        }
        rlt::forward(device, layer, inputs[batch], predictions, buffer, rng);
        rl_gradient<Spec<2>>(device, predictions, gradient, 0.5);
        mixed::Metrics<Spec<2>> metrics;
        rlt::mix_policy_output_gradients(device, Spec<2>{}, predictions, targets[batch], gradient, metrics, 0.5, rlt::Mode<mixed::FixedWeight<>>{});
        EXPECT_GT(metrics.rl_output_gradient_norm, 0);
        EXPECT_GT(metrics.imitation_output_gradient_norm, 0);
        rlt::backward(device, layer, inputs[batch], gradient, buffer);
    }
    auto total_loss = [&](){
        T result = 0;
        for(TI batch = 0; batch < 2; batch++){
            rlt::evaluate(device, layer, inputs[batch], predictions, buffer, rng);
            result += loss<Spec<2>>(device, predictions, targets[batch], 0.5);
        }
        return result;
    };
    T max_error = 0;
    for(TI output = 0; output < 2; output++){
        for(TI input = 0; input < 2; input++){
            constexpr T EPS = 1e-5;
            rlt::set(device, layer.weights.parameters, (T)0.3 + EPS, output, input);
            T plus = total_loss();
            rlt::set(device, layer.weights.parameters, (T)0.3 - EPS, output, input);
            T minus = total_loss();
            rlt::set(device, layer.weights.parameters, (T)0.3, output, input);
            T derivative = rlt::get(device, layer.weights.gradient, output, input);
            T numerical = (plus - minus) / (2 * EPS);
            EXPECT_NEAR(derivative, numerical, 1e-10);
            max_error = std::max(max_error, std::abs(derivative - numerical));
        }
    }
    metra::log("ppo_mixed_imitation/parameter_gradient_max_error", max_error);
    rlt::free(device, buffer);
    rlt::free(device, layer);
    rlt::free(device, rng);
}
