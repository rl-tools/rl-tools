#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_SAMPLE_AND_SQUASH_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_SAMPLE_AND_SQUASH_OPERATIONS_GENERIC_H
#include "../../../nn/activation_functions.h"
#include "../../../utils/generic/typing.h"
#include "../../../containers.h"
#include "../../../nn/mode.h"


#include "layer.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn::layers::sample_and_squash{
        template <typename MODE>
        constexpr bool is_mode_external_noise(const MODE& mode){
            return false;
        }
        template <typename BASE_MODE>
        constexpr bool is_mode_external_noise(const mode::ExternalNoise<BASE_MODE> & mode){
            return true;
        }
        template <typename MODE>
        constexpr bool is_mode_sample(const MODE& mode){
            return false;
        }
        template <typename BASE_MODE>
        constexpr bool is_mode_sample(const mode::Sample<BASE_MODE> & mode){
            return true;
        }
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::sample_and_squash::LayerForward<SPEC>& layer){ }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn::layers::sample_and_squash::LayerForward<SPEC>& layer){ }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::sample_and_squash::LayerBackward<SPEC>& layer){
        malloc(device, static_cast<nn::layers::sample_and_squash::LayerForward<SPEC>&>(layer));
        malloc(device, layer.pre_squashing);
        malloc(device, layer.noise);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn::layers::sample_and_squash::LayerBackward<SPEC>& layer){
        free(device, static_cast<nn::layers::sample_and_squash::LayerForward<SPEC>&>(layer));
        free(device, layer.pre_squashing);
        free(device, layer.noise);
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer){
        malloc(device, static_cast<nn::layers::sample_and_squash::LayerBackward<SPEC>&>(layer));
        malloc(device, layer.log_alpha);
        malloc(device, layer.log_probabilities);
        malloc(device, layer.output);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer){
        free(device, static_cast<nn::layers::sample_and_squash::LayerBackward<SPEC>&>(layer));
        free(device, layer.log_alpha);
        free(device, layer.log_probabilities);
        free(device, layer.output);
    }
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, nn::layers::sample_and_squash::Buffer<SPEC>& buffer) {
        malloc(device, buffer.noise);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, nn::layers::sample_and_squash::Buffer<SPEC>& buffer) {
        free(device, buffer.noise);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, nn::layers::sample_and_squash::Buffer<SOURCE_SPEC>& source, nn::layers::sample_and_squash::Buffer<TARGET_SPEC>& target){
        copy(source_device, target_device, source.noise, target.noise);
    }
    template <typename DEVICE, typename SPEC, typename RNG>
    void init_weights(DEVICE& device, nn::layers::sample_and_squash::LayerForward<SPEC>& layer, RNG& rng){ }
    template <typename DEVICE, typename SPEC, typename RNG>
    void init_weights(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, RNG& rng){
        init_weights(device, static_cast<nn::layers::sample_and_squash::LayerForward<SPEC>&>(layer), rng);
        set(layer.log_alpha.parameters, 0, 0, math::log(typename DEVICE::SPEC::MATH{}, SPEC::PARAMETERS::ALPHA));
    }
    template <typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, rl_tools::nn::layers::sample_and_squash::LayerBackward<SPEC>& l) {
        set_all(device, l.pre_squashing, 0);
        set_all(device, l.noise, 0);
    }
    template <typename DEVICE, typename SPEC>
    void reset_forward_state(DEVICE& device, rl_tools::nn::layers::sample_and_squash::LayerGradient<SPEC>& l) {
        reset_forward_state(device, (rl_tools::nn::layers::sample_and_squash::LayerBackward<SPEC>&) l);
        set_all(device, l.log_probabilities, 0);
        set_all(device, l.output, 0);
    }
    template<typename DEVICE, typename SPEC>
    void zero_gradient(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer) {
        zero_gradient(device, layer.log_alpha);
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void update(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer){
        if constexpr(SPEC::PARAMETERS::ADAPTIVE_ALPHA){
            update(device, layer.log_alpha, optimizer);
        }
    }
    template<typename DEVICE, typename SPEC, typename OPTIMIZER>
    void _reset_optimizer_state(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, OPTIMIZER& optimizer) {
        _reset_optimizer_state(device, layer.log_alpha, optimizer);
    }

    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn::layers::sample_and_squash::LayerForward<SOURCE_SPEC>& source, nn::layers::sample_and_squash::LayerForward<TARGET_SPEC>& target){ }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn::layers::sample_and_squash::LayerBackward<SOURCE_SPEC>& source, nn::layers::sample_and_squash::LayerBackward<TARGET_SPEC>& target){
        copy(source_device, target_device, source.pre_squashing, target.pre_squashing);
        copy(source_device, target_device, source.noise, target.noise);
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  nn::layers::sample_and_squash::LayerGradient<SOURCE_SPEC>& source, nn::layers::sample_and_squash::LayerGradient<TARGET_SPEC>& target){
        copy(source_device, target_device, static_cast<const nn::layers::sample_and_squash::LayerBackward<SOURCE_SPEC>&>(source), static_cast<nn::layers::sample_and_squash::LayerBackward<TARGET_SPEC>&>(target));
        copy(source_device, target_device, source.log_probabilities, target.log_probabilities);
        copy(source_device, target_device, source.output, target.output);
        copy(source_device, target_device, source.log_alpha, target.log_alpha);
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    void sample(DEVICE& device, nn::layers::sample_and_squash::Buffer<SPEC>& buffer, RNG& rng) {
        randn(device, buffer.noise, rng);
    }
    template <typename DEVICE, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = nn::mode::Default>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_per_sample(const DEVICE& device, const nn::layers::sample_and_squash::LayerForward<SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>& buffer, RNG& rng, typename DEVICE::index_t row_i, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        using PARAMETERS = typename SPEC::PARAMETERS;
        T log_prob = 0;
        for(TI col_i = 0; col_i < SPEC::DIM; col_i++){
            T mean = get(input, row_i, col_i);
            T log_std = get(input, row_i, SPEC::DIM + col_i);
            T log_std_clipped = math::clamp(device.math, log_std, (T)PARAMETERS::LOG_STD_LOWER_BOUND, (T)PARAMETERS::LOG_STD_UPPER_BOUND);
            T std = math::exp(device.math, log_std_clipped);
            T noise;
            if constexpr(nn::layers::sample_and_squash::is_mode_external_noise(MODE())){
                noise = get(buffer.noise, row_i, col_i);
            }
            else{
                if constexpr(nn::layers::sample_and_squash::is_mode_sample(MODE())){
                    noise = random::normal_distribution::sample(device.random, (T)0, (T)1, rng);
                }
                else{
                    noise = 0;
                }
            }
//                set(layer.noise, row_i, col_i, noise);
            T sample;
            if constexpr(utils::typing::is_base_of_v<nn::mode::Inference, MODE>){
                sample = mean;
            }
            else{
                sample = mean + noise * std;
            }
//                set(layer.pre_squashing, row_i, col_i, sample);
            T squashed = math::tanh(device.math, sample);

            set(output, row_i, col_i, squashed);
//                set(layer.output, row_i, col_i, squashed);
            T one_minus_square_plus_eps = (1-squashed*squashed + SPEC::PARAMETERS::LOG_PROBABILITY_EPSILON);
            log_prob += random::normal_distribution::log_prob(device.random, mean, log_std_clipped, sample) - math::log(typename DEVICE::SPEC::MATH{}, one_minus_square_plus_eps);
        }
//            set(layer.log_probabilities, 0, row_i, log_prob);
    }
    template <typename DEVICE, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = nn::mode::Default>
    void evaluate(const DEVICE& device, const nn::layers::sample_and_squash::LayerForward<SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
        static_assert(INPUT_SPEC::COLS == 2*SPEC::DIM);
        static_assert(OUTPUT_SPEC::COLS == SPEC::DIM);
        static_assert(INPUT_SPEC::ROWS == OUTPUT_SPEC::ROWS);
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        using PARAMETERS = typename SPEC::PARAMETERS;
        for(TI row_i = 0; row_i < INPUT_SPEC::ROWS; row_i++){
            evaluate_per_sample(device, layer, input, output, buffer, rng, row_i, mode);
        }
    }
    template <typename DEVICE, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = nn::mode::Default>
    void forward(const DEVICE& device, nn::layers::sample_and_squash::LayerBackward<SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
        evaluate(device, layer, input, output, buffer, rng, mode);
    }
    template <typename DEVICE, typename SPEC, typename INPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = nn::mode::Default>
    RL_TOOLS_FUNCTION_PLACEMENT void forward_per_sample(const DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, const Matrix<INPUT_SPEC>& input, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>& buffer, RNG& rng, typename DEVICE::index_t row_i, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
        // copy of the evaluate but with the log_probabilities commented in
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        using PARAMETERS = typename SPEC::PARAMETERS;
        T log_prob = 0;
        for(TI col_i = 0; col_i < SPEC::DIM; col_i++){
            T mean = get(input, row_i, col_i);
            T log_std = get(input, row_i, SPEC::DIM + col_i);
            T log_std_clipped = math::clamp(device.math, log_std, (T)PARAMETERS::LOG_STD_LOWER_BOUND, (T)PARAMETERS::LOG_STD_UPPER_BOUND);
            T std = math::exp(device.math, log_std_clipped);
            T noise;
            if constexpr(nn::layers::sample_and_squash::is_mode_external_noise(MODE())){
                noise = get(buffer.noise, row_i, col_i);
            }
            else{
                if constexpr(nn::layers::sample_and_squash::is_mode_sample(MODE())){
                    noise = random::normal_distribution::sample(device.random, (T)0, (T)1, rng);
                }
                else{
                    noise = 0;
                }
            }
            set(layer.noise, row_i, col_i, noise);
            T sample;
            if constexpr(utils::typing::is_base_of_v<nn::mode::Inference, MODE>){
                sample = mean;
            }
            else{
                sample = mean + noise * std;
            }
            set(layer.pre_squashing, row_i, col_i, sample);
            T squashed = math::tanh(device.math, sample);

//            set(output, row_i, col_i, squashed);
            set(layer.output, row_i, col_i, squashed);
            T one_minus_square_plus_eps = (1-squashed*squashed + SPEC::PARAMETERS::LOG_PROBABILITY_EPSILON);
            log_prob += random::normal_distribution::log_prob(device.random, mean, log_std_clipped, sample) - math::log(typename DEVICE::SPEC::MATH{}, one_minus_square_plus_eps);
        }
        set(layer.log_probabilities, 0, row_i, log_prob);
    }
    template <typename DEVICE, typename SPEC, typename INPUT_SPEC, typename BUFFER_SPEC, typename RNG, typename MODE = nn::mode::Default>
    void forward(const DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, const Matrix<INPUT_SPEC>& input, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>& buffer, RNG& rng, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){
        static_assert(INPUT_SPEC::COLS == 2*SPEC::DIM);
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::T;
        using PARAMETERS = typename SPEC::PARAMETERS;
        for(TI row_i = 0; row_i < INPUT_SPEC::ROWS; row_i++){
            forward_per_sample(device, layer, input, buffer, rng, row_i, mode);
        }
    }
    template<typename DEVICE, typename SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = nn::mode::Default>
    void backward_input(DEVICE& device, const nn::layers::sample_and_squash::LayerBackward<SPEC>& layer, const Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>&, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}){

    }
    template<typename DEVICE, typename SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename BUFFER_SPEC, typename MODE = nn::mode::Default>
    void backward(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>&, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}) {

    }
    template<typename DEVICE, typename SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = nn::mode::Default>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T backward_full_per_sample(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>&, typename SPEC::T alpha, typename DEVICE::index_t batch_i, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI ACTION_DIM = SPEC::DIM;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
/*
        Gradient of the loss function:
        mu, std = policy(observation)
        action_sample = gaussian::sample(mu, std)
        action = tanh(action_sample)
        action_prob = gaussian::prob(mu, std, action_sample) * | d/d_action tanh^{-1}(action) |
                    = gaussian::prob(mu, std, action_sample) * | (d/d_action_sample tanh(action_sample))^{-1} |
                    = gaussian::prob(mu, std, action_sample) * | (d/d_action_sample tanh(action_sample))|^{-1}
                    = gaussian::prob(mu, std, action_sample) * ((1-tanh(action_sample)^2))^{-1}
        action_log_prob = gaussian::log_prob(mu, std, action_sample) - log(1-tanh(action_sample)^2))
        actor_loss = alpha  * action_log_prob - min(Q_1, Q_2);
        d/d_mu _actor_loss = alpha * d/d_mu action_log_prob - d/d_mu min(Q_1, Q_2)
        d/d_mu action_log_prob = d/d_mu gaussian::log_prob(mu, std, action_sample) + d/d_action_sample gaussian::log_prob(mu, std, action_sample) * d/d_mu action_sample - 1/(1-tanh(action_sample)^2) * (-2*tanh(action_sample))*(1-tanh(action_sample)^2) * d/d_mu action_sample)
                               = d/d_mu gaussian::log_prob(mu, std, action_sample) + d/d_action_sample gaussian::log_prob(mu, std, action_sample) * d/d_mu action_sample + 2*tanh(action_sample)) * d/d_mu action_sample
        d/d_std action_log_prob = d/d_std gaussian::log_prob(mu, std, action_sample) + d/d_action_sample gaussian::log_prob(mu, std, action_sample) * d/d_std action_sample + 2*tanh(action_sample) * d/d_std action_sample
        d/d_mu action_sample = 1
        d/d_std action_sample = noise
        d/d_mu min(Q_1, Q_2) = d/d_action min(Q_1, Q_2) * d/d_mu action
        d/d_mu action = d/d_action_sample tanh(action_sample) * d/d_mu action_sample
*/
        T d_alpha = 0;
        T entropy = 0;
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            T action = get(layer.output, batch_i, action_i); // tanh(action_sample)
            T d_mu = 0;
            T d_std = 0;
            T d_output_value = get(d_output, batch_i, action_i);
            T d_tanh_pre_activation = d_output_value * (1-action*action);
            d_mu = d_tanh_pre_activation;
            d_std = d_tanh_pre_activation * get(layer.noise, batch_i, action_i);
            T log_std_pre_clamp = get(input, batch_i, action_i + ACTION_DIM);
            T log_std_clamped = math::clamp(device.math, log_std_pre_clamp, (T)SPEC::PARAMETERS::LOG_STD_LOWER_BOUND, (T)SPEC::PARAMETERS::LOG_STD_UPPER_BOUND);
            T std = math::exp(typename DEVICE::SPEC::MATH{}, log_std_clamped);

            T d_log_std_clamped = std * d_std;

            T mu = get(input, batch_i, action_i);
            T action_sample = get(layer.pre_squashing, batch_i, action_i);
            T d_log_prob_d_mean = random::normal_distribution::d_log_prob_d_mean(device.random, mu, log_std_clamped, action_sample);
            T d_log_prob_d_sample = random::normal_distribution::d_log_prob_d_sample(device.random, mu, log_std_clamped, action_sample);
            // NOTE: The following needs to be divided by BATCH_SIZE (in contrast to the previous d_mu and d_std). d_mu and d_std are already taking into account the mean prior to the backward call of the critic. Thence the d_critic_X_input is already divided by BATCH_SIZE
            d_mu += alpha/BATCH_SIZE * (d_log_prob_d_mean + d_log_prob_d_sample + 2*action);

            T noise = get(layer.noise, batch_i, action_i);
            T d_log_prob_d_log_std = random::normal_distribution::d_log_prob_d_log_std(device.random, mu, log_std_clamped, action_sample);
            d_log_std_clamped += alpha/BATCH_SIZE * (d_log_prob_d_log_std + d_log_prob_d_sample * noise * std + 2*action * noise * std);
            T d_log_std = log_std_pre_clamp < SPEC::PARAMETERS::LOG_STD_LOWER_BOUND || log_std_pre_clamp > SPEC::PARAMETERS::LOG_STD_UPPER_BOUND ? 0 : d_log_std_clamped;

            set(d_input, batch_i, action_i, d_mu);
            set(d_input, batch_i, action_i + ACTION_DIM, d_log_std);

            T one_minus_action_square_plus_eps = (1-action*action + SPEC::PARAMETERS::LOG_PROBABILITY_EPSILON);
            T action_log_prob = random::normal_distribution::log_prob(device.random, mu, log_std_clamped, action_sample) - math::log(typename DEVICE::SPEC::MATH{}, one_minus_action_square_plus_eps);
            entropy += -action_log_prob;
        }
        d_alpha += entropy - SPEC::PARAMETERS::TARGET_ENTROPY;
        return alpha*d_alpha; // d_log_alpha
    }
    template<typename DEVICE, typename SPEC, typename INPUT_SPEC, typename D_OUTPUT_SPEC, typename D_INPUT_SPEC, typename BUFFER_SPEC, typename MODE = nn::mode::Default>
    void backward_full(DEVICE& device, nn::layers::sample_and_squash::LayerGradient<SPEC>& layer, const Matrix<INPUT_SPEC>& input, Matrix<D_OUTPUT_SPEC>& d_output, Matrix<D_INPUT_SPEC>& d_input, nn::layers::sample_and_squash::Buffer<BUFFER_SPEC>& buffer, const nn::Mode<MODE>& mode = nn::Mode<nn::mode::Default>{}) {
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI BATCH_SIZE = INPUT_SPEC::ROWS;
        T log_alpha = get(layer.log_alpha.parameters, 0, 0);
        T alpha = math::exp(typename DEVICE::SPEC::MATH{}, log_alpha);
        T d_log_alpha = 0;
        for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
            d_log_alpha += backward_full_per_sample(device, layer, input, d_output, d_input, buffer, alpha, batch_i, mode);
        }
        increment(layer.log_alpha.gradient, 0, 0, d_log_alpha/BATCH_SIZE);
    }
    template<typename SPEC>
    constexpr auto& output(nn::layers::sample_and_squash::LayerGradient<SPEC>& l){
        return l.output;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
