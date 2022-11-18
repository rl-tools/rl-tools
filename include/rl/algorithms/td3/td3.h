#include "replay_buffer.h"
#include <random>
#include <nn/nn.h>
using namespace layer_in_c;


template <typename T, typename CRITIC, int OBSERVATION_DIM, int ACTION_DIM, int CAPACITY, int BATCH_SIZE, typename RNG>
void train_critic(CRITIC critic, ReplayBuffer<T, OBSERVATION_DIM, ACTION_DIM, CAPACITY> replay_buffer, RNG& rng) {
    T loss = 0;
    zero_gradient(critic);
    for (int sample_i=0; sample_i < BATCH_SIZE; sample_i++){
        T input[OBSERVATION_DIM];
        T output[ACTION_DIM];
//        standardise<T,  OBSERVATION_DIM>(X_train[batch_i * batch_size + sample_i].data(), X_mean.data(), X_std.data(), input);
//        standardise<T, ACTION_DIM>(Y_train[batch_i * batch_size + sample_i].data(), Y_mean.data(), Y_std.data(), output);

        int sample_distribution = std::uniform_int_distribution<uint32_t>(0, (replay_buffer.full ? CAPACITY : replay_buffer.position) - 1)(rng);
        forward(critic, input);
        T d_loss_d_output[ACTION_DIM];
        nn::loss_functions::d_mse_d_x<T, ACTION_DIM>(critic.output_layer.output, output, d_loss_d_output);
        loss += nn::loss_functions::mse<T, ACTION_DIM>(critic.output_layer.output, output);

        T d_input[OBSERVATION_DIM];
        backward(critic, input, d_loss_d_output, d_input);
    }
    loss /= BATCH_SIZE;
    update(critic, batch_i + 1, batch_size);
    std::cout << "epoch_i " << epoch_i << " batch_i " << batch_i << " loss: " << loss << std::endl;
}