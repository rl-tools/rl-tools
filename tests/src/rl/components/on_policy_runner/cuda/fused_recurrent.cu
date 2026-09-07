#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_mux.h>
#include <rl_tools/rl/algorithms/ppo/operations_collection.h>
#include <gtest/gtest.h>
#include <metra/metra.h>

namespace {
    using namespace rl_tools;
    using CPU = devices::DefaultCPU;
    using GPU = devices::DefaultCUDA;
    using TI = unsigned int;
    using TP = numeric_types::Policy<float>;
    constexpr TI N = 37;
    constexpr TI STEPS = 8;
    using PENDULUM = rl::environments::Pendulum<rl::environments::pendulum::Specification<float, TI>>;
    using L2F = rl::environments::Multirotor<rl::environments::l2f::Specification<float, TI>>;
    using RNG = devices::generic::random::ArrayENGINE<devices::generic::random::ArraySpecification<TI, N>>;

    template <bool TRUNCATE, typename ENV = PENDULUM>
    void compare_collection() {
        using BATCH = rl::environments::batch::Independent<rl::environments::batch::Specification<ENV, N>>;
        using GRU = nn::layers::gru::BindConfiguration<nn::layers::gru::Configuration<TP, TI, 8>>;
        using MLP = nn_models::mlp_unconditional_stddev::BindConfiguration<nn_models::mlp::Configuration<TP, TI, ENV::ACTION_DIM, 2, 8, nn::activation_functions::TANH, nn::activation_functions::IDENTITY>>;
        using ACTOR = nn_models::sequential::Build<nn::capability::Forward<>, nn_models::sequential::Module<GRU, nn_models::sequential::Module<MLP>>, tensor::Shape<TI, STEPS, N, ENV::Observation::DIM>>;
        using RS = rl::components::on_policy_runner::Specification<TP, BATCH, typename ACTOR::template State<>, typename ENV::Observation, typename ENV::ObservationPrivileged, float, float, 3, TRUNCATE>;
        using DS = rl::components::on_policy_runner::DatasetSpecification<RS, STEPS>;
        CPU cpu;
        GPU gpu;
        init(gpu);
        ACTOR initial_actor, actors[2];
        typename ACTOR::template Buffer<> actor_buffers[2];
        typename ACTOR::template State<> host_states[2];
        RNG initial_rng, rngs[2], host_rngs[2];
        BATCH environments[2];
        rl::components::OnPolicyRunner<RS> runners[2];
        rl::components::on_policy_runner::Buffer<RS> buffers[2];
        rl::components::on_policy_runner::Dataset<DS> datasets[2], host_datasets[2];
        malloc(cpu, initial_rng); init(cpu, initial_rng, 11);
        malloc(cpu, initial_actor); init_weights(cpu, initial_actor, initial_rng);
        init(cpu, initial_rng, 17);
        for(TI i = 0; i < 2; i++) {
            malloc(gpu, actors[i]); copy(cpu, gpu, initial_actor, actors[i]);
            malloc(gpu, actor_buffers[i]); malloc(cpu, host_states[i]);
            malloc(gpu, rngs[i]); copy(cpu, gpu, initial_rng, rngs[i]); malloc(cpu, host_rngs[i]);
            malloc(gpu, environments[i]); init(gpu, environments[i]);
            malloc(gpu, runners[i]); init(gpu, runners[i], environments[i], rngs[i]);
            malloc(gpu, buffers[i]); malloc(gpu, datasets[i]); malloc(cpu, host_datasets[i]);
            set_all(gpu, datasets[i].scalar_data, 0.0f);
        }
        for(TI rollout = 0; rollout < 3; rollout++) {
            collect(gpu, datasets[0], runners[0], buffers[0], environments[0], actors[0], actor_buffers[0], rngs[0]);
            if constexpr(TRUNCATE) {
                set_all(gpu, runners[1].reset, true);
                set_all(gpu, runners[1].episode_step, TI(0));
                sample_initial_parameters(gpu, environments[1], runners[1].env_parameters, runners[1].reset, rngs[1]);
                sample_initial_state(gpu, environments[1], runners[1].env_parameters, runners[1].states, runners[1].reset, rngs[1]);
            }
            prologue<GPU, DS, RS, BATCH, RNG>(gpu, datasets[1], runners[1], environments[1], rngs[1]);
            for(TI t = 0; t < STEPS; t++) {
                interlude(gpu, datasets[1], runners[1], buffers[1], actors[1], actor_buffers[1], rngs[1], t);
                epilogue<GPU, DS, RS, BATCH, RNG>(gpu, datasets[1], runners[1], buffers[1], environments[1], rngs[1], t);
            }
            for(TI i = 0; i < 2; i++) {
                copy(gpu, cpu, datasets[i].scalar_data, host_datasets[i].scalar_data);
                copy(gpu, cpu, datasets[i].all_observations, host_datasets[i].all_observations);
                copy(gpu, cpu, datasets[i].all_observations_privileged, host_datasets[i].all_observations_privileged);
                copy(gpu, cpu, runners[i].policy_state, host_states[i]);
                copy(gpu, cpu, rngs[i], host_rngs[i]);
            }
            EXPECT_EQ(abs_diff(cpu, host_datasets[0], host_datasets[1]), 0);
            EXPECT_EQ(abs_diff(cpu, host_states[0], host_states[1]), 0);
            EXPECT_EQ(abs_diff(cpu, host_rngs[0], host_rngs[1]), 0);
        }
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for(TI i = 0; i < 2; i++) {
            free(gpu, actors[i]); free(gpu, actor_buffers[i]); free(cpu, host_states[i]);
            free(gpu, rngs[i]); free(cpu, host_rngs[i]); free(gpu, environments[i]);
            free(gpu, runners[i]); free(gpu, buffers[i]); free(gpu, datasets[i]); free(cpu, host_datasets[i]);
        }
        free(cpu, initial_actor); free(cpu, initial_rng);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
        EXPECT_EQ(cudnnDestroy(gpu.cudnn_handle), CUDNN_STATUS_SUCCESS);
#endif
        EXPECT_EQ(cublasDestroy(gpu.handle), CUBLAS_STATUS_SUCCESS);
        EXPECT_EQ(cudaStreamDestroy(gpu.stream), cudaSuccess);
        metra::log("on_policy_runner/fused/recurrent_mismatches", ::testing::Test::HasFailure() ? 1.0 : 0.0);
    }
}

TEST(RL_TOOLS_ON_POLICY_RUNNER_FUSED, RECURRENT_COLLECTION) { compare_collection<false>(); }
TEST(RL_TOOLS_ON_POLICY_RUNNER_FUSED, RECURRENT_FORCED_BOUNDARIES) { compare_collection<true>(); }
TEST(RL_TOOLS_ON_POLICY_RUNNER_FUSED, L2F_RECURRENT_COLLECTION) { compare_collection<false, L2F>(); }
TEST(RL_TOOLS_ON_POLICY_RUNNER_FUSED, L2F_RECURRENT_FORCED_BOUNDARIES) { compare_collection<true, L2F>(); }

TEST(RL_TOOLS_ON_POLICY_RUNNER_FUSED, GRU_CRITIC_BOOTSTRAP){
    using namespace rl_tools;
    using BATCH = rl::environments::batch::Independent<rl::environments::batch::Specification<PENDULUM, N>>;
    using GRU = nn::layers::gru::BindConfiguration<nn::layers::gru::Configuration<TP, TI, 1, nn::parameters::groups::Normal, false>>;
    using CRITIC = nn_models::sequential::Build<nn::capability::Forward<>, nn_models::sequential::Module<GRU>, tensor::Shape<TI, 1, N, PENDULUM::ObservationPrivileged::DIM>>;
    using RS = rl::components::on_policy_runner::Specification<TP, BATCH, typename CRITIC::template State<>>;
    using DS = rl::components::on_policy_runner::DatasetSpecification<RS, STEPS>;
    CPU cpu;
    GPU gpu;
    init(gpu);
    CRITIC critic_cpu, critic_gpu;
    rl::algorithms::ppo::CollectionBuffer<CRITIC, DS> buffer_cpu, buffer_gpu;
    rl::components::on_policy_runner::Dataset<DS> dataset_cpu, dataset_gpu, result;
    rl::components::on_policy_runner::Buffer<RS> next_cpu, next_gpu;
    RNG rng;
    malloc(cpu, rng); init(cpu, rng, 17);
    malloc(cpu, critic_cpu); malloc(gpu, critic_gpu);
    auto& layer = get_first_layer(critic_cpu);
    set_all(cpu, layer.weights_input.parameters, 0.0f);
    set_all(cpu, layer.weights_hidden.parameters, 0.0f);
    set_all(cpu, layer.biases_input.parameters, 0.0f);
    set_all(cpu, layer.biases_hidden.parameters, 0.0f);
    set_all(cpu, layer.initial_hidden_state.parameters, 0.0f);
    set(cpu, layer.weights_input.parameters, 1.0f, 2, 0);
    copy(cpu, gpu, critic_cpu, critic_gpu);
    malloc(cpu, buffer_cpu); malloc(gpu, buffer_gpu);
    malloc(cpu, dataset_cpu); malloc(gpu, dataset_gpu); malloc(cpu, result);
    malloc(cpu, next_cpu); malloc(gpu, next_gpu);
    set_all(cpu, dataset_cpu.scalar_data, 0.0f);
    set_all(cpu, dataset_cpu.all_observations_privileged, 0.0f);
    for(TI t = 0; t < STEPS; t++) for(TI i = 0; i < N; i++){
        set(dataset_cpu.reset, t * N + i, 0, (t + i) % 3 == 0);
        set(cpu, dataset_cpu.all_observations_privileged, 0.1f * (t + 1), t * N + i, 0);
    }
    copy(cpu, gpu, dataset_cpu.scalar_data, dataset_gpu.scalar_data);
    copy(cpu, gpu, dataset_cpu.all_observations_privileged, dataset_gpu.all_observations_privileged);
    float hidden[N]{};
    for(TI t = 0; t < STEPS; t++){
        set_all(cpu, next_cpu.next_observations_privileged, 0.0f);
        for(TI i = 0; i < N; i++) set(cpu, next_cpu.next_observations_privileged, 0.1f * (t + 1) + 0.05f * (i + 1), i, 0);
        copy(cpu, gpu, next_cpu.next_observations_privileged, next_gpu.next_observations_privileged);
        evaluate_values(cpu, dataset_cpu, critic_cpu, buffer_cpu, rng, t);
        evaluate_values(gpu, dataset_gpu, critic_gpu, buffer_gpu, rng, t);
        evaluate_bootstrap_values(cpu, dataset_cpu, next_cpu.next_observations_privileged, critic_cpu, buffer_cpu, rng, t);
        evaluate_bootstrap_values(gpu, dataset_gpu, next_gpu.next_observations_privileged, critic_gpu, buffer_gpu, rng, t);
        copy(gpu, cpu, dataset_gpu.scalar_data, result.scalar_data);
        for(TI i = 0; i < N; i++){
            if(t == 0 || (t + i) % 3 == 0) hidden[i] = 0;
            hidden[i] = 0.5f * std::tanh(0.1f * (t + 1)) + 0.5f * hidden[i];
            float bootstrap = 0.5f * std::tanh(0.1f * (t + 1) + 0.05f * (i + 1)) + 0.5f * hidden[i];
            EXPECT_NEAR(get(result.values, t * N + i, 0), hidden[i], 1e-6f);
            EXPECT_NEAR(get(result.bootstrap_values, t * N + i, 0), bootstrap, 1e-6f);
            EXPECT_NEAR(get(dataset_cpu.bootstrap_values, t * N + i, 0), bootstrap, 1e-6f);
        }
    }
    free(cpu, next_cpu); free(gpu, next_gpu); free(cpu, result);
    free(cpu, dataset_cpu); free(gpu, dataset_gpu);
    free(cpu, buffer_cpu); free(gpu, buffer_gpu);
    free(cpu, critic_cpu); free(gpu, critic_gpu); free(cpu, rng);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
    EXPECT_EQ(cudnnDestroy(gpu.cudnn_handle), CUDNN_STATUS_SUCCESS);
#endif
    EXPECT_EQ(cublasDestroy(gpu.handle), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cudaStreamDestroy(gpu.stream), cudaSuccess);
}
