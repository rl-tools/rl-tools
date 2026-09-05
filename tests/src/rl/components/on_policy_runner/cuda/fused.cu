#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/random/operations_generic_array.h>
#include <rl_tools/rl/environments/observation.h>
#include <gtest/gtest.h>
#include <metra/metra.h>
#include <type_traits>

namespace rl_tools::rl::environments::test_fused {
    struct Parameters { unsigned int offset; unsigned int terminate_at; };
    struct State { unsigned int steps; float value; };
    template <bool ASYMMETRIC>
    struct Environment {
        using T = double;
        using TI = unsigned int;
        using Parameters = test_fused::Parameters;
        using State = test_fused::State;
        struct Observation { static constexpr TI DIM = 2; using SHAPE = tensor::Shape<TI, 2>; };
        struct Privileged { static constexpr TI DIM = 3; using SHAPE = tensor::Shape<TI, 3>; };
        using ObservationPrivileged = std::conditional_t<ASYMMETRIC, Privileged, Observation>;
        static constexpr TI N_AGENTS = 2;
        static constexpr TI ACTION_DIM = 4;
        static constexpr TI EPISODE_STEP_LIMIT = 0;
        TI calls;
    };
}
namespace rl_tools {
    template <typename DEVICE, bool A>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, rl::environments::test_fused::Environment<A>&) {}
    template <typename DEVICE, bool A>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, rl::environments::test_fused::Environment<A>&) {}
    template <typename DEVICE, bool A>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE&, rl::environments::test_fused::Environment<A>& env) { env.calls = 0; }
    template <typename DEVICE, bool A, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_parameters(DEVICE& device, rl::environments::test_fused::Environment<A>& env, rl::environments::test_fused::Parameters& parameters, RNG& rng) {
        parameters.offset = random::uniform_int_distribution(device.random, 0u, 1023u, rng);
        parameters.terminate_at = 2 + parameters.offset % 4;
        env.calls++;
    }
    template <typename DEVICE, bool A, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void sample_initial_state(DEVICE& device, rl::environments::test_fused::Environment<A>& env, const rl::environments::test_fused::Parameters& parameters, rl::environments::test_fused::State& state, RNG& rng) {
        state = {0, float(parameters.offset + random::uniform_int_distribution(device.random, 0u, 1023u, rng))};
        env.calls++;
    }
    template <typename DEVICE, bool A, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT double step(DEVICE& device, rl::environments::test_fused::Environment<A>& env, const rl::environments::test_fused::Parameters&, const rl::environments::test_fused::State& state, const Matrix<ACTION_SPEC>& action, rl::environments::test_fused::State& next_state, RNG& rng) {
        next_state = {state.steps + 1, state.value + float(random::uniform_int_distribution(device.random, 0u, 1023u, rng))};
        for(unsigned int i = 0; i < 4; i++) next_state.value += get(action, 0, i);
        env.calls++;
        return 0.01;
    }
    template <typename DEVICE, bool A, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT double reward(DEVICE& device, rl::environments::test_fused::Environment<A>& env, const rl::environments::test_fused::Parameters&, const rl::environments::test_fused::State& state, const Matrix<ACTION_SPEC>&, const rl::environments::test_fused::State& next_state, RNG& rng) {
        env.calls++;
        return double(next_state.value - state.value) + double(random::uniform_int_distribution(device.random, 0u, 1023u, rng)) / 30000001.0;
    }
    template <typename DEVICE, bool A, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT bool terminated(DEVICE& device, rl::environments::test_fused::Environment<A>& env, const rl::environments::test_fused::Parameters& parameters, const rl::environments::test_fused::State& state, RNG& rng) {
        random::uniform_int_distribution(device.random, 0u, 1023u, rng);
        env.calls++;
        return state.steps >= parameters.terminate_at;
    }
    template <typename DEVICE, bool A, typename OBSERVATION, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe(DEVICE& device, rl::environments::test_fused::Environment<A>& env, const rl::environments::test_fused::Parameters&, const rl::environments::test_fused::State& state, OBSERVATION, Matrix<OBS_SPEC>& output, RNG& rng) {
        set(output, 0, 0, state.value);
        set(output, 0, 1, float(random::uniform_int_distribution(device.random, 0u, 1023u, rng)));
        if constexpr(OBSERVATION::DIM == 3) set(output, 0, 2, float(state.steps));
        env.calls++;
    }
}

#ifdef RL_TOOLS_TEST_RUNNER_CUDA_DIRECT
#include <rl_tools/rl/components/on_policy_runner/operations_cuda.h>
#else
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#endif

namespace {
    using namespace rl_tools;
    using CPU = devices::DefaultCPU;
    using GPU = devices::DefaultCUDA;
    using TI = unsigned int;

    template <typename T> void equal_value(const T& a, const T& b) { EXPECT_EQ(a, b); }
    void equal_value(const rl::environments::test_fused::State& a, const rl::environments::test_fused::State& b) {
        EXPECT_EQ(a.steps, b.steps); EXPECT_EQ(a.value, b.value);
    }
    void equal_value(const rl::environments::test_fused::Parameters& a, const rl::environments::test_fused::Parameters& b) {
        EXPECT_EQ(a.offset, b.offset); EXPECT_EQ(a.terminate_at, b.terminate_at);
    }
    template <bool A> void equal_value(const rl::environments::test_fused::Environment<A>& a, const rl::environments::test_fused::Environment<A>& b) { EXPECT_EQ(a.calls, b.calls); }
    void equal_value(const devices::generic::random::PortableState& a, const devices::generic::random::PortableState& b) { EXPECT_EQ(a.state, b.state); }
    void equal_value(const curandState& a, const curandState& b) {
        EXPECT_EQ(a.d, b.d);
        for(TI i = 0; i < 5; i++) EXPECT_EQ(a.v[i], b.v[i]);
        EXPECT_EQ(a.boxmuller_flag, b.boxmuller_flag);
        EXPECT_EQ(a.boxmuller_flag_double, b.boxmuller_flag_double);
        if(a.boxmuller_flag) EXPECT_EQ(a.boxmuller_extra, b.boxmuller_extra);
        if(a.boxmuller_flag_double) EXPECT_EQ(a.boxmuller_extra_double, b.boxmuller_extra_double);
    }
    template <typename CONTAINER>
    void equal_container(CPU& cpu, GPU& gpu, const CONTAINER& a, const CONTAINER& b) {
        CONTAINER host_a, host_b;
        malloc(cpu, host_a); malloc(cpu, host_b);
        copy(gpu, cpu, a, host_a); copy(gpu, cpu, b, host_b);
        for(TI i = 0; i < CONTAINER::SPEC::SIZE; i++) equal_value(host_a._data[i], host_b._data[i]);
        free(cpu, host_a); free(cpu, host_b);
    }

    template <TI N, bool ASYMMETRIC, bool CURAND, bool MIXED_STORAGE = ASYMMETRIC, TI STEP_LIMIT = 3>
    void compare_phases() {
        using ENV = rl::environments::test_fused::Environment<ASYMMETRIC>;
        using BATCH = rl::environments::batch::Independent<rl::environments::batch::Specification<ENV, N>>;
        using POLICY_STATE = Tensor<tensor::Specification<float, TI, tensor::Shape<TI, 1>>>;
        using PRIV_T = std::conditional_t<MIXED_STORAGE, double, float>;
        using RS = rl::components::on_policy_runner::Specification<numeric_types::Policy<float>, BATCH, POLICY_STATE, typename ENV::Observation, typename ENV::ObservationPrivileged, float, PRIV_T, STEP_LIMIT>;
        using DS = rl::components::on_policy_runner::DatasetSpecification<RS, 8>;
        using RNG = std::conditional_t<CURAND, devices::random::CUDA::ENGINE<devices::random::CUDA::Specification<TI, N>>, devices::generic::random::ArrayENGINE<devices::generic::random::ArraySpecification<TI, N>>>;
        using MASK_SPEC = tensor::Specification<bool, TI, tensor::Shape<TI, N>, true, tensor::Stride<TI, 2>>;
        CPU cpu;
        GPU gpu;
        init(gpu);
        rl::components::OnPolicyRunner<RS> runners[2];
        rl::components::on_policy_runner::Buffer<RS> buffers[2];
        rl::components::on_policy_runner::Dataset<DS> datasets[2];
        BATCH environments[2];
        RNG rngs[2];
        Tensor<MASK_SPEC> mask, host_mask;
        Matrix<matrix::Specification<float, TI, 1, 2>> log_std;
        malloc(gpu, mask); malloc(cpu, host_mask); malloc(gpu, log_std);
        set_all(gpu, log_std, 0.125f);
        for(TI i = 0; i < 2; i++) {
            malloc(gpu, runners[i]); malloc(gpu, buffers[i]); malloc(gpu, datasets[i]); malloc(gpu, environments[i]); malloc(gpu, rngs[i]);
            if constexpr(CURAND) init(gpu, rngs[i], 19);
            else {
                RNG initial; malloc(cpu, initial); init(cpu, initial, 19); copy(cpu, gpu, initial, rngs[i]); free(cpu, initial);
            }
            init(gpu, environments[i]); init(gpu, runners[i], environments[i], rngs[i]);
            set_all(gpu, datasets[i].scalar_data, 0.0f);
            set_all(gpu, datasets[i].all_observations, 0.0f);
            set_all(gpu, datasets[i].all_observations_privileged, PRIV_T(0));
        }
        auto compare = [&]() {
            equal_container(cpu, gpu, environments[0].environments, environments[1].environments);
            equal_container(cpu, gpu, runners[0].states, runners[1].states);
            equal_container(cpu, gpu, runners[0].env_parameters, runners[1].env_parameters);
            equal_container(cpu, gpu, runners[0].reset, runners[1].reset);
            equal_container(cpu, gpu, runners[0].episode_step, runners[1].episode_step);
            equal_container(cpu, gpu, datasets[0].scalar_data, datasets[1].scalar_data);
            equal_container(cpu, gpu, datasets[0].all_observations, datasets[1].all_observations);
            equal_container(cpu, gpu, datasets[0].all_observations_privileged, datasets[1].all_observations_privileged);
            equal_container(cpu, gpu, rngs[0].states, rngs[1].states);
        };
        for(TI rollout = 0; rollout < 3; rollout++) {
            prologue(gpu, datasets[0], runners[0], environments[0], rngs[0]);
            prologue<GPU, DS, RS, BATCH, RNG>(gpu, datasets[1], runners[1], environments[1], rngs[1]);
            compare();
            for(TI t = 0; t < DS::STEPS_PER_ENV; t++) {
                SCOPED_TRACE(::testing::Message() << "rollout=" << rollout << " step=" << t);
                if(t == 1 || t == 3 || t == 5) {
                    for(TI i = 0; i < N; i++) set(cpu, host_mask, t == 1 ? false : t == 3 ? i % 2 == 0 : true, i);
                    ASSERT_EQ(cudaMemcpyAsync(mask._data, host_mask._data, MASK_SPEC::SIZE_BYTES, cudaMemcpyHostToDevice, gpu.stream), cudaSuccess);
                    reset(gpu, runners[0], environments[0], mask, rngs[0]);
                    reset<GPU, RS, BATCH, MASK_SPEC, RNG>(gpu, runners[1], environments[1], mask, rngs[1]);
                    compare();
                }
                if(t == 6) {
                    reset(gpu, runners[0], environments[0], rngs[0]);
                    reset<GPU, RS, BATCH, RNG>(gpu, runners[1], environments[1], rngs[1]);
                    compare();
                }
                for(TI i = 0; i < 2; i++) {
                    if(t % 2 == 0) sample_actions(gpu, datasets[i], log_std, buffers[i].actions, t, rngs[i]);
                    else set_all(gpu, buffers[i].actions, 0.25f);
                }
                epilogue(gpu, datasets[0], runners[0], buffers[0], environments[0], rngs[0], t);
                epilogue<GPU, DS, RS, BATCH, RNG>(gpu, datasets[1], runners[1], buffers[1], environments[1], rngs[1], t);
                compare();
                equal_container(cpu, gpu, buffers[0].next_states, buffers[1].next_states);
                equal_container(cpu, gpu, buffers[0].actions, buffers[1].actions);
                equal_container(cpu, gpu, buffers[0].rewards, buffers[1].rewards);
                equal_container(cpu, gpu, buffers[0].terminated, buffers[1].terminated);
            }
        }
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        for(TI i = 0; i < 2; i++) {
            free(gpu, runners[i]); free(gpu, buffers[i]); free(gpu, datasets[i]); free(gpu, environments[i]); free(gpu, rngs[i]);
        }
        free(gpu, mask); free(cpu, host_mask); free(gpu, log_std);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
        EXPECT_EQ(cudnnDestroy(gpu.cudnn_handle), CUDNN_STATUS_SUCCESS);
#endif
        EXPECT_EQ(cublasDestroy(gpu.handle), CUBLAS_STATUS_SUCCESS);
        EXPECT_EQ(cudaStreamDestroy(gpu.stream), cudaSuccess);
        metra::log("on_policy_runner/fused/mismatches", ::testing::Test::HasFailure() ? 1.0 : 0.0);
    }
}

TEST(RL_TOOLS_ON_POLICY_RUNNER_FUSED, SINGLE) { compare_phases<1, false, false>(); }
TEST(RL_TOOLS_ON_POLICY_RUNNER_FUSED, SYMMETRIC) {
    compare_phases<37, false, false, false, 0>();
    compare_phases<37, false, false, false, 1>();
    compare_phases<37, false, false, false, 3>();
}
TEST(RL_TOOLS_ON_POLICY_RUNNER_FUSED, SYMMETRIC_MIXED_STORAGE) { compare_phases<37, false, false, true>(); }
TEST(RL_TOOLS_ON_POLICY_RUNNER_FUSED, ASYMMETRIC_MIXED_STORAGE) { compare_phases<37, true, false>(); }
TEST(RL_TOOLS_ON_POLICY_RUNNER_FUSED, CURAND) { compare_phases<37, true, true>(); }
