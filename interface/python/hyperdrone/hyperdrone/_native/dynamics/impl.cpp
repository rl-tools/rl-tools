#if defined(HYPERDRONE_DYNAMICS_CUDA)
#define RL_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION
#define RL_TOOLS_FUNCTION_PLACEMENT __host__ __device__
#include <rl_tools/operations/cpu.h>
#include <rl_tools/operations/cuda.h>
#define HYPERDRONE_HD __host__ __device__
#else
#include <rl_tools/operations/cpu.h>
#define HYPERDRONE_HD
#endif

#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/parameters/default.h>

#include "iface.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef HYPERDRONE_DYNAMICS_NUM_DRONES
#error "HYPERDRONE_DYNAMICS_NUM_DRONES must be defined"
#endif
#ifndef HYPERDRONE_DYNAMICS_DOMAIN_RANDOMIZATION
#define HYPERDRONE_DYNAMICS_DOMAIN_RANDOMIZATION 0
#endif

namespace rlt = rl_tools;

namespace hyperdrone_dynamics_impl {
    using DEVICE_CPU = rlt::devices::CPU<rlt::devices::DefaultCPUSpecification>;
    using T = float;
    using TI = DEVICE_CPU::index_t;

    constexpr TI NUM_DRONES = HYPERDRONE_DYNAMICS_NUM_DRONES;

    using namespace rlt::rl::environments::l2f;

    using DR_OPTIONS = parameters::DEFAULT_DOMAIN_RANDOMIZATION_OPTIONS<HYPERDRONE_DYNAMICS_DOMAIN_RANDOMIZATION != 0>;
    using FACTORY = parameters::DEFAULT_PARAMETERS_FACTORY<T, TI, 5, DR_OPTIONS>;

    struct STATIC_PARAMETERS{
        static constexpr auto ACTION_INTERFACE = parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 16;
        static constexpr TI EPISODE_STEP_LIMIT = FACTORY::EPISODE_STEP_LIMIT_OUTER;
        static constexpr TI CLOSED_FORM = false;
        static constexpr TI ANGULAR_VELOCITY_DELAY = 0;
        static constexpr TI ANGULAR_VELOCITY_HISTORY = ANGULAR_VELOCITY_DELAY;
        using STATE_BASE = StateBase<StateSpecification<T, TI>>;
        using STATE_TYPE = StateRotors<StateRotorsSpecification<T, TI, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>;
        using OBSERVATION_TYPE =
            observation::Position<observation::PositionSpecification<T, TI,
            observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
            observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
            observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = typename FACTORY::PARAMETERS_TYPE;
        static constexpr PARAMETERS PARAMETER_VALUES = FACTORY::nominal_parameters;
        static constexpr T STATE_LIMIT_POSITION_X = 100000;
        static constexpr T STATE_LIMIT_POSITION_Y = 100000;
        static constexpr T STATE_LIMIT_POSITION_Z = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
    };

    using ENVIRONMENT_SPEC = Specification<T, TI, STATIC_PARAMETERS>;
    using ENVIRONMENT = rlt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    using STATE = typename ENVIRONMENT::State;
    using PARAMETERS = typename ENVIRONMENT::Parameters;
    constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
    constexpr TI OBSERVATION_DIM = ENVIRONMENT::Observation::DIM;

    using DYNAMICS = typename FACTORY::PARAMETERS_TYPE::Dynamics;
    using PARAMETERS_SPEC = typename FACTORY::PARAMETERS_SPEC;

    bool model_by_name(const std::string& name, DYNAMICS& out){
        namespace pd = parameters::dynamics;
        if(name == "crazyflie"){ out = pd::registry<pd::REGISTRY::crazyflie, PARAMETERS_SPEC>; return true; }
        if(name == "crazyflie_openmv"){ out = pd::registry<pd::REGISTRY::crazyflie_openmv, PARAMETERS_SPEC>; return true; }
        if(name == "mrs"){ out = pd::registry<pd::REGISTRY::mrs, PARAMETERS_SPEC>; return true; }
        if(name == "x500"){ out = pd::registry<pd::REGISTRY::x500, PARAMETERS_SPEC>; return true; }
        if(name == "x500_real"){ out = pd::registry<pd::REGISTRY::x500_real, PARAMETERS_SPEC>; return true; }
        if(name == "x500_sim"){ out = pd::registry<pd::REGISTRY::x500_sim, PARAMETERS_SPEC>; return true; }
        if(name == "arpl"){ out = pd::registry<pd::REGISTRY::arpl, PARAMETERS_SPEC>; return true; }
        if(name == "fs_base"){ out = pd::registry<pd::REGISTRY::fs_base, PARAMETERS_SPEC>; return true; }
        if(name == "flightmare"){ out = pd::registry<pd::REGISTRY::flightmare, PARAMETERS_SPEC>; return true; }
        if(name == "soft"){ out = pd::registry<pd::REGISTRY::soft, PARAMETERS_SPEC>; return true; }
        if(name == "soft_rigid"){ out = pd::registry<pd::REGISTRY::soft_rigid, PARAMETERS_SPEC>; return true; }
        return false;
    }

    HYPERDRONE_HD inline void publish_one(const STATE& state, float* position, float* orientation, float* linear_velocity, float* angular_velocity, float* rpm){
        for(int i = 0; i < 3; i++){
            position[i] = state.position[i];
            linear_velocity[i] = state.linear_velocity[i];
            angular_velocity[i] = state.angular_velocity[i];
        }
        for(int i = 0; i < 4; i++){
            orientation[i] = state.orientation[i];
        }
        for(TI i = 0; i < ACTION_DIM; i++){
            rpm[i] = state.rpm[i];
        }
    }

    // packed renderer camera basis from drone pose + body-frame mount transform (3x4
    // row-major [R|t]; camera axes in FLU: +X forward, +Y left, +Z up)
    HYPERDRONE_HD inline void camera_base_one(const float* position, const float* orientation_wxyz,
                                              const float* mount, float scale, float aspect, float* out){
        const float w = orientation_wxyz[0], x = orientation_wxyz[1], y = orientation_wxyz[2], z = orientation_wxyz[3];
        float R[3][3] = {
            {1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)},
            {    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)},
            {    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)}
        };
        float camera_axes[3][3]; // columns: forward, left, up (world frame)
        float camera_position[3];
        for(int row = 0; row < 3; row++){
            for(int column = 0; column < 3; column++){
                float value = 0;
                for(int inner = 0; inner < 3; inner++){
                    value += R[row][inner] * mount[inner * 4 + column];
                }
                camera_axes[row][column] = value;
            }
            float offset = 0;
            for(int inner = 0; inner < 3; inner++){
                offset += R[row][inner] * mount[inner * 4 + 3];
            }
            camera_position[row] = position[row] + offset;
        }
        float forward[3], right[3], up[3];
        for(int row = 0; row < 3; row++){
            forward[row] = camera_axes[row][0];
            right[row] = -camera_axes[row][1];
            up[row] = camera_axes[row][2];
        }
        float dir_du[3], dir_dv[3], dir_00[3];
        for(int row = 0; row < 3; row++){
            dir_du[row] = right[row] * scale;
            dir_dv[row] = up[row] * scale / aspect;
            dir_00[row] = forward[row] - 0.5f * dir_du[row] + 0.5f * dir_dv[row];
        }
        for(int row = 0; row < 3; row++){
            out[0 + row] = camera_position[row];
            out[3 + row] = dir_00[row];
            out[6 + row] = dir_du[row];
            out[9 + row] = -dir_dv[row];
        }
    }

#if defined(HYPERDRONE_DYNAMICS_CUDA)
    using DEVICE_CUDA = rlt::devices::CUDA<rlt::devices::DefaultCUDASpecification>;

    __global__ void kernel_init_rng(curandState* rng_states, unsigned long long seed){
        const TI drone = blockIdx.x * blockDim.x + threadIdx.x;
        if(drone < NUM_DRONES){
            curand_init(seed, drone, 0, &rng_states[drone]);
        }
    }

    __global__ void kernel_publish(const STATE* states, float* position, float* orientation, float* linear_velocity, float* angular_velocity, float* rpm){
        const TI drone = blockIdx.x * blockDim.x + threadIdx.x;
        if(drone < NUM_DRONES){
            publish_one(states[drone], &position[drone * 3], &orientation[drone * 4],
                        &linear_velocity[drone * 3], &angular_velocity[drone * 3], &rpm[drone * ACTION_DIM]);
        }
    }

    __global__ void kernel_step(DEVICE_CUDA device, const ENVIRONMENT* envs, PARAMETERS* parameters,
                                const STATE* states, STATE* next_states, const float* actions,
                                curandState* rng_states,
                                float* position, float* orientation, float* linear_velocity, float* angular_velocity, float* rpm){
        const TI drone = blockIdx.x * blockDim.x + threadIdx.x;
        if(drone >= NUM_DRONES){
            return;
        }
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACTION_DIM, false>> action;
        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            rlt::set(action, 0, action_i, actions[drone * ACTION_DIM + action_i]);
        }
        STATE state = states[drone];
        STATE next_state;
        rlt::step(device, envs[drone], parameters[drone], state, action, next_state, rng_states[drone]);
        next_states[drone] = next_state;
        publish_one(next_state, &position[drone * 3], &orientation[drone * 4],
                    &linear_velocity[drone * 3], &angular_velocity[drone * 3], &rpm[drone * ACTION_DIM]);
    }

    __global__ void kernel_observe(DEVICE_CUDA device, const ENVIRONMENT* envs, PARAMETERS* parameters,
                                   const STATE* states, curandState* rng_states, float* observations){
        const TI drone = blockIdx.x * blockDim.x + threadIdx.x;
        if(drone >= NUM_DRONES){
            return;
        }
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBSERVATION_DIM, false>> observation;
        rlt::observe(device, envs[drone], parameters[drone], states[drone], typename ENVIRONMENT::Observation{}, observation, rng_states[drone]);
        for(TI i = 0; i < OBSERVATION_DIM; i++){
            observations[drone * OBSERVATION_DIM + i] = rlt::get(observation, 0, i);
        }
    }

    __global__ void kernel_camera_bases(const float* position, const float* orientation,
                                        const float* mount, float scale, float aspect, float* out){
        const TI drone = blockIdx.x * blockDim.x + threadIdx.x;
        if(drone < NUM_DRONES){
            camera_base_one(&position[drone * 3], &orientation[drone * 4], mount, scale, aspect, &out[drone * 12]);
        }
    }

    template <typename VALUE>
    VALUE* device_alloc(size_t count){
        VALUE* pointer = nullptr;
        if(cudaMalloc((void**)&pointer, count * sizeof(VALUE)) != cudaSuccess){
            throw std::runtime_error("hyperdrone: cudaMalloc failed in dynamics");
        }
        return pointer;
    }
#endif

    constexpr TI BLOCK_DIM = 256;
    constexpr TI GRID_DIM = (NUM_DRONES + BLOCK_DIM - 1) / BLOCK_DIM;

    struct SimImpl final : hyperdrone::dynamics::Sim {
        using ENGINE = DEVICE_CPU::SPEC::RANDOM::ENGINE<>;

        DEVICE_CPU device_cpu;
        std::vector<ENVIRONMENT> envs;
        std::vector<PARAMETERS> parameters;
        std::vector<STATE> states, next_states;
        std::vector<ENGINE> engines;
        DYNAMICS model_dynamics;
        std::string model_name = "crazyflie";
        float integration_dt;

        // SoA buffers (device-resident on the CUDA variant)
        float *buffer_position, *buffer_orientation, *buffer_linear_velocity, *buffer_angular_velocity, *buffer_rpm;
        float *buffer_observations, *buffer_camera_bases;
        std::vector<float> host_position, host_orientation, host_linear_velocity, host_angular_velocity, host_rpm;
        std::vector<float> host_observations, host_camera_bases;

#if defined(HYPERDRONE_DYNAMICS_CUDA)
        DEVICE_CUDA device_cuda;
        ENVIRONMENT* device_envs;
        PARAMETERS* device_parameters;
        STATE *device_states, *device_next_states;
        float* device_actions;
        float* device_mount;
        curandState* device_rng;
        cudaStream_t cuda_stream;
#endif

        SimImpl(){
            envs.resize(NUM_DRONES);
            parameters.resize(NUM_DRONES);
            states.resize(NUM_DRONES);
            next_states.resize(NUM_DRONES);
            engines.resize(NUM_DRONES);
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                rlt::malloc(device_cpu, envs[drone]);
                rlt::init(device_cpu, envs[drone]);
                rlt::malloc(device_cpu, engines[drone]);
            }
            model_by_name(model_name, model_dynamics);
            integration_dt = FACTORY::nominal_parameters.integration.dt;
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            device_envs = device_alloc<ENVIRONMENT>(NUM_DRONES);
            device_parameters = device_alloc<PARAMETERS>(NUM_DRONES);
            device_states = device_alloc<STATE>(NUM_DRONES);
            device_next_states = device_alloc<STATE>(NUM_DRONES);
            device_actions = device_alloc<float>(NUM_DRONES * ACTION_DIM);
            device_mount = device_alloc<float>(12);
            device_rng = device_alloc<curandState>(NUM_DRONES);
            buffer_position = device_alloc<float>(NUM_DRONES * 3);
            buffer_orientation = device_alloc<float>(NUM_DRONES * 4);
            buffer_linear_velocity = device_alloc<float>(NUM_DRONES * 3);
            buffer_angular_velocity = device_alloc<float>(NUM_DRONES * 3);
            buffer_rpm = device_alloc<float>(NUM_DRONES * ACTION_DIM);
            buffer_observations = device_alloc<float>(NUM_DRONES * OBSERVATION_DIM);
            buffer_camera_bases = device_alloc<float>(NUM_DRONES * 12);
            cudaStreamCreate(&cuda_stream);
            cudaMemcpy(device_envs, envs.data(), NUM_DRONES * sizeof(ENVIRONMENT), cudaMemcpyHostToDevice);
#else
            host_position.resize(NUM_DRONES * 3);
            host_orientation.resize(NUM_DRONES * 4);
            host_linear_velocity.resize(NUM_DRONES * 3);
            host_angular_velocity.resize(NUM_DRONES * 3);
            host_rpm.resize(NUM_DRONES * ACTION_DIM);
            host_observations.resize(NUM_DRONES * OBSERVATION_DIM);
            host_camera_bases.resize(NUM_DRONES * 12);
            buffer_position = host_position.data();
            buffer_orientation = host_orientation.data();
            buffer_linear_velocity = host_linear_velocity.data();
            buffer_angular_velocity = host_angular_velocity.data();
            buffer_rpm = host_rpm.data();
            buffer_observations = host_observations.data();
            buffer_camera_bases = host_camera_bases.data();
#endif
            reset(0, false, false);
        }

        ~SimImpl() override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            cudaStreamDestroy(cuda_stream);
            cudaFree(device_envs); cudaFree(device_parameters);
            cudaFree(device_states); cudaFree(device_next_states);
            cudaFree(device_actions); cudaFree(device_mount); cudaFree(device_rng);
            cudaFree(buffer_position); cudaFree(buffer_orientation);
            cudaFree(buffer_linear_velocity); cudaFree(buffer_angular_velocity); cudaFree(buffer_rpm);
            cudaFree(buffer_observations); cudaFree(buffer_camera_bases);
#endif
        }

        hyperdrone::dynamics::Config config() const override {
            hyperdrone::dynamics::Config c;
            c.num_drones = NUM_DRONES;
            c.action_dim = ACTION_DIM;
            c.observation_dim = OBSERVATION_DIM;
            c.domain_randomization = HYPERDRONE_DYNAMICS_DOMAIN_RANDOMIZATION != 0;
            return c;
        }

        const char* device_name() const override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            return "cuda";
#else
            return "cpu";
#endif
        }

        int buffer_device_type() const override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            return 2;
#else
            return 1;
#endif
        }

        unsigned long long stream() const override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            return (unsigned long long)cuda_stream;
#else
            return 0;
#endif
        }

        void synchronize() override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            cudaStreamSynchronize(cuda_stream);
#endif
        }

        bool set_model(const char* name) override {
            DYNAMICS dynamics_values;
            if(!model_by_name(name, dynamics_values)){
                return false;
            }
            model_dynamics = dynamics_values;
            model_name = name;
            apply_parameter_overrides();
            upload_parameters();
            return true;
        }

        void set_dt(float dt) override {
            integration_dt = dt;
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                parameters[drone].integration.dt = dt;
            }
            upload_parameters();
        }

        float dt() const override {
            return integration_dt;
        }

        void apply_parameter_overrides(){
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                parameters[drone].dynamics = model_dynamics;
                parameters[drone].integration.dt = integration_dt;
            }
        }

        void upload_parameters(){
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            cudaMemcpy(device_parameters, parameters.data(), NUM_DRONES * sizeof(PARAMETERS), cudaMemcpyHostToDevice);
#endif
        }

        void upload_states_and_publish(){
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            cudaMemcpy(device_states, states.data(), NUM_DRONES * sizeof(STATE), cudaMemcpyHostToDevice);
            kernel_publish<<<GRID_DIM, BLOCK_DIM, 0, cuda_stream>>>(device_states, buffer_position, buffer_orientation,
                                                                    buffer_linear_velocity, buffer_angular_velocity, buffer_rpm);
            cudaStreamSynchronize(cuda_stream);
#else
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                publish_one(states[drone], &buffer_position[drone * 3], &buffer_orientation[drone * 4],
                            &buffer_linear_velocity[drone * 3], &buffer_angular_velocity[drone * 3], &buffer_rpm[drone * ACTION_DIM]);
            }
#endif
        }

        void download_states(){
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            cudaStreamSynchronize(cuda_stream);
            cudaMemcpy(states.data(), device_states, NUM_DRONES * sizeof(STATE), cudaMemcpyDeviceToHost);
#endif
        }

        // all sampling runs on the host with the rl_tools CPU engine so cpu and cuda
        // variants reset to identical parameters and states
        void reset(unsigned long long seed, bool sample_parameters, bool sample_states) override {
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                rlt::init(device_cpu, engines[drone], (TI)(seed + drone));
                rlt::initial_parameters(device_cpu, envs[drone], parameters[drone]);
                if(sample_parameters){
                    rlt::sample_initial_parameters(device_cpu, envs[drone], parameters[drone], engines[drone]);
                }
            }
            apply_parameter_overrides();
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                if(sample_states){
                    rlt::sample_initial_state(device_cpu, envs[drone], parameters[drone], states[drone], engines[drone]);
                }
                else {
                    rlt::initial_state(device_cpu, envs[drone], parameters[drone], states[drone]);
                }
            }
            upload_parameters();
            upload_states_and_publish();
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            kernel_init_rng<<<GRID_DIM, BLOCK_DIM, 0, cuda_stream>>>(device_rng, seed);
            cudaStreamSynchronize(cuda_stream);
#endif
        }

        void step(const float* actions, int actions_device_type, unsigned long long producer_stream) override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            if(actions_device_type == 2){
                cudaStream_t producer = (cudaStream_t)producer_stream;
                if(producer != nullptr && producer != cuda_stream){
                    cudaEvent_t actions_ready;
                    cudaEventCreateWithFlags(&actions_ready, cudaEventDisableTiming);
                    cudaEventRecord(actions_ready, producer);
                    cudaStreamWaitEvent(cuda_stream, actions_ready, 0);
                    cudaEventDestroy(actions_ready);
                }
                cudaMemcpyAsync(device_actions, actions, NUM_DRONES * ACTION_DIM * sizeof(float), cudaMemcpyDeviceToDevice, cuda_stream);
            }
            else {
                cudaMemcpyAsync(device_actions, actions, NUM_DRONES * ACTION_DIM * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
            }
            kernel_step<<<GRID_DIM, BLOCK_DIM, 0, cuda_stream>>>(device_cuda, device_envs, device_parameters,
                                                                 device_states, device_next_states, device_actions,
                                                                 device_rng,
                                                                 buffer_position, buffer_orientation,
                                                                 buffer_linear_velocity, buffer_angular_velocity, buffer_rpm);
            STATE* swap = device_states;
            device_states = device_next_states;
            device_next_states = swap;
#else
            if(actions_device_type != 1){
                throw std::runtime_error("hyperdrone: the cpu dynamics variant accepts host actions only");
            }
            (void)producer_stream;
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACTION_DIM, false>> action;
                for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
                    rlt::set(action, 0, action_i, actions[drone * ACTION_DIM + action_i]);
                }
                rlt::step(device_cpu, envs[drone], parameters[drone], states[drone], action, next_states[drone], engines[drone]);
                states[drone] = next_states[drone];
                publish_one(states[drone], &buffer_position[drone * 3], &buffer_orientation[drone * 4],
                            &buffer_linear_velocity[drone * 3], &buffer_angular_velocity[drone * 3], &buffer_rpm[drone * ACTION_DIM]);
            }
#endif
        }

        const float* state_buffer(int component) const override {
            switch((hyperdrone::dynamics::StateComponent)component){
                case hyperdrone::dynamics::StateComponent::POSITION: return buffer_position;
                case hyperdrone::dynamics::StateComponent::ORIENTATION: return buffer_orientation;
                case hyperdrone::dynamics::StateComponent::LINEAR_VELOCITY: return buffer_linear_velocity;
                case hyperdrone::dynamics::StateComponent::ANGULAR_VELOCITY: return buffer_angular_velocity;
                case hyperdrone::dynamics::StateComponent::RPM: return buffer_rpm;
            }
            throw std::runtime_error("hyperdrone: unknown state component");
        }

        void set_state_component(int component, const float* values_host) override {
            download_states();
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                STATE& state = states[drone];
                switch((hyperdrone::dynamics::StateComponent)component){
                    case hyperdrone::dynamics::StateComponent::POSITION:
                        for(int i = 0; i < 3; i++){ state.position[i] = values_host[drone * 3 + i]; }
                        break;
                    case hyperdrone::dynamics::StateComponent::ORIENTATION:
                        for(int i = 0; i < 4; i++){ state.orientation[i] = values_host[drone * 4 + i]; }
                        break;
                    case hyperdrone::dynamics::StateComponent::LINEAR_VELOCITY:
                        for(int i = 0; i < 3; i++){ state.linear_velocity[i] = values_host[drone * 3 + i]; }
                        break;
                    case hyperdrone::dynamics::StateComponent::ANGULAR_VELOCITY:
                        for(int i = 0; i < 3; i++){ state.angular_velocity[i] = values_host[drone * 3 + i]; }
                        break;
                    case hyperdrone::dynamics::StateComponent::RPM:
                        for(TI i = 0; i < ACTION_DIM; i++){ state.rpm[i] = values_host[drone * ACTION_DIM + i]; }
                        break;
                    default:
                        throw std::runtime_error("hyperdrone: unknown state component");
                }
            }
            upload_states_and_publish();
        }

        void update_observations() override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            kernel_observe<<<GRID_DIM, BLOCK_DIM, 0, cuda_stream>>>(device_cuda, device_envs, device_parameters,
                                                                    device_states, device_rng, buffer_observations);
#else
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBSERVATION_DIM, false>> observation;
                rlt::observe(device_cpu, envs[drone], parameters[drone], states[drone], typename ENVIRONMENT::Observation{}, observation, engines[drone]);
                for(TI i = 0; i < OBSERVATION_DIM; i++){
                    buffer_observations[drone * OBSERVATION_DIM + i] = rlt::get(observation, 0, i);
                }
            }
#endif
        }

        const float* observation_buffer() const override {
            return buffer_observations;
        }

        void update_camera_bases(const float mount[12], float fov, float aspect) override {
            const float scale = 2.0f * std::tan(fov / 2.0f);
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            cudaMemcpyAsync(device_mount, mount, 12 * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
            kernel_camera_bases<<<GRID_DIM, BLOCK_DIM, 0, cuda_stream>>>(buffer_position, buffer_orientation,
                                                                         device_mount, scale, aspect, buffer_camera_bases);
#else
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                camera_base_one(&buffer_position[drone * 3], &buffer_orientation[drone * 4], mount, scale, aspect,
                                &buffer_camera_bases[drone * 12]);
            }
#endif
        }

        const float* camera_bases_buffer() const override {
            return buffer_camera_bases;
        }

        template <typename ACCESSOR>
        bool parameter_access(const char* name, ACCESSOR&& accessor){
            const std::string key(name);
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                DYNAMICS& dynamics_values = parameters[drone].dynamics;
                if(key == "mass"){ accessor(drone, dynamics_values.mass); }
                else if(key == "hovering_throttle_relative"){ accessor(drone, dynamics_values.hovering_throttle_relative); }
                else if(key == "dt"){ accessor(drone, parameters[drone].integration.dt); }
                else { return false; }
            }
            return true;
        }

        bool read_parameter(const char* name, float* dst) override {
            return parameter_access(name, [dst](TI drone, T& value){ dst[drone] = value; });
        }

        bool write_parameter(const char* name, const float* values) override {
            const bool known = parameter_access(name, [values](TI drone, T& value){ value = values[drone]; });
            if(known){
                upload_parameters();
            }
            return known;
        }

    };

    static char config_string_buffer[64];
    const char* build_config_string(){
        std::snprintf(config_string_buffer, sizeof(config_string_buffer), "n=%d;dr=%d",
                      (int)NUM_DRONES, (int)(HYPERDRONE_DYNAMICS_DOMAIN_RANDOMIZATION != 0));
        return config_string_buffer;
    }
}

extern "C" {
    hyperdrone::dynamics::Sim* hyperdrone_dynamics_create(){
        return new hyperdrone_dynamics_impl::SimImpl();
    }
    void hyperdrone_dynamics_destroy(hyperdrone::dynamics::Sim* sim){
        delete sim;
    }
    const char* hyperdrone_dynamics_config_string(){
        return hyperdrone_dynamics_impl::build_config_string();
    }
    int hyperdrone_dynamics_iface_version(){
        return HYPERDRONE_DYNAMICS_IFACE_VERSION;
    }
}
