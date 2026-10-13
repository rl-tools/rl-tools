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
#include <limits>

#ifndef HYPERDRONE_DYNAMICS_NUM_DRONES
#error "HYPERDRONE_DYNAMICS_NUM_DRONES must be defined"
#endif

namespace rlt = rl_tools;

namespace hyperdrone_dynamics_impl {
    using rlt::malloc;
    using rlt::free;
    using rlt::json;
    using rlt::from_json;
    using rlt::initial_state;
    using rlt::sample_initial_state;
    using DEVICE_CPU = rlt::devices::CPU<rlt::devices::DefaultCPUSpecification>;
    using T = float;
    using TI = DEVICE_CPU::index_t;

    constexpr TI NUM_DRONES = HYPERDRONE_DYNAMICS_NUM_DRONES;

    using namespace rlt::rl::environments::l2f;

    using FACTORY = parameters::DEFAULT_PARAMETERS_FACTORY<T, TI, 5>;

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
        using PARAMETERS = ParametersDisturbances<ParametersSpecification<T, TI, ParametersBase<typename FACTORY::PARAMETERS_SPEC>>>;
        static constexpr PARAMETERS PARAMETER_VALUES = {{FACTORY::dynamics, FACTORY::integration, FACTORY::mdp}, FACTORY::disturbances};
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

    __global__ void kernel_camera_bases(const STATE* states,
                                        const float* mount, float scale, float aspect, float* out){
        const TI drone = blockIdx.x * blockDim.x + threadIdx.x;
        if(drone < NUM_DRONES){
            camera_base_one(states[drone].position, states[drone].orientation, mount, scale, aspect, &out[drone * 12]);
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
        std::vector<ENGINE> engines, observe_engines;
        ENGINE master_engine;
        PARAMETERS nominal_parameters = STATIC_PARAMETERS::PARAMETER_VALUES;
        bool initialized = false;

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
        curandState *device_rng, *device_observe_rng;
        cudaStream_t cuda_stream;
        cudaEvent_t ordering_event;
        int device_ordinal;
#endif

        SimImpl(){
            envs.resize(NUM_DRONES);
            parameters.resize(NUM_DRONES);
            states.resize(NUM_DRONES);
            next_states.resize(NUM_DRONES);
            engines.resize(NUM_DRONES);
            observe_engines.resize(NUM_DRONES);
            malloc(device_cpu, master_engine);
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                rlt::malloc(device_cpu, envs[drone]);
                rlt::init(device_cpu, envs[drone]);
                malloc(device_cpu, engines[drone]);
                malloc(device_cpu, observe_engines[drone]);
            }
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            device_envs = device_alloc<ENVIRONMENT>(NUM_DRONES);
            device_parameters = device_alloc<PARAMETERS>(NUM_DRONES);
            device_states = device_alloc<STATE>(NUM_DRONES);
            device_next_states = device_alloc<STATE>(NUM_DRONES);
            device_actions = device_alloc<float>(NUM_DRONES * ACTION_DIM);
            device_mount = device_alloc<float>(12);
            device_rng = device_alloc<curandState>(NUM_DRONES);
            device_observe_rng = device_alloc<curandState>(NUM_DRONES);
            buffer_position = device_alloc<float>(NUM_DRONES * 3);
            buffer_orientation = device_alloc<float>(NUM_DRONES * 4);
            buffer_linear_velocity = device_alloc<float>(NUM_DRONES * 3);
            buffer_angular_velocity = device_alloc<float>(NUM_DRONES * 3);
            buffer_rpm = device_alloc<float>(NUM_DRONES * ACTION_DIM);
            buffer_observations = device_alloc<float>(NUM_DRONES * OBSERVATION_DIM);
            buffer_camera_bases = device_alloc<float>(NUM_DRONES * 12);
            cudaStreamCreate(&cuda_stream);
            cudaEventCreateWithFlags(&ordering_event, cudaEventDisableTiming);
            cudaPointerAttributes attributes;
            cudaPointerGetAttributes(&attributes, buffer_position);
            device_ordinal = attributes.device;
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
        }

        ~SimImpl() override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            cudaStreamSynchronize(cuda_stream);
            cudaEventDestroy(ordering_event);
            cudaStreamDestroy(cuda_stream);
            cudaFree(device_envs); cudaFree(device_parameters);
            cudaFree(device_states); cudaFree(device_next_states);
            cudaFree(device_actions); cudaFree(device_mount); cudaFree(device_rng); cudaFree(device_observe_rng);
            cudaFree(buffer_position); cudaFree(buffer_orientation);
            cudaFree(buffer_linear_velocity); cudaFree(buffer_angular_velocity); cudaFree(buffer_rpm);
            cudaFree(buffer_observations); cudaFree(buffer_camera_bases);
#endif
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                free(device_cpu, engines[drone]);
                free(device_cpu, observe_engines[drone]);
                free(device_cpu, envs[drone]);
            }
            free(device_cpu, master_engine);
        }

        hyperdrone::dynamics::Config config() const override {
            hyperdrone::dynamics::Config c;
            c.num_drones = NUM_DRONES;
            c.action_dim = ACTION_DIM;
            c.observation_dim = OBSERVATION_DIM;
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

        int buffer_device_id() const override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            return device_ordinal;
#else
            return 0;
#endif
        }

        void order_consumer(unsigned long long consumer_stream) override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            auto consumer = reinterpret_cast<cudaStream_t>(consumer_stream);
            if(consumer != cuda_stream){
                cudaEventRecord(ordering_event, cuda_stream);
                cudaStreamWaitEvent(consumer, ordering_event, 0);
            }
#else
            (void)consumer_stream;
#endif
        }

        std::string parameters_json() override {
            return "{\"dynamics\":" + json(device_cpu, envs[0], nominal_parameters) + "}";
        }

        void init(unsigned long long seed, const std::string& input) override {
            if(initialized){
                throw std::invalid_argument("hyperdrone: a Sim can only be initialized once");
            }
            PARAMETERS parsed = nominal_parameters;
            from_json(device_cpu, envs[0], nlohmann::json::parse(input).at("dynamics"), parsed);
            if(!(parsed.integration.dt > 0)){
                throw std::invalid_argument("hyperdrone: integration.dt must be positive");
            }
            nominal_parameters = parsed;
            rl_tools::init(device_cpu, master_engine, seed);
            const auto step_seed = rlt::random::uniform_int_distribution(device_cpu.random, uint32_t(0), std::numeric_limits<uint32_t>::max() - uint32_t(NUM_DRONES), master_engine);
            const auto observe_seed = rlt::random::uniform_int_distribution(device_cpu.random, uint32_t(0), std::numeric_limits<uint32_t>::max() - uint32_t(NUM_DRONES), master_engine);
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                rl_tools::init(device_cpu, engines[drone], step_seed + drone);
                rl_tools::init(device_cpu, observe_engines[drone], observe_seed + drone);
                parameters[drone] = nominal_parameters;
                initial_state(device_cpu, envs[drone], parameters[drone], states[drone]);
            }
            upload_parameters();
            upload_states_and_publish();
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            kernel_init_rng<<<GRID_DIM, BLOCK_DIM, 0, cuda_stream>>>(device_rng, step_seed);
            kernel_init_rng<<<GRID_DIM, BLOCK_DIM, 0, cuda_stream>>>(device_observe_rng, observe_seed);
            cudaStreamSynchronize(cuda_stream);
#endif
            initialized = true;
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

        void reset(const bool* mask, bool sample_states) override {
            download_states();
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                if(mask[drone]){
                    parameters[drone] = nominal_parameters;
                    if(sample_states){
                        sample_initial_state(device_cpu, envs[drone], parameters[drone], states[drone], master_engine);
                    }
                    else {
                        initial_state(device_cpu, envs[drone], parameters[drone], states[drone]);
                    }
                }
            }
            upload_parameters();
            upload_states_and_publish();
        }

        void step(const float* actions, int actions_device_type, unsigned long long producer_stream) override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            if(actions_device_type == 2){
                cudaStream_t producer = (cudaStream_t)producer_stream;
                if(producer != cuda_stream){
                    cudaEventRecord(ordering_event, producer);
                    cudaStreamWaitEvent(cuda_stream, ordering_event, 0);
                }
                cudaMemcpyAsync(device_actions, actions, NUM_DRONES * ACTION_DIM * sizeof(float), cudaMemcpyDeviceToDevice, cuda_stream);
                order_consumer(producer_stream);
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

        void read_rotor_rpm(float* out_host) override {
            download_states();
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                for(TI rotor_i = 0; rotor_i < ACTION_DIM; rotor_i++){
                    out_host[drone * ACTION_DIM + rotor_i] = rotor_rpm_from_command(device_cpu, parameters[drone], states[drone].rpm[rotor_i]);
                }
            }
        }

        void update_observations() override {
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            kernel_observe<<<GRID_DIM, BLOCK_DIM, 0, cuda_stream>>>(device_cuda, device_envs, device_parameters,
                                                                    device_states, device_observe_rng, buffer_observations);
#else
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBSERVATION_DIM, false>> observation;
                rlt::observe(device_cpu, envs[drone], parameters[drone], states[drone], typename ENVIRONMENT::Observation{}, observation, observe_engines[drone]);
                for(TI i = 0; i < OBSERVATION_DIM; i++){
                    buffer_observations[drone * OBSERVATION_DIM + i] = rlt::get(observation, 0, i);
                }
            }
#endif
        }

        const float* observation_buffer() const override {
            return buffer_observations;
        }

        void update_camera_bases(const float mount[12], float fov_degrees, float aspect) override {
            // same double-precision conversion as rendering::raytracing::degrees_to_radians
            const float fov = (float)((double)fov_degrees * 0.017453292519943295);
            const float scale = 2.0f * std::tan(fov / 2.0f);
#if defined(HYPERDRONE_DYNAMICS_CUDA)
            cudaMemcpyAsync(device_mount, mount, 12 * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
            kernel_camera_bases<<<GRID_DIM, BLOCK_DIM, 0, cuda_stream>>>(device_states,
                                                                         device_mount, scale, aspect, buffer_camera_bases);
#else
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                camera_base_one(states[drone].position, states[drone].orientation, mount, scale, aspect,
                                &buffer_camera_bases[drone * 12]);
            }
#endif
        }

        const float* camera_bases_buffer() const override {
            return buffer_camera_bases;
        }

        std::string instance_parameters_json() override {
            nlohmann::json result = nlohmann::json::array();
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                result.push_back({{"dynamics", nlohmann::json::parse(json(device_cpu, envs[drone], parameters[drone]))}});
            }
            return result.dump();
        }

        void set_instance_parameters(const bool* mask, const std::string& input) override {
            const auto entries = nlohmann::json::parse(input);
            auto updated = parameters;
            size_t entry = 0;
            for(TI drone = 0; drone < NUM_DRONES; drone++){
                if(mask[drone]){
                    const auto& requested = entries.at(entry++).at("dynamics");
                    auto current = nlohmann::json::parse(json(device_cpu, envs[drone], parameters[drone]));
                    for(const auto& item : current.items()){
                        if(item.key() != "dynamics" && requested.at(item.key()) != item.value()){
                            throw std::invalid_argument("hyperdrone: only dynamics.dynamics varies per instance; set shared values through parameters=");
                        }
                    }
                    from_json(device_cpu, envs[drone], requested, updated[drone]);
                }
            }
            if(entry != entries.size()){
                throw std::invalid_argument("hyperdrone: set_instance_parameters takes one entry per selected drone");
            }
            synchronize();
            parameters.swap(updated);
            upload_parameters();
        }

    };

    static char config_string_buffer[64];
    const char* build_config_string(){
        std::snprintf(config_string_buffer, sizeof(config_string_buffer), "n=%d", (int)NUM_DRONES);
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
