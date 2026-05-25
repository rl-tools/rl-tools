#include <rl_tools/operations/cuda.h>

#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rendering/raytracing/types.h>

#include "simulator_matrix_physics.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace rlt = rl_tools;

namespace rl_tools::rendering::raytracing::benchmark {
namespace {

using T = float;
using TI = int;

namespace l2f = rlt::rl::environments::l2f;
namespace obs = l2f::observation;

using DEVICE_GPU = rlt::devices::CUDA<rlt::devices::DefaultCUDASpecification>;
using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;

static constexpr TI SIMULATION_FREQUENCY = 100;
static constexpr TI PHYSICS_EPISODE_STEP_LIMIT = 500;
static constexpr TI ACTION_DIM = 4;
static constexpr TI NUM_ROTORS = 4;

using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, NUM_ROTORS, PHYSICS_EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
using PARAMETERS_BASE = l2f::ParametersBase<PARAMETERS_SPEC>;
using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, PARAMETERS_BASE>>;

static constexpr auto MODEL = l2f::parameters::dynamics::REGISTRY::crazyflie;
static constexpr REWARD_FUNCTION reward_function = {
    false, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    {0.00, 0.00, 0.00, 0.00}, {0.00, 0.00, 0.00, 0.00}, 0.00
};
static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = {
    0.0, 0.0, 0.0, 0.0, 0.0, true, -1, +1
};
static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
    false, 100000, 0, 100000, 100000, 100000, 100000
};
static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
static constexpr typename PARAMETERS_TYPE::Integration integration = {
    static_cast<T>(1) / static_cast<T>(SIMULATION_FREQUENCY)
};
static constexpr typename PARAMETERS_TYPE::MDP mdp = {
    init,
    reward_function,
    {},
    {},
    termination
};
static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
    {0, 0},
    {0, 0}
};
static constexpr PARAMETERS_TYPE nominal_parameters = {
    {
        dynamics,
        integration,
        mdp
    },
    disturbances
};

struct StaticParameters {
    static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI CLOSED_FORM = false;
    static constexpr TI EPISODE_STEP_LIMIT = PHYSICS_EPISODE_STEP_LIMIT;
    using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_TYPE = l2f::StateRotors<l2f::StateRotorsSpecification<T, TI, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, STATE_BASE>>>>;
    using OBSERVATION_TYPE = obs::NONE<TI>;
    using OBSERVATION_TYPE_PRIVILEGED = obs::NONE<TI>;
    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
    using PARAMETERS = PARAMETERS_TYPE;
    static constexpr PARAMETERS PARAMETER_VALUES = nominal_parameters;
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

using ENVIRONMENT_SPEC = l2f::Specification<T, TI, StaticParameters>;
using ENVIRONMENT = rlt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
using STATE = typename ENVIRONMENT::State;
using PARAMETERS = typename ENVIRONMENT::Parameters;
using CAMERA_DATA = rlt::rendering::raytracing::CameraData<T>;
static_assert(StaticParameters::EPISODE_STEP_LIMIT == PHYSICS_EPISODE_STEP_LIMIT);
static_assert(!StaticParameters::PRIVILEGED_OBSERVATION_NOISE);

struct PhysicsSimulationImpl {
    int num_envs = 0;
    T fov = 0;
    T aspect = 1;
    std::uint32_t seed = 0;
    ENVIRONMENT* envs = nullptr;
    PARAMETERS* parameters = nullptr;
    STATE* states = nullptr;
    STATE* initial_states = nullptr;
};

thread_local std::string last_error;

void set_error(const char* message) {
    last_error = message;
}

bool cuda_ok(cudaError_t status, const char* what) {
    if(status == cudaSuccess) {
        return true;
    }
    last_error = std::string(what) + ": " + cudaGetErrorString(status);
    return false;
}

void normalize3(T v[3], const T fallback[3]) {
    const T norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if(norm_sq <= static_cast<T>(1e-12)) {
        v[0] = fallback[0];
        v[1] = fallback[1];
        v[2] = fallback[2];
        return;
    }
    const T inv_norm = static_cast<T>(1) / std::sqrt(norm_sq);
    v[0] *= inv_norm;
    v[1] *= inv_norm;
    v[2] *= inv_norm;
}

void cross3(const T a[3], const T b[3], T out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

RL_TOOLS_FUNCTION_PLACEMENT void sub3_device(const T a[3], const T b[3], T out[3]) {
    out[0] = a[0] - b[0];
    out[1] = a[1] - b[1];
    out[2] = a[2] - b[2];
}

RL_TOOLS_FUNCTION_PLACEMENT void add3_device(const T a[3], const T b[3], T out[3]) {
    out[0] = a[0] + b[0];
    out[1] = a[1] + b[1];
    out[2] = a[2] + b[2];
}

RL_TOOLS_FUNCTION_PLACEMENT void scale3_device(const T in[3], T scale, T out[3]) {
    out[0] = in[0] * scale;
    out[1] = in[1] * scale;
    out[2] = in[2] * scale;
}

RL_TOOLS_FUNCTION_PLACEMENT void cross3_device(const T a[3], const T b[3], T out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename DEVICE>
RL_TOOLS_FUNCTION_PLACEMENT void normalize3_device(DEVICE& device, const T in[3], T out[3]) {
    const T norm_sq = in[0] * in[0] + in[1] * in[1] + in[2] * in[2];
    if(norm_sq <= static_cast<T>(1e-12)) {
        out[0] = 1;
        out[1] = 0;
        out[2] = 0;
        return;
    }
    const T inv_norm = static_cast<T>(1) / rlt::math::sqrt(device.math, norm_sq);
    out[0] = in[0] * inv_norm;
    out[1] = in[1] * inv_norm;
    out[2] = in[2] * inv_norm;
}

template <typename DEVICE>
RL_TOOLS_FUNCTION_PLACEMENT CAMERA_DATA make_camera_data_device(DEVICE& device, const T position[3], const T look_at[3], const T up[3], T fov, T aspect) {
    T raw_dir[3];
    T dir[3];
    sub3_device(look_at, position, raw_dir);
    normalize3_device(device, raw_dir, dir);

    const T image_plane_scale = static_cast<T>(2) * tanf(fov / static_cast<T>(2));

    T du_cross[3];
    T du_dir[3];
    T du[3];
    cross3_device(dir, up, du_cross);
    normalize3_device(device, du_cross, du_dir);
    scale3_device(du_dir, image_plane_scale, du);

    T dv_cross[3];
    T dv_dir[3];
    T dv[3];
    cross3_device(du, dir, dv_cross);
    normalize3_device(device, dv_cross, dv_dir);
    scale3_device(dv_dir, image_plane_scale / aspect, dv);

    T half_du[3];
    T half_dv[3];
    T tmp[3];
    CAMERA_DATA cam;
    scale3_device(du, static_cast<T>(-0.5), half_du);
    scale3_device(dv, static_cast<T>(0.5), half_dv);
    add3_device(dir, half_du, tmp);
    add3_device(tmp, half_dv, cam.dir_00);
    cam.pos[0] = position[0];
    cam.pos[1] = position[1];
    cam.pos[2] = position[2];
    cam.dir_du[0] = du[0];
    cam.dir_du[1] = du[1];
    cam.dir_du[2] = du[2];
    scale3_device(dv, static_cast<T>(-1), cam.dir_dv);
    return cam;
}

void quaternion_from_rotation_columns(const T x_axis[3], const T y_axis[3], const T z_axis[3], T q[4]) {
    const T m00 = x_axis[0];
    const T m01 = y_axis[0];
    const T m02 = z_axis[0];
    const T m10 = x_axis[1];
    const T m11 = y_axis[1];
    const T m12 = z_axis[1];
    const T m20 = x_axis[2];
    const T m21 = y_axis[2];
    const T m22 = z_axis[2];
    const T trace = m00 + m11 + m22;
    if(trace > static_cast<T>(0)) {
        const T s = std::sqrt(trace + static_cast<T>(1)) * static_cast<T>(2);
        q[0] = static_cast<T>(0.25) * s;
        q[1] = (m21 - m12) / s;
        q[2] = (m02 - m20) / s;
        q[3] = (m10 - m01) / s;
    }
    else if(m00 > m11 && m00 > m22) {
        const T s = std::sqrt(static_cast<T>(1) + m00 - m11 - m22) * static_cast<T>(2);
        q[0] = (m21 - m12) / s;
        q[1] = static_cast<T>(0.25) * s;
        q[2] = (m01 + m10) / s;
        q[3] = (m02 + m20) / s;
    }
    else if(m11 > m22) {
        const T s = std::sqrt(static_cast<T>(1) + m11 - m00 - m22) * static_cast<T>(2);
        q[0] = (m02 - m20) / s;
        q[1] = (m01 + m10) / s;
        q[2] = static_cast<T>(0.25) * s;
        q[3] = (m12 + m21) / s;
    }
    else {
        const T s = std::sqrt(static_cast<T>(1) + m22 - m00 - m11) * static_cast<T>(2);
        q[0] = (m10 - m01) / s;
        q[1] = (m02 + m20) / s;
        q[2] = (m12 + m21) / s;
        q[3] = static_cast<T>(0.25) * s;
    }
    const T norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if(norm > static_cast<T>(1e-12)) {
        q[0] /= norm;
        q[1] /= norm;
        q[2] /= norm;
        q[3] /= norm;
    }
    else {
        q[0] = 1;
        q[1] = 0;
        q[2] = 0;
        q[3] = 0;
    }
}

void set_state_pose_from_camera(STATE& state, const T position[3], const PhysicsCameraPose& pose) {
    T x_axis[3] = {pose.direction[0], pose.direction[1], pose.direction[2]};
    const T fallback_x[3] = {1, 0, 0};
    normalize3(x_axis, fallback_x);

    T z_axis[3] = {pose.up[0], pose.up[1], pose.up[2]};
    const T fallback_z[3] = {0, 0, 1};
    normalize3(z_axis, fallback_z);

    T y_axis[3];
    cross3(z_axis, x_axis, y_axis);
    const T fallback_y[3] = {0, 1, 0};
    normalize3(y_axis, fallback_y);

    cross3(x_axis, y_axis, z_axis);
    normalize3(z_axis, fallback_z);

    state.position[0] = position[0];
    state.position[1] = position[1];
    state.position[2] = position[2];
    quaternion_from_rotation_columns(x_axis, y_axis, z_axis, state.orientation);
    for(TI axis_i = 0; axis_i < 3; axis_i++) {
        state.linear_velocity[axis_i] = 0;
        state.angular_velocity[axis_i] = 0;
    }
}

template <typename DEVICE, typename ACTION_SPEC>
RL_TOOLS_FUNCTION_PLACEMENT void make_benchmark_action(DEVICE& device, const PARAMETERS& parameters, TI env_i, TI iteration, rlt::Matrix<ACTION_SPEC>& action) {
    const T hover = parameters.dynamics.hovering_throttle_relative * static_cast<T>(2) - static_cast<T>(1);
    const T phase_a = static_cast<T>(0.013) * static_cast<T>(iteration) + static_cast<T>(0.0017) * static_cast<T>(env_i);
    const T phase_b = static_cast<T>(0.009) * static_cast<T>(iteration) + static_cast<T>(0.0023) * static_cast<T>(env_i);
    const T roll_mix = static_cast<T>(0.010) * rlt::math::sin(device.math, phase_a);
    const T pitch_mix = static_cast<T>(0.010) * rlt::math::cos(device.math, phase_b);
    rlt::set(action, 0, 0, rlt::math::clamp(device.math, hover + roll_mix, static_cast<T>(-1), static_cast<T>(1)));
    rlt::set(action, 0, 1, rlt::math::clamp(device.math, hover - roll_mix, static_cast<T>(-1), static_cast<T>(1)));
    rlt::set(action, 0, 2, rlt::math::clamp(device.math, hover + pitch_mix, static_cast<T>(-1), static_cast<T>(1)));
    rlt::set(action, 0, 3, rlt::math::clamp(device.math, hover - pitch_mix, static_cast<T>(-1), static_cast<T>(1)));
}

template <typename DEVICE>
RL_TOOLS_FUNCTION_PLACEMENT CAMERA_DATA make_camera_from_state(DEVICE& device, const STATE& state, T fov, T aspect) {
    const T offset_body[3] = {0, 0, 0};
    const T forward_body[3] = {1, 0, 0};
    const T up_body[3] = {0, 0, 1};
    T offset_world[3];
    T forward_world[3];
    T up_world[3];
    l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, offset_body, offset_world);
    l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, forward_body, forward_world);
    l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, up_body, up_world);
    T position[3] = {
        state.position[0] + offset_world[0],
        state.position[1] + offset_world[1],
        state.position[2] + offset_world[2]
    };
    T look_at[3] = {
        position[0] + forward_world[0],
        position[1] + forward_world[1],
        position[2] + forward_world[2]
    };
    return make_camera_data_device(device, position, look_at, up_world, fov, aspect);
}

template <typename DEVICE>
__global__ void step_cameras_kernel(
    DEVICE device,
    ENVIRONMENT* envs,
    PARAMETERS* parameters,
    STATE* states,
    const STATE* initial_states,
    CAMERA_DATA* cameras,
    TI num_envs,
    T fov,
    T aspect,
    TI iteration,
    std::uint32_t seed
) {
    const TI env_i = static_cast<TI>(blockIdx.x * blockDim.x + threadIdx.x);
    if(env_i >= num_envs) {
        return;
    }
    STATE state = states[env_i];
    if(iteration > 0 && iteration % PHYSICS_EPISODE_STEP_LIMIT == 0) {
        state = initial_states[env_i];
    }
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ACTION_DIM, false>> action;
    make_benchmark_action(device, parameters[env_i], env_i, iteration, action);
    STATE next_state;
    curandState rng;
    if(parameters[env_i].mdp.action_noise.normalized_rpm != static_cast<T>(0)){
        curand_init(static_cast<unsigned long long>(seed), static_cast<unsigned long long>(env_i), static_cast<unsigned long long>(iteration), &rng);
    }
    rlt::step(device, envs[env_i], parameters[env_i], state, action, next_state, rng);

    const STATE initial_state = initial_states[env_i];
    const T dx = next_state.position[0] - initial_state.position[0];
    const T dy = next_state.position[1] - initial_state.position[1];
    const T dz = next_state.position[2] - initial_state.position[2];
    const T drift_sq = dx * dx + dy * dy + dz * dz;
    if(drift_sq > static_cast<T>(16) || rlt::math::abs(device.math, dz) > static_cast<T>(3)) {
        next_state = initial_state;
    }
    states[env_i] = next_state;
    cameras[env_i] = make_camera_from_state(device, next_state, fov, aspect);
}

} // namespace

bool init_physics_simulation(
    PhysicsSimulation& simulation,
    int num_envs,
    const float camera_position[3],
    const PhysicsCameraPose* poses,
    float fov,
    float aspect,
    std::uint32_t seed
) {
    last_error.clear();
    if(num_envs <= 0) {
        set_error("num_envs must be positive");
        return false;
    }
    if(poses == nullptr) {
        set_error("camera poses pointer is null");
        return false;
    }
    free_physics_simulation(simulation);

    auto impl = std::make_unique<PhysicsSimulationImpl>();
    impl->num_envs = num_envs;
    impl->fov = fov;
    impl->aspect = aspect;
    impl->seed = seed;

    std::vector<ENVIRONMENT> host_envs(static_cast<size_t>(num_envs));
    std::vector<PARAMETERS> host_parameters(static_cast<size_t>(num_envs));
    std::vector<STATE> host_initial_states(static_cast<size_t>(num_envs));
    for(int env_i = 0; env_i < num_envs; env_i++) {
        host_envs[env_i].parameters = nominal_parameters;
        host_parameters[env_i] = nominal_parameters;
        host_initial_states[env_i] = {};
        const T hover_command = host_parameters[env_i].dynamics.hovering_throttle_relative *
            (host_parameters[env_i].dynamics.action_limit.max - host_parameters[env_i].dynamics.action_limit.min) +
            host_parameters[env_i].dynamics.action_limit.min;
        for(TI rotor_i = 0; rotor_i < ACTION_DIM; rotor_i++) {
            host_initial_states[env_i].rpm[rotor_i] = hover_command;
        }
        set_state_pose_from_camera(host_initial_states[env_i], camera_position, poses[env_i]);
    }

    const size_t env_bytes = static_cast<size_t>(num_envs) * sizeof(ENVIRONMENT);
    const size_t parameter_bytes = static_cast<size_t>(num_envs) * sizeof(PARAMETERS);
    const size_t state_bytes = static_cast<size_t>(num_envs) * sizeof(STATE);
    auto free_impl_allocations = [&impl]() {
        cudaFree(impl->envs);
        cudaFree(impl->parameters);
        cudaFree(impl->states);
        cudaFree(impl->initial_states);
        impl->envs = nullptr;
        impl->parameters = nullptr;
        impl->states = nullptr;
        impl->initial_states = nullptr;
    };
    if(!cuda_ok(cudaMalloc(&impl->envs, env_bytes), "cudaMalloc envs") ||
       !cuda_ok(cudaMalloc(&impl->parameters, parameter_bytes), "cudaMalloc parameters") ||
       !cuda_ok(cudaMalloc(&impl->states, state_bytes), "cudaMalloc states") ||
       !cuda_ok(cudaMalloc(&impl->initial_states, state_bytes), "cudaMalloc initial states")) {
        free_impl_allocations();
        return false;
    }
    if(!cuda_ok(cudaMemcpy(impl->envs, host_envs.data(), env_bytes, cudaMemcpyHostToDevice), "cudaMemcpy envs") ||
       !cuda_ok(cudaMemcpy(impl->parameters, host_parameters.data(), parameter_bytes, cudaMemcpyHostToDevice), "cudaMemcpy parameters") ||
       !cuda_ok(cudaMemcpy(impl->states, host_initial_states.data(), state_bytes, cudaMemcpyHostToDevice), "cudaMemcpy states") ||
       !cuda_ok(cudaMemcpy(impl->initial_states, host_initial_states.data(), state_bytes, cudaMemcpyHostToDevice), "cudaMemcpy initial states")) {
        free_impl_allocations();
        return false;
    }

    simulation.impl = impl.release();
    return true;
}

bool physics_step_cameras(
    PhysicsSimulation& simulation,
    void* camera_buffer_device_ptr,
    void* cuda_stream,
    int iteration
) {
    last_error.clear();
    auto* impl = static_cast<PhysicsSimulationImpl*>(simulation.impl);
    if(impl == nullptr) {
        set_error("physics simulation is not initialized");
        return false;
    }
    if(camera_buffer_device_ptr == nullptr) {
        set_error("camera buffer pointer is null");
        return false;
    }
    constexpr int BLOCK = 256;
    const int grid = (impl->num_envs + BLOCK - 1) / BLOCK;
    DEVICE_GPU device;
    auto* cameras = static_cast<CAMERA_DATA*>(camera_buffer_device_ptr);
    auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
    step_cameras_kernel<<<grid, BLOCK, 0, stream>>>(
        device,
        impl->envs,
        impl->parameters,
        impl->states,
        impl->initial_states,
        cameras,
        impl->num_envs,
        impl->fov,
        impl->aspect,
        iteration,
        impl->seed
    );
    return cuda_ok(cudaPeekAtLastError(), "step_cameras_kernel launch");
}

void free_physics_simulation(PhysicsSimulation& simulation) {
    auto* impl = static_cast<PhysicsSimulationImpl*>(simulation.impl);
    if(impl == nullptr) {
        return;
    }
    cudaFree(impl->envs);
    cudaFree(impl->parameters);
    cudaFree(impl->states);
    cudaFree(impl->initial_states);
    delete impl;
    simulation.impl = nullptr;
}

const char* physics_last_error() {
    return last_error.c_str();
}

} // namespace rl_tools::rendering::raytracing::benchmark
