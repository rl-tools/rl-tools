#pragma once
#include <cstdint>

// ABI boundary between the hyperdrone dynamics core module and the JIT-compiled simulator
// libraries. Buffer exchange uses raw float pointers sized by config(); state is published
// after every mutation as SoA component buffers (device-resident on the CUDA variant) so
// consumers get stable zero-copy views. Bump HYPERDRONE_DYNAMICS_IFACE_VERSION on any
// change to this file.
#define HYPERDRONE_DYNAMICS_IFACE_VERSION 3

namespace hyperdrone::dynamics {
    struct Config {
        uint32_t num_drones;
        uint32_t action_dim;
        uint32_t observation_dim;
        bool domain_randomization;
    };

    // SoA state component buffers, each (num_drones, dim) float32
    enum class StateComponent : int {
        POSITION = 0,         // 3
        ORIENTATION = 1,      // 4, quaternion wxyz
        LINEAR_VELOCITY = 2,  // 3
        ANGULAR_VELOCITY = 3, // 3
        RPM = 4               // action_dim
    };
    inline constexpr int state_component_dims[5] = {3, 4, 3, 3, 4};

    class Sim {
    public:
        virtual ~Sim() = default;
        virtual Config config() const = 0;
        virtual const char* device_name() const = 0;    // "cpu" | "cuda"
        virtual int buffer_device_type() const = 0;     // DLPack: 1 = kDLCPU, 2 = kDLCUDA
        virtual unsigned long long stream() const = 0;  // cudaStream_t handle (0 on cpu)
        virtual void synchronize() = 0;

        virtual bool set_model(const char* name) = 0;   // registry preset, all drones
        virtual void set_dt(float dt) = 0;
        virtual float dt() const = 0;

        // deterministic given seed: default parameters (+ optional domain-randomization
        // sampling) and sampled initial states. All sampling runs on the host with the
        // rl_tools CPU engine so cpu and cuda variants reset to identical states.
        virtual void reset(unsigned long long seed, bool sample_parameters, bool sample_states) = 0;

        // actions (num_drones, action_dim) float32, normalized [-1, 1];
        // actions_device_type: 1 = host pointer, 2 = CUDA device pointer
        virtual void step(const float* actions, int actions_device_type, unsigned long long producer_stream) = 0;

        virtual const float* state_buffer(int component) const = 0;
        virtual void set_state_component(int component, const float* values_host) = 0;

        virtual void update_observations() = 0;
        virtual const float* observation_buffer() const = 0;

        // packed camera ray-gen bases (num_drones, 12): pos, dir_00, dir_du, dir_dv in the
        // renderer's layout. mount is a (3, 4) body-frame transform: rotation columns map
        // camera axes (FLU: +X forward) into the body frame, translation is the offset.
        virtual void update_camera_bases(const float mount[12], float fov, float aspect) = 0;
        virtual const float* camera_bases_buffer() const = 0;

        // per-drone runtime physical parameters (host pointers, num_drones scalars each;
        // parameters are host-authoritative). False for unknown names.
        virtual bool read_parameter(const char* name, float* dst) = 0;
        virtual bool write_parameter(const char* name, const float* values) = 0;
    };
}

extern "C" {
    hyperdrone::dynamics::Sim* hyperdrone_dynamics_create();
    void hyperdrone_dynamics_destroy(hyperdrone::dynamics::Sim* sim);
    const char* hyperdrone_dynamics_config_string();
    int hyperdrone_dynamics_iface_version();
}
