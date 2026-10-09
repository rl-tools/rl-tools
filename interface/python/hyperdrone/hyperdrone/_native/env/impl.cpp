#if defined(HYPERDRONE_ENV_CUDA)
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#endif
#include "environment.h"
#if HYPERDRONE_ENV_DEVICE_CUDA
#include <rl_tools/rl/environments/hyperdrone/operations_cuda.h>
#if HYPERDRONE_ENV_TASK == 1
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cuda.h>
#endif
#if HYPERDRONE_ENV_TASK == 2
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cuda.h>
#endif
#include <cuda_runtime.h>
#endif

#include "../common/models.h"

#include <filesystem>
#include <optional>
#include <stdexcept>

namespace HYPERDRONE_MODULE_NAME {
#if HYPERDRONE_ENV_DEVICE_CUDA
    using DEVICE = rlt::devices::CUDA<rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, HOST_DEVICE>>;
    // one curand state per instance across all members (the composite partitions them per member)
    using RNG = rlt::devices::random::CUDA::ENGINE<rlt::devices::random::CUDA::Specification<TI, TOTAL>>;
    struct CurrentDevice {
        int previous = 0;
        int device;
        explicit CurrentDevice(int device): device(device){
            cudaGetDevice(&previous);
            if(previous != device){
                cudaSetDevice(device);
            }
        }
        ~CurrentDevice(){
            if(previous != device){
                cudaSetDevice(previous);
            }
        }
    };
#else
    using DEVICE = HOST_DEVICE;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    struct CurrentDevice {
        explicit CurrentDevice(int){}
    };
#endif

    template <typename ELEMENT, TI... DIMS>
    using InstanceTensor = rlt::Tensor<rlt::tensor::Specification<ELEMENT, TI, rlt::tensor::Shape<TI, TOTAL, DIMS...>>>;

    struct Environment::Impl {
        DEVICE device{};
        ENV env{};
        RNG rng{};
        int device_id = 0;
        InstanceTensor<typename WORLD::Parameters> parameters{};
        InstanceTensor<typename WORLD::Parameters> parameters_host{};
        InstanceTensor<typename WORLD::State> states{}, next_states{};
        InstanceTensor<T> rewards{};
        InstanceTensor<bool> terminated_flags{}, all_instances{}, no_instances{};
        // host arrays pass through these on a CUDA World
        InstanceTensor<bool> mask_staging{};
        InstanceTensor<T, WORLD::ACTION_DIM> actions_staging{};
        InstanceTensor<T, WORLD::OBSERVATION_DIM> observations_staging{};
        InstanceTensor<T, WORLD::OBSERVATION_DIM_PRIVILEGED> observations_privileged_staging{};
        InstanceTensor<T, (OBSERVATION_DIM_IMU > 0 ? OBSERVATION_DIM_IMU : 1)> observations_imu_staging{};

        HOST_DEVICE& host(){
            return rlt::get_rendering_device(device);
        }
        template <typename SPEC>
        rlt::Tensor<SPEC> input(const void* source, bool resident, rlt::Tensor<SPEC>& staging){
            rlt::Tensor<SPEC> alias;
            alias._data = (typename SPEC::T*)source;
            if(resident){
                return alias;
            }
            rlt::copy(host(), device, alias, staging);
            return staging;
        }
        template <typename SPEC, typename PRODUCE>
        void output(void* destination, bool resident, rlt::Tensor<SPEC>& staging, PRODUCE produce){
            rlt::Tensor<SPEC> alias;
            alias._data = (typename SPEC::T*)destination;
            if(resident){
                produce(alias);
                return;
            }
            produce(staging);
            rlt::copy(device, host(), staging, alias);
        }
        template <typename SPEC>
        void copy_out(rlt::Tensor<SPEC>& source, void* destination, bool resident){
            rlt::Tensor<SPEC> alias;
            alias._data = (typename SPEC::T*)destination;
            if(resident){
                rlt::copy(device, device, source, alias);
            }
            else{
                rlt::copy(device, host(), source, alias);
            }
        }
        // the per-instance parameters as a host-readable tensor (the World's own on a host World)
        InstanceTensor<typename WORLD::Parameters>& parameters_on_host(){
            if constexpr (CUDA_WORLD){
                rlt::copy(device, host(), parameters, parameters_host);
                return parameters_host;
            }
            else{
                return parameters;
            }
        }
        void publish_parameters_from_host(){
            if constexpr (CUDA_WORLD){
                rlt::copy(host(), device, parameters_host, parameters);
            }
        }
    };

    // one schema for the nominal, the drone table and the per-instance parameters: {"dynamics":
    // <l2f parameters>, "camera": {...}}; instances also carry "drone", their table entry
    template <typename DYNAMICS_PARAMETERS>
    std::string parameters_to_json(Environment::Impl& impl, const DYNAMICS_PARAMETERS& dynamics, const rlt::rl::environments::hyperdrone::Camera<T>& camera, std::optional<TI> drone = std::nullopt){
        const auto& world = impl.env.environments[0];
        std::string value = "{\"dynamics\": " + rlt::json(impl.host(), world.dynamics, dynamics) + ", \"camera\": " + rlt::json(impl.host(), world, camera);
        if(drone){
            value += ", \"drone\": " + std::to_string(*drone);
        }
        return value + "}";
    }
    template <typename DYNAMICS_PARAMETERS>
    void parameters_from_json(Environment::Impl& impl, const nlohmann::json& parameters, DYNAMICS_PARAMETERS& dynamics, rlt::rl::environments::hyperdrone::Camera<T>& camera){
        auto& world = impl.env.environments[0];
        rlt::from_json(impl.host(), world.dynamics, parameters.at("dynamics"), dynamics);
        rlt::from_json(impl.host(), world, parameters.at("camera"), camera);
    }

    template <typename WORLD_TYPE, typename = void>
    struct HasGateAssetPath { static constexpr bool VALUE = false; };
    template <typename WORLD_TYPE>
    struct HasGateAssetPath<WORLD_TYPE, rlt::utils::typing::void_t<decltype(WORLD_TYPE::gate_asset_path)>> { static constexpr bool VALUE = true; };

    template <typename WORLD_TYPE>
    void set_gate_asset_path([[maybe_unused]] WORLD_TYPE& world, [[maybe_unused]] const std::string& path){
        if constexpr (HasGateAssetPath<WORLD_TYPE>::VALUE){
            world.gate_asset_path = path;
        }
    }

    // the drone table's checks as exceptions: the World asserts them, which would end the process
    void check_drones(Environment::Impl& impl){
        using SPEC = typename WORLD::SPEC;
        const auto& drones = impl.env.shared.drones;
        if(drones.empty() || drones.size() > SPEC::MAX_DRONES){
            throw std::invalid_argument("hyperdrone: the drone table needs 1 to " + std::to_string(SPEC::MAX_DRONES) + " entries");
        }
        if constexpr (SPEC::SELF_VISIBLE){
            TI parts = 0;
            for(size_t drone_i = 0; drone_i < drones.size(); drone_i++){
                const auto& drone = drones[drone_i];
                const std::string name = "hyperdrone: drone " + std::to_string(drone_i) + " (" + drone.asset + ")";
                std::string path;
                rlt::rendering::raytracing::ObjectAssembly assembly;
                if(drone.asset.empty() || !rlt::rendering::datasets::resolve_reference(impl.host(), drone.asset, path) || !rlt::load<typename SPEC::SHADING, SPEC::OUTPUT_RGB>(impl.host(), assembly, path)){
                    throw std::invalid_argument(name + ": failed to load the asset");
                }
                rlt::rl::environments::hyperdrone::DroneBody<SPEC> body;
                const auto& rotors = drone.parameters.dynamics.dynamics;
                if(!rlt::init(impl.host(), body.rig, assembly, rotors.rotor_positions, rotors.rotor_torque_directions, (TI)WORLD::DYNAMICS_ENV::ACTION_DIM)){
                    throw std::invalid_argument(name + ": the asset needs a body and one prop_* per rotor, each hub nearest a distinct rotor with a nonzero reaction torque");
                }
                if(drone_i > 0 && body.rig.num_parts != parts){
                    throw std::invalid_argument(name + ": all drone assets need the same part count, the body swaps in place at reset");
                }
                parts = body.rig.num_parts;
            }
        }
    }

    template <typename WORLD_TYPE>
    void observe_imu_into([[maybe_unused]] Environment::Impl& impl, [[maybe_unused]] float* out, [[maybe_unused]] bool resident){
        if constexpr (ObservationIMUDim<WORLD_TYPE>::VALUE > 0){
            impl.output(out, resident, impl.observations_imu_staging, [&](auto& observations){
                rlt::observe(impl.device, impl.env, impl.parameters, impl.states, typename WORLD_TYPE::ObservationIMU{}, observations, impl.rng);
            });
        }
    }

    Environment::Environment(): impl(new Impl()){
        if constexpr (CUDA_WORLD){
            rlt::init(impl->host());
        }
        rlt::init(impl->device);
#if HYPERDRONE_ENV_DEVICE_CUDA
        cudaGetDevice(&impl->device_id);
#endif
        rlt::malloc(impl->device, impl->env);
        rlt::malloc(impl->device, impl->rng);
        rlt::init(impl->device, impl->rng, 0);
        rlt::malloc(impl->device, impl->parameters);
        rlt::malloc(impl->device, impl->states);
        rlt::malloc(impl->device, impl->next_states);
        rlt::malloc(impl->device, impl->rewards);
        rlt::malloc(impl->device, impl->terminated_flags);
        rlt::malloc(impl->device, impl->all_instances);
        rlt::malloc(impl->device, impl->no_instances);
        rlt::set_all(impl->device, impl->rewards, (T)0);
        rlt::set_all(impl->device, impl->terminated_flags, false);
        rlt::set_all(impl->device, impl->all_instances, true);
        rlt::set_all(impl->device, impl->no_instances, false);
        rlt::malloc(impl->host(), impl->parameters_host);
        {
            // every instance holds defined values before its first reset
            InstanceTensor<typename WORLD::State> states_host;
            rlt::malloc(impl->host(), states_host);
            for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
                rlt::set(impl->host(), impl->parameters_host, typename WORLD::Parameters{}, instance_i);
                rlt::set(impl->host(), states_host, typename WORLD::State{}, instance_i);
            }
            rlt::copy(impl->host(), impl->device, impl->parameters_host, impl->parameters);
            rlt::copy(impl->host(), impl->device, states_host, impl->states);
            rlt::free(impl->host(), states_host);
        }
        if constexpr (CUDA_WORLD){
            rlt::malloc(impl->device, impl->mask_staging);
            rlt::malloc(impl->device, impl->actions_staging);
            rlt::malloc(impl->device, impl->observations_staging);
            rlt::malloc(impl->device, impl->observations_privileged_staging);
            rlt::malloc(impl->device, impl->observations_imu_staging);
        }
    }
    Environment::~Environment(){
        const CurrentDevice current(impl->device_id);
        if constexpr (CUDA_WORLD){
            rlt::free(impl->device, impl->mask_staging);
            rlt::free(impl->device, impl->actions_staging);
            rlt::free(impl->device, impl->observations_staging);
            rlt::free(impl->device, impl->observations_privileged_staging);
            rlt::free(impl->device, impl->observations_imu_staging);
        }
        rlt::free(impl->host(), impl->parameters_host);
        rlt::free(impl->device, impl->parameters);
        rlt::free(impl->device, impl->states);
        rlt::free(impl->device, impl->next_states);
        rlt::free(impl->device, impl->rewards);
        rlt::free(impl->device, impl->terminated_flags);
        rlt::free(impl->device, impl->all_instances);
        rlt::free(impl->device, impl->no_instances);
        rlt::free(impl->device, impl->rng);
        rlt::free(impl->device, impl->env);
        rlt::free(impl->device);
        delete impl;
    }
    void Environment::init(const std::vector<std::string>& references, const std::vector<std::optional<std::string>>& drone_assets, const std::vector<std::string>& drone_parameters, const std::optional<std::string>& gate_asset_path, unsigned long long seed){
        const CurrentDevice current(impl->device_id);
        rlt::init(impl->device, impl->rng, seed);
        if(drone_assets.size() != drone_parameters.size()){
            throw std::invalid_argument("hyperdrone: drone assets and parameters must pair up");
        }
        for(size_t drone_i = 0; drone_i < drone_assets.size(); drone_i++){
            typename ENV::SharedContext::DRONE drone;
            drone.asset = drone_assets[drone_i].value_or("");
            parameters_from_json(*impl, nlohmann::json::parse(drone_parameters[drone_i]), drone.parameters.dynamics, drone.parameters.camera);
            impl->env.shared.drones.push_back(drone);
        }
        check_drones(*impl);
        if(gate_asset_path){
            for(TI environment_i = 0; environment_i < NUM_ENVIRONMENTS; environment_i++){
                set_gate_asset_path(impl->env.environments[environment_i], *gate_asset_path);
            }
        }
        rlt::rendering::datasets::procthor::GLB dataset{};
        if(references.size() == 1 && std::filesystem::is_directory(references[0])){
            dataset.directory = references[0];
        }
        else{
            dataset.references = references;
        }
        rlt::init(impl->device, impl->env, dataset);
    }
    std::string Environment::parameters_json(){
        const auto& world = impl->env.environments[0];
        return parameters_to_json(*impl, world.dynamics.parameters, world.parameters.camera);
    }
    std::string Environment::drones_json(){
        std::string value = "[";
        for(size_t drone_i = 0; drone_i < impl->env.shared.drones.size(); drone_i++){
            const auto& drone = impl->env.shared.drones[drone_i];
            const std::string asset = drone.asset.empty() ? "null" : nlohmann::json(drone.asset).dump();
            value += (drone_i > 0 ? ", " : "") + std::string("{\"asset\": ") + asset + ", \"parameters\": " + parameters_to_json(*impl, drone.parameters.dynamics, drone.parameters.camera) + "}";
        }
        return value + "]";
    }
    std::string Environment::instance_parameters_json(){
        const CurrentDevice current(impl->device_id);
        auto& parameters = impl->parameters_on_host();
        std::string value = "[";
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            const auto& instance = rlt::get_ref(impl->host(), parameters, instance_i);
            value += (instance_i > 0 ? ", " : "") + parameters_to_json(*impl, instance.dynamics, instance.camera, instance.drone);
        }
        value += "]";
        return value;
    }
    void Environment::set_instance_parameters(const bool* mask, const std::string& parameters_json_array){
        const CurrentDevice current(impl->device_id);
        auto& parameters = impl->parameters_on_host();
        const nlohmann::json entries = nlohmann::json::parse(parameters_json_array);
        std::vector<typename WORLD::Parameters> updated;
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            if(mask[instance_i]){
                updated.push_back(rlt::get_ref(impl->host(), parameters, instance_i));
                const auto& entry = entries.at(updated.size() - 1);
                parameters_from_json(*impl, entry, updated.back().dynamics, updated.back().camera);
                if(entry.contains("drone")){
                    updated.back().drone = entry.at("drone").get<TI>();
                    if(updated.back().drone >= impl->env.shared.num_drones){
                        throw std::invalid_argument("hyperdrone: drone must index the drone table");
                    }
                }
            }
        }
        if(updated.size() != entries.size()){
            throw std::invalid_argument("hyperdrone: set_instance_parameters takes one entry per selected instance");
        }
        TI entry_i = 0;
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            if(mask[instance_i]){
                rlt::get_ref(impl->host(), parameters, instance_i) = updated[entry_i++];
            }
        }
        impl->publish_parameters_from_host();
    }
    std::optional<std::string> Environment::model(const std::string& name){
        DYNAMICS dynamics;
        if(!hyperdrone::model_by_name<typename WORLD::SPEC>(name, dynamics)){
            return std::nullopt;
        }
        return rlt::json(impl->host(), impl->env.environments[0].dynamics, dynamics);
    }
    void Environment::rotate_scene(){
        const CurrentDevice current(impl->device_id);
        rlt::rotate_scene(impl->device, impl->env);
    }

    void Environment::reset(const bool* mask, bool resident){
        const CurrentDevice current(impl->device_id);
        const auto reset_mask = mask == nullptr ? impl->all_instances : impl->input(mask, resident, impl->mask_staging);
        rlt::sample_initial_parameters(impl->device, impl->env, impl->parameters, reset_mask, impl->rng);
        rlt::sample_initial_state(impl->device, impl->env, impl->parameters, impl->states, reset_mask, impl->rng);
    }
    void Environment::render(const bool* mask, bool resident){
        const CurrentDevice current(impl->device_id);
        const auto reset_mask = mask == nullptr ? impl->no_instances : impl->input(mask, resident, impl->mask_staging);
        rlt::render(impl->device, impl->env, impl->parameters, impl->states, reset_mask);
    }
    void Environment::step(const float* actions, bool resident){
        const CurrentDevice current(impl->device_id);
        const auto action_tensor = impl->input(actions, resident, impl->actions_staging);
        rlt::step(impl->device, impl->env, impl->parameters, impl->states, action_tensor, impl->next_states, impl->rng);
        rlt::reward(impl->device, impl->env, impl->parameters, impl->states, action_tensor, impl->next_states, impl->rewards, impl->rng);
        rlt::copy(impl->device, impl->device, impl->next_states, impl->states);
        rlt::terminated(impl->device, impl->env, impl->parameters, impl->states, impl->terminated_flags, impl->rng);
    }
    void Environment::observe(float* out, bool resident){
        const CurrentDevice current(impl->device_id);
        impl->output(out, resident, impl->observations_staging, [&](auto& observations){
            rlt::observe(impl->device, impl->env, impl->parameters, impl->states, typename WORLD::Observation{}, observations, impl->rng);
        });
    }
    void Environment::observe_privileged(float* out, bool resident){
        const CurrentDevice current(impl->device_id);
        impl->output(out, resident, impl->observations_privileged_staging, [&](auto& observations){
            rlt::observe(impl->device, impl->env, impl->parameters, impl->states, typename WORLD::ObservationPrivileged{}, observations, impl->rng);
        });
    }
    void Environment::observe_imu(float* out, bool resident){
        const CurrentDevice current(impl->device_id);
        observe_imu_into<WORLD>(*impl, out, resident);
    }
    void Environment::rewards(float* out, bool resident){
        const CurrentDevice current(impl->device_id);
        impl->copy_out(impl->rewards, out, resident);
    }
    void Environment::terminated(bool* out, bool resident){
        const CurrentDevice current(impl->device_id);
        impl->copy_out(impl->terminated_flags, out, resident);
    }
    uintptr_t Environment::stream() const {
#if HYPERDRONE_ENV_DEVICE_CUDA
        return (uintptr_t)impl->device.stream;
#else
        return 0;
#endif
    }
    void Environment::set_stream([[maybe_unused]] uintptr_t stream){
#if HYPERDRONE_ENV_DEVICE_CUDA
        const CurrentDevice current(impl->device_id);
        // recording the World's event fails on a stream of another GPU
        if(cudaEventRecord(impl->device.ordering_event, (cudaStream_t)stream) != cudaSuccess){
            cudaGetLastError();
            throw std::invalid_argument("hyperdrone: the stream must belong to the environment's GPU (device_id)");
        }
        rlt::set_stream(impl->device, (cudaStream_t)stream);
#endif
    }
    int Environment::device_id() const {
        return impl->device_id;
    }
    void Environment::synchronize(){
#if HYPERDRONE_ENV_DEVICE_CUDA
        const CurrentDevice current(impl->device_id);
        cudaStreamSynchronize(impl->device.stream);
#endif
    }
}
