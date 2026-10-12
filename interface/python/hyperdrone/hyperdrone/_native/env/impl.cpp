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
#ifdef RL_TOOLS_ENABLE_HYPERDRONE_SPY
#include <rl_tools/rl/environments/hyperdrone/spy/operations_cpu.h>
#if HYPERDRONE_ENV_DEVICE_CUDA
#include <rl_tools/rl/environments/hyperdrone/spy/operations_cuda.h>
#endif
#endif

#include "../common/models.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace HYPERDRONE_MODULE_NAME {
    using rl_tools::view_range;
    using rl_tools::render;
    using rl_tools::output;
    using rl_tools::get_renderer;
    using rl_tools::copy;
    using rl_tools::request_render;

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
#ifdef RL_TOOLS_ENABLE_HYPERDRONE_SPY
        rlt::rl::environments::hyperdrone::Spy<rlt::rl::environments::hyperdrone::spy::Config<ENV>> spy{};
        bool spy_open = false;
#endif
        int device_id = 0;
        std::array<bool, TOTAL> sampled{};
        bool all_sampled = false;
        InstanceTensor<typename WORLD::Parameters> parameters{};
        InstanceTensor<typename WORLD::Parameters> parameters_host{};
        InstanceTensor<typename WORLD::State> states_host{};
        InstanceTensor<typename WORLD::State> states{}, next_states{};
        InstanceTensor<T> rewards{};
        InstanceTensor<bool> terminated_flags{}, all_instances{};
        // host arrays pass through these on a CUDA World
        InstanceTensor<bool> mask_staging{};
        InstanceTensor<T, WORLD::ACTION_DIM> actions_staging{};
        InstanceTensor<T, WORLD::OBSERVATION_DIM> observations_staging{};
        InstanceTensor<T, WORLD::OBSERVATION_DIM_PRIVILEGED> observations_privileged_staging{};
        InstanceTensor<T, (OBSERVATION_DIM_IMU > 0 ? OBSERVATION_DIM_IMU : 1)> observations_imu_staging{};

        HOST_DEVICE& host(){
            return rlt::get_rendering_device(device);
        }
        void require_sampled(const char* verb) const {
            if(!all_sampled){
                throw std::runtime_error(std::string("hyperdrone: reset every instance before ") + verb);
            }
        }
        std::array<bool, TOTAL> reset_selection(const bool* mask, bool resident){
            std::array<bool, TOTAL> selected{};
            if(mask == nullptr){
                selected.fill(true);
            }
            else if(CUDA_WORLD && resident){
                InstanceTensor<bool> source, destination;
                source._data = const_cast<bool*>(mask);
                destination._data = selected.data();
                rlt::copy(device, host(), source, destination);
            }
            else{
                std::copy(mask, mask + TOTAL, selected.begin());
            }
            if constexpr (SynchronizedResets<WORLD>::VALUE){
                for(TI environment_i = 0; environment_i < NUM_ENVIRONMENTS; environment_i++){
                    const auto first = selected.begin() + environment_i * WORLD::INSTANCES;
                    const auto last = first + WORLD::INSTANCES;
                    if(std::any_of(first, last, [](bool value){ return value; }) && !std::all_of(first, last, [](bool value){ return value; })){
                        throw std::invalid_argument("hyperdrone: the reset mask must be all-or-none per environment");
                    }
                }
            }
            return selected;
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
        // the per-instance parameters and states as host-readable tensors (the World's own on a
        // host World); edits reach a CUDA World through publish
        template <typename SPEC>
        rlt::Tensor<SPEC>& on_host(rlt::Tensor<SPEC>& resident, rlt::Tensor<SPEC>& staging){
            if constexpr (CUDA_WORLD){
                rlt::copy(device, host(), resident, staging);
                return staging;
            }
            else{
                return resident;
            }
        }
        template <typename SPEC>
        void publish(rlt::Tensor<SPEC>& staging, rlt::Tensor<SPEC>& resident){
            if constexpr (CUDA_WORLD){
                rlt::copy(host(), device, staging, resident);
            }
        }
        InstanceTensor<typename WORLD::Parameters>& parameters_on_host(){
            return on_host(parameters, parameters_host);
        }
        void publish_parameters_from_host(){
            publish(parameters_host, parameters);
        }
    };

    template <typename DYNAMICS_PARAMETERS>
    std::string parameters_to_json(Environment::Impl& impl, const DYNAMICS_PARAMETERS& dynamics, const rlt::rl::environments::hyperdrone::Camera<T> (&cameras)[WORLD::N_CAMERAS_PER_AGENT], std::optional<TI> drone = std::nullopt){
        const auto& world = impl.env.environments[0];
        std::string value = "{\"dynamics\": " + rlt::json(impl.host(), world.dynamics, dynamics) + ", \"cameras\": [";
        for(TI camera_i = 0; camera_i < WORLD::N_CAMERAS_PER_AGENT; camera_i++){
            value += (camera_i > 0 ? ", " : "") + rlt::json(impl.host(), world, cameras[camera_i]);
        }
        value += "]";
        if(drone){
            value += ", \"drone\": " + std::to_string(*drone);
        }
        return value + "}";
    }
    template <typename DYNAMICS_PARAMETERS>
    void parameters_from_json(Environment::Impl& impl, const nlohmann::json& parameters, DYNAMICS_PARAMETERS& dynamics, rlt::rl::environments::hyperdrone::Camera<T> (&cameras)[WORLD::N_CAMERAS_PER_AGENT]){
        auto& world = impl.env.environments[0];
        const auto& values = parameters.at("cameras");
        if(!values.is_array() || values.size() != WORLD::N_CAMERAS_PER_AGENT){
            throw std::invalid_argument("hyperdrone: provide one camera parameter dict per camera");
        }
        rlt::from_json(impl.host(), world.dynamics, parameters.at("dynamics"), dynamics);
        for(TI camera_i = 0; camera_i < WORLD::N_CAMERAS_PER_AGENT; camera_i++){
            rlt::from_json(impl.host(), world, values.at(camera_i), cameras[camera_i]);
        }
    }

    template <typename WORLD_TYPE, typename = void>
    struct HasGateAssetPath { static constexpr bool VALUE = false; };
    template <typename WORLD_TYPE>
    struct HasGateAssetPath<WORLD_TYPE, rlt::utils::typing::void_t<decltype(WORLD_TYPE::gate_asset_path)>> { static constexpr bool VALUE = true; };

    // the drone table's checks as exceptions: the World asserts them, which would end the process
    TI check_drones(Environment::Impl& impl){
        using SPEC = typename WORLD::SPEC;
        const auto& drones = impl.env.shared.drones;
        if(drones.empty() || drones.size() > SPEC::MAX_DRONES){
            throw std::invalid_argument("hyperdrone: the drone table needs 1 to " + std::to_string(SPEC::MAX_DRONES) + " entries");
        }
        for(const auto& drone : drones){
            if(drone.parameters.dynamics.integration.dt != drones[0].parameters.dynamics.integration.dt){
                throw std::invalid_argument("hyperdrone: every drone must use the same integration timestep");
            }
        }
        TI parts = 0;
        if constexpr (SPEC::SELF_VISIBLE){
            for(size_t drone_i = 0; drone_i < drones.size(); drone_i++){
                const auto& drone = drones[drone_i];
                const std::string name = "hyperdrone: drone " + std::to_string(drone_i) + " (" + drone.asset + ")";
                std::string path;
                rlt::rendering::raytracing::ObjectAssembly assembly;
                if(drone.asset.empty() || !rlt::rendering::datasets::resolve_reference(impl.host(), drone.asset, path) || !rlt::load<typename SPEC::SHADING, WORLD::RENDERER_SPEC::HAS_RGB>(impl.host(), assembly, path)){
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
        if(parts > SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE / WORLD::N_AGENTS){
            throw std::invalid_argument("hyperdrone: drone parts exceed MAX_ENTITY_SLOTS_PER_INSTANCE");
        }
        return parts * WORLD::N_AGENTS;
    }

    // a target-frame World adopted its setups' target frames at reset; replayed spawns move the
    // instances, so their target frames follow
    template <typename WORLD_TYPE, typename = void>
    struct HasTargetFrames { static constexpr bool VALUE = false; };
    template <typename WORLD_TYPE>
    struct HasTargetFrames<WORLD_TYPE, rlt::utils::typing::void_t<decltype(WORLD_TYPE::target_frames)>> { static constexpr bool VALUE = true; };
    template <typename WORLD_TYPE, typename ENV_TYPE, typename PARAMETER_SPEC, typename RESET_SPEC>
    void render_spawned_target_frames([[maybe_unused]] DEVICE& device, [[maybe_unused]] ENV_TYPE& env, [[maybe_unused]] rlt::Tensor<PARAMETER_SPEC>& parameters, [[maybe_unused]] const rlt::Tensor<RESET_SPEC>& reset_mask){
        if constexpr (HasTargetFrames<WORLD_TYPE>::VALUE){
            render_target_frames(device, env, parameters, reset_mask);
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
        rlt::set_all(impl->device, impl->rewards, (T)0);
        rlt::set_all(impl->device, impl->terminated_flags, false);
        rlt::set_all(impl->device, impl->all_instances, true);
        rlt::malloc(impl->host(), impl->parameters_host);
        rlt::malloc(impl->host(), impl->states_host);
        // every instance holds defined values before its first reset
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            rlt::set(impl->host(), impl->parameters_host, typename WORLD::Parameters{}, instance_i);
            rlt::set(impl->host(), impl->states_host, typename WORLD::State{}, instance_i);
        }
        rlt::copy(impl->host(), impl->device, impl->parameters_host, impl->parameters);
        rlt::copy(impl->host(), impl->device, impl->states_host, impl->states);
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
        spy_close();
        if constexpr (CUDA_WORLD){
            rlt::free(impl->device, impl->mask_staging);
            rlt::free(impl->device, impl->actions_staging);
            rlt::free(impl->device, impl->observations_staging);
            rlt::free(impl->device, impl->observations_privileged_staging);
            rlt::free(impl->device, impl->observations_imu_staging);
        }
        rlt::free(impl->host(), impl->parameters_host);
        rlt::free(impl->host(), impl->states_host);
        rlt::free(impl->device, impl->parameters);
        rlt::free(impl->device, impl->states);
        rlt::free(impl->device, impl->next_states);
        rlt::free(impl->device, impl->rewards);
        rlt::free(impl->device, impl->terminated_flags);
        rlt::free(impl->device, impl->all_instances);
        rlt::free(impl->device, impl->rng);
        rlt::free(impl->device, impl->env);
        rlt::free(impl->device);
        delete impl;
    }
    void Environment::init(const std::vector<std::string>& references, const std::vector<std::optional<std::string>>& drone_assets, const std::vector<std::string>& drone_parameters, const std::map<std::string, std::string>& assets, unsigned long long seed){
        if constexpr (HasGateAssetPath<WORLD>::VALUE){
            if(assets.count("gate") == 0){
                throw std::invalid_argument("hyperdrone: this world needs assets={'gate': ...} (the gate GLB)");
            }
        }
        const CurrentDevice current(impl->device_id);
        rlt::init(impl->device, impl->rng, seed);
        if(drone_assets.size() != drone_parameters.size()){
            throw std::invalid_argument("hyperdrone: drone assets and parameters must pair up");
        }
        for(size_t drone_i = 0; drone_i < drone_assets.size(); drone_i++){
            typename ENV::SharedContext::DRONE drone;
            drone.asset = drone_assets[drone_i].value_or("");
            parameters_from_json(*impl, nlohmann::json::parse(drone_parameters[drone_i]), drone.parameters.dynamics, drone.parameters.cameras);
            impl->env.shared.drones.push_back(drone);
        }
        const TI drone_parts = check_drones(*impl);
        if constexpr (HasGateAssetPath<WORLD>::VALUE){
            rlt::rendering::raytracing::ObjectAssembly assembly;
            if(!rlt::load<typename WORLD::SPEC::SHADING, WORLD::RENDERER_SPEC::HAS_RGB>(impl->host(), assembly, assets.at("gate")) || assembly.parts.empty()){
                throw std::invalid_argument("hyperdrone: failed to load the gate asset");
            }
            if(assembly.parts.size() > WORLD::SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE - drone_parts){
                throw std::invalid_argument("hyperdrone: gate and drone parts exceed MAX_ENTITY_SLOTS_PER_INSTANCE");
            }
        }
        for(const auto& asset : assets){
            for(TI environment_i = 0; environment_i < NUM_ENVIRONMENTS; environment_i++){
                if(!rlt::set_asset(impl->device, impl->env.environments[environment_i], asset.first, asset.second)){
                    throw std::invalid_argument("hyperdrone: no task of this world consumes asset '" + asset.first + "'");
                }
            }
        }
        DATASET dataset{};
        if(references.size() == 1 && std::filesystem::is_directory(references[0])){
            dataset.directory = references[0];
        }
        else{
            dataset.references = references;
        }
        rlt::init(impl->device, impl->env, dataset);
    }
    std::string Environment::segmentation_taxonomy() const {
        const auto& taxonomy = impl->env.shared.segmentation_taxonomy;
        return taxonomy ? taxonomy->name : std::string{};
    }
    std::map<uint32_t, std::string> Environment::segmentation_classes() const {
        const auto& taxonomy = impl->env.shared.segmentation_taxonomy;
        return taxonomy ? taxonomy->classes : std::map<uint32_t, std::string>{};
    }
    std::string Environment::parameters_json(){
        const auto& world = impl->env.environments[0];
        return parameters_to_json(*impl, world.dynamics.parameters, world.parameters.agents[0].cameras);
    }
    std::string Environment::drones_json(){
        std::string value = "[";
        for(size_t drone_i = 0; drone_i < impl->env.shared.drones.size(); drone_i++){
            const auto& drone = impl->env.shared.drones[drone_i];
            const std::string asset = drone.asset.empty() ? "null" : nlohmann::json(drone.asset).dump();
            value += (drone_i > 0 ? ", " : "") + std::string("{\"asset\": ") + asset + ", \"parameters\": " + parameters_to_json(*impl, drone.parameters.dynamics, drone.parameters.cameras) + "}";
        }
        return value + "]";
    }
    std::string Environment::instance_parameters_json(){
        impl->require_sampled("instance_parameters");
        const CurrentDevice current(impl->device_id);
        auto& parameters = impl->parameters_on_host();
        std::string value = "[";
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            const auto& instance = rlt::get_ref(impl->host(), parameters, instance_i);
            value += instance_i > 0 ? ", " : "";
            if constexpr (WORLD::N_AGENTS > 1){
                value += "{\"agents\": [";
            }
            for(TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++){
                const auto& agent = instance.agents[agent_i];
                value += (agent_i > 0 ? ", " : "") + parameters_to_json(*impl, agent.dynamics, agent.cameras, agent.drone);
            }
            if constexpr (WORLD::N_AGENTS > 1){
                value += "]}";
            }
        }
        value += "]";
        return value;
    }
    void Environment::set_instance_parameters(const bool* mask, const std::string& parameters_json_array){
        using rlt::rotor_wheelbase;
        impl->require_sampled("set_instance_parameters");
        const CurrentDevice current(impl->device_id);
        auto& parameters = impl->parameters_on_host();
        const nlohmann::json entries = nlohmann::json::parse(parameters_json_array);
        std::vector<typename WORLD::Parameters> updated;
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            if(mask[instance_i]){
                updated.push_back(rlt::get_ref(impl->host(), parameters, instance_i));
                const auto& entry = entries.at(updated.size() - 1);
                if constexpr (WORLD::N_AGENTS > 1){
                    if(!entry.at("agents").is_array() || entry.at("agents").size() != WORLD::N_AGENTS){
                        throw std::invalid_argument("hyperdrone: instance parameters need one entry per agent");
                    }
                }
                for(TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++){
                    const auto& agent_entry = WORLD::N_AGENTS == 1 ? entry : entry.at("agents").at(agent_i);
                    auto& agent = updated.back().agents[agent_i];
                    parameters_from_json(*impl, agent_entry, agent.dynamics, agent.cameras);
                    if(agent.dynamics.integration.dt != impl->env.environments[0].dynamics.parameters.integration.dt){
                        throw std::invalid_argument("hyperdrone: every drone must use the same integration timestep");
                    }
                    if(agent_entry.contains("drone")){
                        const auto drone = agent_entry.at("drone").get<long long>();
                        if(drone < 0 || drone >= (long long)impl->env.shared.num_drones){
                            throw std::invalid_argument("hyperdrone: drone must index the drone table");
                        }
                        agent.drone = (TI)drone;
                    }
                    if constexpr (WORLD::SPEC::SELF_VISIBLE){
                        const T wheelbase = rotor_wheelbase(impl->host(), agent.dynamics.dynamics.rotor_positions, (TI)WORLD::DYNAMICS_ENV::ACTION_DIM);
                        const float scale = (float)(wheelbase / impl->env.shared.drone_bodies[agent.drone].rig.wheelbase);
                        if(wheelbase <= 0 || !rlt::rendering::rig::detail::is_finite(scale) || scale <= 0){
                            throw std::invalid_argument("hyperdrone: rotor_positions must define a finite positive wheelbase and visual scale");
                        }
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
        for(TI environment_i = 0; environment_i < NUM_ENVIRONMENTS; environment_i++){
            request_render(impl->device, impl->env.environments[environment_i]);
        }
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
        impl->sampled.fill(false);
        impl->all_sampled = false;
    }

    std::string Environment::spawn_points_json(){
        impl->require_sampled("spawn_points");
        const CurrentDevice current(impl->device_id);
        const auto& parameters = impl->parameters_on_host();
        const auto& states = impl->on_host(impl->states, impl->states_host);
        nlohmann::json entries = nlohmann::json::array();
        for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
            const auto& world = impl->env.environments[instance_i / WORLD::INSTANCES];
            const auto& instance = rlt::get_ref(impl->host(), parameters, instance_i);
            nlohmann::json poses = nlohmann::json::array();
            for(TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++){
                const auto& state = rlt::rl::environments::hyperdrone::_agent_state<typename WORLD::SPEC>(rlt::get_ref(impl->host(), states, instance_i), agent_i);
                poses.push_back({state.position[0], state.position[1], state.position[2], state.orientation[0], state.orientation[1], state.orientation[2], state.orientation[3]});
            }
            nlohmann::json entry = {{"scene", world.slots[world.active_slot].metadata.content_hash},
                                   {"scene_translation", instance.scene_translation}, {"scene_yaw", {instance.scene_yaw_cos, instance.scene_yaw_sin}},
                                   {"poses", poses}, {"shutter_fraction", instance.shutter_fraction}};
            entry["drone"] = nlohmann::json::array();
            for(const auto& agent : instance.agents){
                entry["drone"].push_back(agent.drone);
            }
            entries.push_back(entry);
        }
        return entries.dump();
    }

    void Environment::reset(const bool* mask, bool resident, const std::string& spawn_points){
        const CurrentDevice current(impl->device_id);
        std::array<bool, TOTAL> selected{};
        if(!impl->all_sampled || SynchronizedResets<WORLD>::VALUE){
            selected = impl->reset_selection(mask, resident);
        }
        struct Spawn {
            TI instance;
            std::array<TI, WORLD::N_AGENTS> drones;
            std::array<T, 3> translation;
            std::array<T, 2> yaw;
            std::array<std::array<T, 7>, WORLD::N_AGENTS> poses;
            T shutter;
        };
        std::vector<Spawn> spawns;
        if(!spawn_points.empty()){
            if(CUDA_WORLD && resident && mask != nullptr){
                throw std::invalid_argument("hyperdrone: spawn points take a host mask");
            }
            const auto entries = nlohmann::json::parse(spawn_points);
            for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
                if(mask == nullptr || mask[instance_i]){
                    const auto& entry = entries.at(spawns.size());
                    const auto& world = impl->env.environments[instance_i / WORLD::INSTANCES];
                    const auto& drone_entries = entry.at("drone");
                    if(!drone_entries.is_array() || drone_entries.size() != WORLD::N_AGENTS || !entry.at("poses").is_array() || entry.at("poses").size() != WORLD::N_AGENTS){
                        throw std::invalid_argument("hyperdrone: spawn drone and poses need one entry per agent");
                    }
                    std::array<TI, WORLD::N_AGENTS> drones;
                    for(TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++){
                        const auto& value = drone_entries.at(agent_i);
                        if(!value.is_number_integer()){
                            throw std::invalid_argument("hyperdrone: spawn needs one integer drone index per agent");
                        }
                        const auto drone = value.get<long long>();
                        if(drone < 0 || drone >= (long long)impl->env.shared.drones.size()){
                            throw std::invalid_argument("hyperdrone: spawn drone is outside the drones table");
                        }
                        drones[agent_i] = (TI)drone;
                    }
                    if(entry.at("scene") != world.slots[world.active_slot].metadata.content_hash){
                        throw std::invalid_argument("hyperdrone: spawn scene does not match the active scene");
                    }
                    for(const auto& pose : entry.at("poses")){
                        if(!pose.is_array() || pose.size() != 7){
                            throw std::invalid_argument("hyperdrone: each spawn pose needs seven values");
                        }
                    }
                    spawns.push_back({instance_i, drones, entry.at("scene_translation").get<std::array<T, 3>>(),
                                      entry.at("scene_yaw").get<std::array<T, 2>>(),
                                      entry.at("poses").get<std::array<std::array<T, 7>, WORLD::N_AGENTS>>(),
                                      entry.at("shutter_fraction").get<T>()});
                }
            }
            if(spawns.size() != entries.size()){
                throw std::invalid_argument("hyperdrone: provide one spawn per selected instance");
            }
        }
        const auto reset_mask = mask == nullptr ? impl->all_instances : impl->input(mask, resident, impl->mask_staging);
        rlt::sample_initial_parameters(impl->device, impl->env, impl->parameters, reset_mask, impl->rng);
        if(!spawns.empty()){
            // the entry's nominal before the state sampling, which reads the dynamics
            auto& parameters = impl->parameters_on_host();
            for(const auto& spawn : spawns){
                auto& instance = rlt::get_ref(impl->host(), parameters, spawn.instance);
                for(TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++){
                    auto& agent = instance.agents[agent_i];
                    const auto& nominal = impl->env.shared.drones[spawn.drones[agent_i]].parameters;
                    agent.drone = spawn.drones[agent_i];
                    agent.dynamics = nominal.dynamics;
                    for(TI camera_i = 0; camera_i < WORLD::N_CAMERAS_PER_AGENT; camera_i++){
                        agent.cameras[camera_i] = nominal.cameras[camera_i];
                    }
                }
                instance.brightness_scale = 1;
            }
            impl->publish_parameters_from_host();
        }
        rlt::sample_initial_state(impl->device, impl->env, impl->parameters, impl->states, reset_mask, impl->rng);
        if(!spawns.empty()){
            auto& parameters = impl->parameters_on_host();
            auto& states = impl->on_host(impl->states, impl->states_host);
            for(const auto& spawn : spawns){
                auto& instance = rlt::get_ref(impl->host(), parameters, spawn.instance);
                std::copy(spawn.translation.begin(), spawn.translation.end(), instance.scene_translation);
                instance.scene_yaw_cos = spawn.yaw[0];
                instance.scene_yaw_sin = spawn.yaw[1];
                instance.shutter_fraction = spawn.shutter;
                for(TI agent_i = 0; agent_i < WORLD::N_AGENTS; agent_i++){
                    auto& state = rlt::rl::environments::hyperdrone::_agent_state<typename WORLD::SPEC>(rlt::get_ref(impl->host(), states, spawn.instance), agent_i);
                    std::copy(spawn.poses[agent_i].begin(), spawn.poses[agent_i].begin() + 3, state.position);
                    std::copy(spawn.poses[agent_i].begin() + 3, spawn.poses[agent_i].end(), state.orientation);
                }
            }
            impl->publish_parameters_from_host();
            impl->publish(impl->states_host, impl->states);
            render_spawned_target_frames<WORLD>(impl->device, impl->env, impl->parameters, reset_mask);
        }
        if(!impl->all_sampled){
            for(TI instance_i = 0; instance_i < TOTAL; instance_i++){
                impl->sampled[instance_i] = impl->sampled[instance_i] || selected[instance_i];
            }
            impl->all_sampled = std::all_of(impl->sampled.begin(), impl->sampled.end(), [](bool value){ return value; });
        }
    }
    void Environment::step(const float* actions, bool resident){
        impl->require_sampled("step");
        const CurrentDevice current(impl->device_id);
        const auto action_tensor = impl->input(actions, resident, impl->actions_staging);
        rlt::step(impl->device, impl->env, impl->parameters, impl->states, action_tensor, impl->next_states, impl->rng);
        rlt::reward(impl->device, impl->env, impl->parameters, impl->states, action_tensor, impl->next_states, impl->rewards, impl->rng);
        rlt::copy(impl->device, impl->device, impl->next_states, impl->states);
        rlt::terminated(impl->device, impl->env, impl->parameters, impl->states, impl->terminated_flags, impl->rng);
    }
    template <bool ENABLED, typename VALUE, typename GET>
    void read_image_output(Environment::Impl& impl, VALUE* destination, bool resident, const char* name, GET get_buffer){
        impl.require_sampled(name);
        if constexpr (ENABLED){
            const CurrentDevice current(impl.device_id);
            for(TI environment_i = 0; environment_i < NUM_ENVIRONMENTS; environment_i++){
                auto& world = impl.env.environments[environment_i];
                if(world.render_pending){
                    auto parameters = view_range(impl.device, impl.parameters, environment_i * WORLD::INSTANCES, rlt::tensor::ViewSpec<0, WORLD::INSTANCES>{});
                    auto states = view_range(impl.device, impl.states, environment_i * WORLD::INSTANCES, rlt::tensor::ViewSpec<0, WORLD::INSTANCES>{});
                    render(impl.device, world, parameters, states, world.render_reset);
                }
                if(world.history_step == 0){
                    throw std::runtime_error("hyperdrone: reset and render before reading image outputs");
                }
                auto& renderer = get_renderer(impl.device, world);
                auto& buffer = get_buffer(impl.device, renderer);
                using BUFFER = std::remove_reference_t<decltype(buffer)>;
                rlt::Tensor<typename BUFFER::SPEC> alias;
                alias._data = destination + environment_i * BUFFER::SPEC::SIZE;
                if constexpr (CUDA_WORLD){
                    impl.copy_out(buffer, alias._data, resident);
                }
                else {
                    copy(renderer.device, impl.host(), buffer, alias);
                }
            }
        }
        else {
            throw std::runtime_error(std::string("hyperdrone: this environment has no ") + name + " output");
        }
    }

    void Environment::depth(float* out, bool resident){
        read_image_output<rro::contains<OUTPUTS, rro::Depth<>>::value>(*impl, out, resident, "depth",
            [](auto& device, auto& renderer) -> auto& { return output(device, renderer, rl_tools::rendering::raytracing::outputs::Depth<>{}); });
    }
    void Environment::segmentation(uint32_t* out, bool resident){
        read_image_output<rro::contains<OUTPUTS, SEGMENTATION_OUTPUT>::value>(*impl, out, resident, "segmentation",
            [](auto& device, auto& renderer) -> auto& { return output(device, renderer, SEGMENTATION_OUTPUT{}); });
    }
    void Environment::normals(float* out, bool resident){
        read_image_output<rro::contains<OUTPUTS, rro::Normals<>>::value>(*impl, out, resident, "normals",
            [](auto& device, auto& renderer) -> auto& { return output(device, renderer, rl_tools::rendering::raytracing::outputs::Normals<>{}); });
    }
    void Environment::flow(float* out, bool resident){
        read_image_output<rro::contains<OUTPUTS, rro::Flow<>>::value>(*impl, out, resident, "flow",
            [](auto& device, auto& renderer) -> auto& { return output(device, renderer, rl_tools::rendering::raytracing::outputs::Flow<>{}); });
    }
    void Environment::flow_status(uint32_t* out, bool resident){
        read_image_output<rro::contains<OUTPUTS, rro::FlowStatus<>>::value>(*impl, out, resident, "flow_status",
            [](auto& device, auto& renderer) -> auto& { return output(device, renderer, rl_tools::rendering::raytracing::outputs::FlowStatus<>{}); });
    }
    void Environment::observe(float* out, bool resident){
        impl->require_sampled("observe");
        const CurrentDevice current(impl->device_id);
        impl->output(out, resident, impl->observations_staging, [&](auto& observations){
            rlt::observe(impl->device, impl->env, impl->parameters, impl->states, typename WORLD::Observation{}, observations, impl->rng);
        });
    }
    void Environment::observe_privileged(float* out, bool resident){
        impl->require_sampled("observe_privileged");
        const CurrentDevice current(impl->device_id);
        impl->output(out, resident, impl->observations_privileged_staging, [&](auto& observations){
            rlt::observe(impl->device, impl->env, impl->parameters, impl->states, typename WORLD::ObservationPrivileged{}, observations, impl->rng);
        });
    }
    void Environment::observe_imu(float* out, bool resident){
        impl->require_sampled("observe_imu");
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
    void Environment::spy_open(){
#ifdef RL_TOOLS_ENABLE_HYPERDRONE_SPY
        const CurrentDevice current(impl->device_id);
        if(impl->spy_open){
            throw std::runtime_error("hyperdrone: this environment already has an open spy");
        }
        using rlt::malloc;
        using rlt::init;
        using rlt::free;
        try{
            malloc(impl->device, impl->spy);
            init(impl->device, impl->spy, impl->env);
            impl->spy_open = true;
        }
        catch(...){
            free(impl->device, impl->spy);
            throw;
        }
#else
        throw std::runtime_error("hyperdrone: the spy was not compiled; provide GLFW, desktop OpenGL and Dear ImGui sources, then reconfigure the HyperDrone JIT build (see the HyperDrone README)");
#endif
    }
    bool Environment::spy_update(){
#ifdef RL_TOOLS_ENABLE_HYPERDRONE_SPY
        const CurrentDevice current(impl->device_id);
        if(impl->spy_open){
            using rlt::update;
            if(update(impl->device, impl->spy, impl->env)){
                return true;
            }
            spy_close();
        }
#endif
        return false;
    }
    void Environment::spy_close(){
#ifdef RL_TOOLS_ENABLE_HYPERDRONE_SPY
        const CurrentDevice current(impl->device_id);
        if(impl->spy_open){
            using rlt::free;
            free(impl->device, impl->spy);
            impl->spy_open = false;
        }
#endif
    }
}
