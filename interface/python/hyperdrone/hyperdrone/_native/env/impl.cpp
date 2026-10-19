#if defined(HYPERDRONE_ENV_CUDA)
#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#endif
#include "environment.h"
#if HYPERDRONE_ENV_DEVICE_CUDA
#include <rl_tools/rl/environments/hyperdrone/controllers/raptor/operations_cuda.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cuda.h>
#if HYPERDRONE_ENV_TASK == 1
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cuda.h>
#endif
#if HYPERDRONE_ENV_TASK == 2
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cuda.h>
#endif
#endif
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
#include <cuda_runtime.h>
#include "../common/cuda_device.h"
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
#include <cstring>

namespace HYPERDRONE_MODULE_NAME {
    using rl_tools::view_range;
    using rl_tools::render;
    using rl_tools::output;
    using rl_tools::get_renderer;
    using rl_tools::copy;
    using rl_tools::request_render;
    using rl_tools::data;
    using rl_tools::render_sync;
    using rl_tools::probe_sync;
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
    using rl_tools::collision_stream;
#endif

#if HYPERDRONE_ENV_DEVICE_CUDA
    using DEVICE = rlt::devices::CUDA<rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, HOST_DEVICE>>;
    // one curand state per instance across all members (the composite partitions them per member)
    using RNG = rlt::devices::random::CUDA::ENGINE<rlt::devices::random::CUDA::Specification<TI, TOTAL>>;
#endif
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
    using CurrentDevice = hyperdrone::CurrentCudaDevice;
#else
    struct CurrentDevice {
        explicit CurrentDevice(int){}
    };
#endif

#if !HYPERDRONE_ENV_DEVICE_CUDA
    using DEVICE = HOST_DEVICE;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
#endif

    using DECLARED_OUTPUTS = typename WORLD::RENDERER_CONFIG::DECLARED_OUTPUTS;

    std::vector<hyperdrone::OutputMetadata> environment_output_metadata(){
        auto metadata = hyperdrone::output_metadata_list<typename WORLD::RENDERER_SPEC>(DECLARED_OUTPUTS{});
        for(auto& value: metadata){
            auto reshape = [](hyperdrone::OutputLayout& layout){
                const size_t views = layout.shape[0] / WORLD::INSTANCES;
                layout.shape[0] = views;
                layout.shape.insert(layout.shape.begin(), TOTAL);
                layout.axes[0] = "view";
                layout.axes.insert(layout.axes.begin(), "instance");
                const int64_t element_stride = layout.strides.back();
                hyperdrone::output_contiguous_strides(layout);
                for(auto& stride: layout.strides){
                    stride *= element_stride;
                }
            };
            reshape(value.layout);
            if(value.has_packed){
                reshape(value.packed);
            }
            for(auto& field: value.fields){
                reshape(field.second);
            }
            value.cameras.erase(std::lower_bound(value.cameras.begin(), value.cameras.end(), WORLD::N_VIEWS), value.cameras.end());
            value.block_count = NUM_WORLDS;
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            value.device_type = 2;
#endif
        }
        return metadata;
    }

    template <typename ELEMENT, TI... DIMS>
    using InstanceTensor = rlt::Tensor<rlt::tensor::Specification<ELEMENT, TI, rlt::tensor::Shape<TI, TOTAL, DIMS...>>>;

    struct Environment::Impl {
        ~Impl();
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
        std::array<bool, NUM_WORLDS> captured{};
        std::array<bool, NUM_WORLDS> probed{};
        std::vector<hyperdrone::OutputMetadata> outputs = environment_output_metadata();
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
        void require_current(TI member, bool probes, const char* verb) const {
            if(env.worlds[member].render_pending || !(probes ? probed[member] : captured[member])){
                throw std::runtime_error(std::string(probes ? "hyperdrone: render and probe before " : "hyperdrone: render or observe before ") + verb + " this output");
            }
        }
        void invalidate_pending_probes(){
            for(TI member = 0; member < NUM_WORLDS; member++){
                if(env.worlds[member].render_pending){
                    probed[member] = false;
                }
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
                for(TI world_i = 0; world_i < NUM_WORLDS; world_i++){
                    const auto first = selected.begin() + world_i * WORLD::INSTANCES;
                    const auto last = first + WORLD::INSTANCES;
                    if(std::any_of(first, last, [](bool value){ return value; }) && !std::all_of(first, last, [](bool value){ return value; })){
                        throw std::invalid_argument("hyperdrone: the reset mask must be all-or-none per World");
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
        const auto& world = impl.env.worlds[0];
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
        auto& world = impl.env.worlds[0];
        const auto& values = parameters.at("cameras");
        if(!values.is_array() || values.size() != WORLD::N_CAMERAS_PER_AGENT){
            throw std::invalid_argument("hyperdrone: provide one camera parameter dict per camera");
        }
        rlt::from_json(impl.host(), world.dynamics, parameters.at("dynamics"), dynamics);
        using rlt::validate;
        if(!validate(impl.host(), world.controller, dynamics.dynamics)){
            throw std::invalid_argument("hyperdrone: controller requires canonical Crazyflie/RAPTOR motor allocation");
        }
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
                rlt::rl::environments::hyperdrone::rig::Rotorcraft rotorcraft;
                const auto& rotors = drone.parameters.dynamics.dynamics;
                if(!rlt::init(impl.host(), rotorcraft, assembly, rotors.rotor_positions, rotors.rotor_torque_directions, (TI)WORLD::DYNAMICS_ENV::ACTION_DIM)){
                    throw std::invalid_argument(name + ": the asset needs a body and one prop_* per rotor, each hub nearest a distinct rotor with a nonzero reaction torque");
                }
                if(assembly.parts.size() > rlt::rl::environments::hyperdrone::MAX_DRONE_PARTS){
                    throw std::invalid_argument(name + ": a drone asset has at most " + std::to_string(rlt::rl::environments::hyperdrone::MAX_DRONE_PARTS) + " parts");
                }
                if(drone_i > 0 && assembly.parts.size() != parts){
                    throw std::invalid_argument(name + ": all drone assets need the same part count, the body swaps in place at reset");
                }
                parts = assembly.parts.size();
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
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
        cudaGetDevice(&impl->device_id);
#endif
        for(auto& metadata: impl->outputs){
            metadata.device_id = impl->device_id;
        }
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
            if constexpr (WORLD::HAS_IMAGE_OBSERVATION){
                rlt::malloc(impl->device, impl->observations_staging);
            }
            rlt::malloc(impl->device, impl->observations_privileged_staging);
            rlt::malloc(impl->device, impl->observations_imu_staging);
        }
    }
    Environment::~Environment(){
        spy_close();
    }
    Environment::Impl::~Impl(){
        auto* impl = this;
        const CurrentDevice current(impl->device_id);
        if constexpr (CUDA_WORLD){
            rlt::free(impl->device, impl->mask_staging);
            rlt::free(impl->device, impl->actions_staging);
            if constexpr (WORLD::HAS_IMAGE_OBSERVATION){
                rlt::free(impl->device, impl->observations_staging);
            }
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
            for(TI world_i = 0; world_i < NUM_WORLDS; world_i++){
                if(!rlt::set_asset(impl->device, impl->env.worlds[world_i], asset.first, asset.second)){
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
        const auto& world = impl->env.worlds[0];
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
                    if(agent.dynamics.integration.dt != impl->env.worlds[0].dynamics.parameters.integration.dt){
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
                        const float scale = (float)(wheelbase / impl->env.shared.drone_bodies[agent.drone].rotorcraft.wheelbase);
                        if(wheelbase <= 0 || !rlt::rendering::is_finite(scale) || scale <= 0){
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
        const auto reset_mask = impl->input(mask, !CUDA_WORLD, impl->mask_staging);
        using rlt::reset_controller;
        reset_controller(impl->device, impl->env, reset_mask, impl->rng);
        for(TI world_i = 0; world_i < NUM_WORLDS; world_i++){
            request_render(impl->device, impl->env.worlds[world_i]);
        }
    }
    std::optional<std::string> Environment::model(const std::string& name){
        DYNAMICS dynamics;
        if(!hyperdrone::model_by_name<typename WORLD::SPEC>(name, dynamics)){
            return std::nullopt;
        }
        return rlt::json(impl->host(), impl->env.worlds[0].dynamics, dynamics);
    }
    void Environment::rotate_scene(){
        impl->captured.fill(false);
        impl->probed.fill(false);
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
            const auto& world = impl->env.worlds[instance_i / WORLD::INSTANCES];
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
                    const auto& world = impl->env.worlds[instance_i / WORLD::INSTANCES];
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
        impl->captured.fill(false);
        impl->probed.fill(false);
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
        impl->invalidate_pending_probes();
        rlt::step(impl->device, impl->env, impl->parameters, impl->states, action_tensor, impl->next_states, impl->rng);
        rlt::reward(impl->device, impl->env, impl->parameters, impl->states, action_tensor, impl->next_states, impl->rewards, impl->rng);
        rlt::copy(impl->device, impl->device, impl->next_states, impl->states);
        rlt::terminated(impl->device, impl->env, impl->parameters, impl->states, impl->terminated_flags, impl->rng);
    }
    void Environment::render(){
        using rlt::render;
        impl->require_sampled("render");
        const CurrentDevice current(impl->device_id);
        impl->invalidate_pending_probes();
        for(TI member = 0; member < NUM_WORLDS; member++){
            auto& world = impl->env.worlds[member];
            if(world.render_pending){
                auto parameters = view_range(impl->device, impl->parameters, member * WORLD::INSTANCES, rlt::tensor::ViewSpec<0, WORLD::INSTANCES>{});
                auto states = view_range(impl->device, impl->states, member * WORLD::INSTANCES, rlt::tensor::ViewSpec<0, WORLD::INSTANCES>{});
                render(impl->device, world, parameters, states, world.render_reset);
            }
            impl->captured[member] = true;
        }
    }

    void Environment::probe(){
        using rlt::probe;
        if constexpr (WORLD::RENDERER_CONFIG::INTERNAL_PROBES){
            throw std::runtime_error("hyperdrone: no public probe output is configured");
        }
        impl->require_sampled("probe");
        const CurrentDevice current(impl->device_id);
        for(TI member = 0; member < NUM_WORLDS; member++){
            if(impl->env.worlds[member].render_pending || !impl->captured[member]){
                throw std::runtime_error("hyperdrone: render the camera poses before probing");
            }
        }
        for(TI member = 0; member < NUM_WORLDS; member++){
            auto& renderer = get_renderer(impl->device, impl->env.worlds[member]);
            probe(impl->device, renderer);
            impl->probed[member] = true;
        }
    }

    const std::vector<hyperdrone::OutputMetadata>& Environment::output_metadata() const {
        return impl->outputs;
    }

    void Environment::output_read(size_t index, void* destination, int destination_device, const std::string& field, uintptr_t consumer_stream){
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
        using rlt::stream;
#endif
        const auto& metadata = impl->outputs.at(index);
        const auto& layout = hyperdrone::output_layout(metadata, field);
        const size_t count = layout.size() / NUM_WORLDS;
        const size_t bytes = count * layout.item_size;
        const CurrentDevice current(impl->device_id);
        hyperdrone::visit_output(index, [&](auto tag){
            using OUTPUT = typename decltype(tag)::TYPE;
            constexpr bool PROBES = std::is_same<OUTPUT, rro::Probes>::value;
            for(TI member = 0; member < NUM_WORLDS; member++){
                impl->require_current(member, PROBES, "reading");
            }
            for(TI member = 0; member < NUM_WORLDS; member++){
                auto& renderer = get_renderer(impl->device, impl->env.worlds[member]);
                auto& source = output(impl->device, renderer, OUTPUT{});
                auto* target = static_cast<unsigned char*>(destination) + member * bytes;
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
                const auto producer = PROBES ? collision_stream(impl->device, renderer) : stream(impl->device, renderer);
                const auto consumer = reinterpret_cast<cudaStream_t>(consumer_stream);
                if(destination_device == 2){
                    rlt::devices::cuda::order_after(renderer.backend->ordering_event, consumer, producer);
                }
                const auto kind = destination_device == 2 ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
                cudaError_t status;
                if constexpr (PROBES){
                    status = cudaMemcpy2DAsync(target, layout.item_size, reinterpret_cast<const unsigned char*>(data(source)) + layout.offset, sizeof(rlt::rendering::CollisionResult), layout.item_size, count, kind, producer);
                }
                else {
                    status = cudaMemcpyAsync(target, data(source), bytes, kind, producer);
                }
                if(status != cudaSuccess){
                    throw std::runtime_error(std::string("hyperdrone: output transfer failed: ") + cudaGetErrorString(status));
                }
                if(destination_device == 2){
                    rlt::devices::cuda::order_after(renderer.backend->ordering_event, producer, consumer);
                }
                else if(cudaStreamSynchronize(producer) != cudaSuccess){
                    throw std::runtime_error("hyperdrone: output transfer synchronization failed");
                }
#else
                if(destination_device != 1){
                    throw std::invalid_argument("hyperdrone: CUDA output destinations require the OptiX renderer");
                }
                if constexpr (PROBES){
                    probe_sync(impl->device, renderer);
                    const auto* values = reinterpret_cast<const unsigned char*>(data(source)) + layout.offset;
                    for(size_t item = 0; item < count; item++){
                        std::memcpy(target + item * layout.item_size, values + item * sizeof(rlt::rendering::CollisionResult), layout.item_size);
                    }
                }
                else {
                    render_sync(impl->device, renderer);
                    std::memcpy(target, data(source), bytes);
                }
#endif
            }
        }, DECLARED_OUTPUTS{});
    }

    OutputBlock Environment::output_block(size_t index, size_t member, const std::string& field){
        if(member >= NUM_WORLDS){
            throw std::out_of_range("hyperdrone: output block index out of range");
        }
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
        using rlt::stream;
#endif
        const auto& metadata = impl->outputs.at(index);
        OutputBlock result;
        result.layout = hyperdrone::output_layout(metadata, field);
        result.layout.shape[0] = WORLD::INSTANCES;
        result.device_type = metadata.device_type;
        result.device_id = metadata.device_id;
        result.owner = impl;
        const CurrentDevice current(impl->device_id);
        hyperdrone::visit_output(index, [&](auto tag){
            using OUTPUT = typename decltype(tag)::TYPE;
            constexpr bool PROBES = std::is_same<OUTPUT, rro::Probes>::value;
            impl->require_current(member, PROBES, "exporting");
            auto& renderer = get_renderer(impl->device, impl->env.worlds[member]);
            auto& source = output(impl->device, renderer, OUTPUT{});
            result.pointer = reinterpret_cast<unsigned char*>(data(source)) + result.layout.offset;
#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
            const auto producer = PROBES ? collision_stream(impl->device, renderer) : stream(impl->device, renderer);
            const auto event = renderer.backend->ordering_event;
            const int device_id = impl->device_id;
            result.order_consumer = [producer, event, device_id](uintptr_t consumer){
                const CurrentDevice current(device_id);
                rlt::devices::cuda::order_after(event, producer, reinterpret_cast<cudaStream_t>(consumer));
            };
#else
            if constexpr (PROBES){
                probe_sync(impl->device, renderer);
            }
            else {
                render_sync(impl->device, renderer);
            }
#endif
        }, DECLARED_OUTPUTS{});
        return result;
    }

    template <typename WORLD_TYPE>
    void observe_into(Environment::Impl& impl, float* out, bool resident){
        if constexpr (WORLD_TYPE::HAS_IMAGE_OBSERVATION){
            impl.invalidate_pending_probes();
            impl.output(out, resident, impl.observations_staging, [&](auto& observations){
                using rlt::observe;
                observe(impl.device, impl.env, impl.parameters, impl.states, typename WORLD_TYPE::Observation{}, observations, impl.rng);
            });
            impl.captured.fill(true);
        }
        else {
            throw std::runtime_error("hyperdrone: this environment has no policy observation");
        }
    }
    void Environment::observe(float* out, bool resident){
        impl->require_sampled("observe");
        const CurrentDevice current(impl->device_id);
        observe_into<WORLD>(*impl, out, resident);
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
        const CurrentDevice current(impl->device_id);
#if HYPERDRONE_ENV_DEVICE_CUDA
        cudaStreamSynchronize(impl->device.stream);
#endif
        for(TI member = 0; member < NUM_WORLDS; member++){
            auto& renderer = get_renderer(impl->device, impl->env.worlds[member]);
            render_sync(impl->device, renderer);
            probe_sync(impl->device, renderer);
        }
    }
#ifdef RL_TOOLS_ENABLE_HYPERDRONE_SPY
    template <typename DEVICE, typename CONFIG>
    void allocate_spy(DEVICE& device, rlt::rl::environments::hyperdrone::Spy<CONFIG>& spy, typename CONFIG::ENVIRONMENT& environment){
        if constexpr (!CONFIG::WORLD::RENDERER_SPEC::HAS_RGB){
            throw std::runtime_error("hyperdrone: spy requires a camera with an RGB output");
        }
        else{
            using rlt::malloc;
            using rlt::init;
            using rlt::free;
            try{
                malloc(device, spy);
                init(device, spy, environment);
            }
            catch(...){
                free(device, spy);
                throw;
            }
        }
    }
#endif
    void Environment::spy_open(){
#ifdef RL_TOOLS_ENABLE_HYPERDRONE_SPY
        const CurrentDevice current(impl->device_id);
        if(impl->spy_open){
            throw std::runtime_error("hyperdrone: this environment already has an open spy");
        }
        allocate_spy(impl->device, impl->spy, impl->env);
        impl->spy_open = true;
#else
        throw std::runtime_error("hyperdrone: the spy was not compiled; provide GLFW, desktop OpenGL and git (to fetch Dear ImGui), then reconfigure the HyperDrone JIT build (see the HyperDrone README)");
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
