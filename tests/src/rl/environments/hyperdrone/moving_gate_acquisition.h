#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace test_moving_gate_acquisition {
    using namespace rl_tools;
    using Snapshot = std::map<std::string, std::vector<unsigned char>>;

    template <typename BASE_SPEC, typename BASE_SPEC::TI HISTORY, bool FLOW>
    struct Specification: BASE_SPEC {
        using TI = typename BASE_SPEC::TI;
        static constexpr TI INSTANCES_PER_WORLD = 2;
        static constexpr TI CAM_WIDTH = 17;
        static constexpr TI CAM_HEIGHT = 13;
        static constexpr TI HISTORY_LENGTH = HISTORY;
        static constexpr bool SELF_VISIBLE = false;
        using CAMERA_OUTPUTS = rl::environments::hyperdrone::CameraOutputs<typename rl_tools::hyperlux::outputs::Concat<
            rl_tools::hyperlux::outputs::List<rl_tools::hyperlux::outputs::RGB<rl_tools::hyperlux::outputs::format::Float32>>,
            rl_tools::hyperlux::outputs::List<rl_tools::hyperlux::outputs::RGB<>>,
            rl_tools::hyperlux::outputs::List<rl_tools::hyperlux::outputs::Depth<>>,
            rl_tools::hyperlux::outputs::List<rl_tools::hyperlux::outputs::Segmentation<rl_tools::hyperlux::outputs::format::UInt32, rl_tools::hyperlux::outputs::labels::Instance>>,
            rl_tools::hyperlux::outputs::List<rl_tools::hyperlux::outputs::Normals<>>,
            rl_tools::hyperlux::outputs::If<(FLOW), rl_tools::hyperlux::outputs::Flow<>>,
            rl_tools::hyperlux::outputs::If<(FLOW), rl_tools::hyperlux::outputs::FlowStatus<>>,
            rl_tools::hyperlux::outputs::List<rl_tools::hyperlux::outputs::Probes>
        >::type>;
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr TI MOTION_BLUR_SAMPLES = 4;
    };

    template <typename HOST, typename DEVICE, typename SPEC>
    auto append_tensor(HOST& host, DEVICE& device, const Tensor<SPEC>& source, std::vector<unsigned char>& snapshot){
        std::vector<typename SPEC::T> values(SPEC::SIZE);
        Tensor<SPEC> alias;
        alias._data = values.data();
        copy(device, host, source, alias);
        const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
        snapshot.insert(snapshot.end(), bytes, bytes + SPEC::SIZE_BYTES);
        return values;
    }

    template <typename BASE_SPEC, typename BASE_SPEC::TI HISTORY, bool FLOW, typename DEVICE, typename HOST>
    auto rollout(DEVICE& device, HOST& host, const std::string& scene_path, const std::string& asset_path, bool eager){
        using T = typename BASE_SPEC::T;
        using TI = typename BASE_SPEC::TI;
        using BASE_WORLD = rl::environments::hyperdrone::World<Specification<BASE_SPEC, HISTORY, FLOW>>;
        using TASK_SPEC = rl::environments::hyperdrone::tasks::moving_gate::Specification<BASE_WORLD>;
        using WORLD = rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;
        using RNG = typename DEVICE::SPEC::RANDOM::template ENGINE<>;
        constexpr TI COUNT = WORLD::INSTANCES;
        using PARAMETERS_SPEC = tensor::Specification<typename WORLD::Parameters, TI, tensor::Shape<TI, COUNT>>;
        using STATES_SPEC = tensor::Specification<typename WORLD::State, TI, tensor::Shape<TI, COUNT>>;
        Tensor<PARAMETERS_SPEC> parameters, host_parameters;
        Tensor<STATES_SPEC> states, next_states, host_states;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, COUNT>>> reset_mask;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, COUNT, WORLD::ACTION_DIM>>> actions;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, COUNT, BASE_WORLD::OBSERVATION_DIM>>> observations;
        WORLD world;
        typename BASE_WORLD::SharedContext shared;
        RNG rng;
        malloc(device, rng);
        init(device, rng, 1927);
        malloc(host, shared.library);
        malloc(device, world);
        world.gate_asset_path = asset_path;
        hyperlux::datasets::procthor::GLB dataset{{}, {scene_path}};
        typename decltype(dataset)::Corpus corpus;
        hyperlux::datasets::procthor::enumerate(host, dataset, corpus);
        rl_tools::init(device, world, shared, dataset, corpus, 0, 1, 0);
        malloc(device, parameters);
        malloc(device, states);
        malloc(device, next_states);
        malloc(device, reset_mask);
        malloc(device, actions);
        malloc(device, observations);
        malloc(host, host_parameters);
        malloc(host, host_states);
        set_all(device, actions, T(0));

        auto reset = [&](bool all){
            set_all(device, reset_mask, all);
            if(!all){
                auto first = view_range(device, reset_mask, TI(0), tensor::ViewSpec<0, 1>{});
                set_all(device, first, true);
            }
            sample_initial_parameters(device, world, parameters, reset_mask, rng);
            sample_initial_state(device, world, parameters, states, reset_mask, rng);
            copy(device, host, parameters, host_parameters);
            copy(device, host, states, host_states);
            for(TI row = 0; row < (all ? COUNT : TI(1)); row++){
                auto& parameter = get_ref(host, host_parameters, row);
                auto& state = get_ref(host, host_states, row);
                parameter.scene_yaw_cos = 1;
                parameter.scene_yaw_sin = 0;
                parameter.shutter_fraction = T(0.75);
                parameter.brightness_scale = 1;
                parameter.gate_amplitude = T(0.35);
                parameter.gate_frequency = 1;
                parameter.agents[0].cameras[0] = rl::environments::hyperdrone::Camera<T>{};
                for(TI axis = 0; axis < 3; axis++){
                    parameter.scene_translation[axis] = T(1000 + row * 10);
                    parameter.gate_center[axis] = parameter.scene_translation[axis] + (axis == 0 ? T(2) : T(0));
                    parameter.gate_axis[axis] = axis == 1 ? T(1) : T(0);
                    state.position[axis] = 0;
                    state.linear_velocity[axis] = 0;
                    state.angular_velocity[axis] = 0;
                }
                for(TI element = 0; element < 4; element++){
                    parameter.gate_orientation[element] = element == 0 ? T(1) : T(0);
                    state.orientation[element] = element == 0 ? T(1) : T(0);
                }
                state.gate_phase = T(0.2) * T(row);
            }
            copy(host, device, host_parameters, parameters);
            copy(host, device, host_states, states);
            set_all(device, reset_mask, false);
        };
        reset(true);
        observe(device, world, parameters, states, typename BASE_WORLD::Observation{}, observations, rng);
        auto retained_image = [&](){
            std::vector<unsigned char> snapshot;
            auto& renderer = get_renderer(device, world);
            append_tensor(host, renderer.device, output(device, renderer, rl_tools::hyperlux::outputs::RGB<>{}), snapshot);
            append_tensor(host, device, world.history, snapshot);
            return snapshot;
        };
        std::vector<unsigned char> retained;
        if constexpr (HISTORY == 1){
            retained = retained_image();
        }
        std::vector<Snapshot> checkpoints;
        for(TI index = 0; index < 8; index++){
            if(index == 3){
                reset(false);
            }
            if(eager){
                observe(device, world, parameters, states, typename BASE_WORLD::Observation{}, observations, rng);
            }
            step(device, world, parameters, states, actions, next_states, rng);
            copy(device, device, next_states, states);
            if constexpr (HISTORY == 1){
                if(!eager){
                    EXPECT_EQ(retained_image(), retained) << "step must leave acquired images unchanged";
                }
            }
            const bool checkpoint = index == 2 || index == 4 || index == 7;
            if(eager || checkpoint){
                observe(device, world, parameters, states, typename BASE_WORLD::Observation{}, observations, rng);
            }
            if(checkpoint){
                auto& renderer = get_renderer(device, world);
                checkpoints.emplace_back();
                auto& snapshot = checkpoints.back();
                auto append_poses = [&](auto& poses, const char* name){
                    for(TI overlay = 0; overlay < WORLD::RENDERER_SPEC::NUM_OVERLAYS; overlay++){
                        auto rows = view(device, poses, overlay);
                        for(TI slot = 0; slot < WORLD::RENDERER_SPEC::MAX_OVERLAY_INSTANCES; slot++){
                            if(renderer.overlays[overlay].slots[slot].active){
                                auto pose = view(device, rows, slot);
                                append_tensor(host, renderer.device, pose, snapshot[name]);
                            }
                        }
                    }
                };
                append_tensor(host, renderer.device, output(device, renderer, rl_tools::hyperlux::outputs::RGB<>{}), snapshot["rgba"]);
                append_tensor(host, renderer.device, output(device, renderer, rl_tools::hyperlux::outputs::Depth<>{}), snapshot["depth"]);
                const auto ids = append_tensor(host, renderer.device, output(device, renderer, rl_tools::hyperlux::outputs::Segmentation<>{}), snapshot["segmentation"]);
                append_tensor(host, renderer.device, output(device, renderer, rl_tools::hyperlux::outputs::Normals<>{}), snapshot["normals"]);
                append_poses(renderer.transforms, "transforms");
                append_tensor(host, device, observations, snapshot["observation"]);
                append_tensor(host, device, world.history, snapshot["history"]);
                const auto episode_start = append_tensor(host, device, world.episode_start, snapshot["episode_start"]);
                EXPECT_EQ(episode_start[0], index < 3 ? 0 : 3);
                EXPECT_EQ(episode_start[1], 0);
                EXPECT_FALSE(world.render_pending);
                if constexpr (FLOW){
                    const auto flow = append_tensor(host, renderer.device, output(device, renderer, rl_tools::hyperlux::outputs::Flow<>{}), snapshot["flow"]);
                    const auto status = append_tensor(host, renderer.device, output(device, renderer, rl_tools::hyperlux::outputs::FlowStatus<>{}), snapshot["flow_status"]);
                    append_poses(renderer.transforms_previous, "transforms_previous");
                    EXPECT_EQ(world.flow_scene_version, world.scene_version);
                    bool moving_gate_flow = false;
                    const size_t scene_instances = shared.library.scenes[world.slots[world.active_slot].scene_id].instances.size();
                    for(size_t pixel = 0; pixel < ids.size(); pixel++){
                        if(ids[pixel] != uint32_t(-1) && ids[pixel] >= scene_instances && status[pixel] == hyperlux::FlowStatus::VALID){
                            moving_gate_flow = moving_gate_flow || std::abs(flow[pixel * 2]) + std::abs(flow[pixel * 2 + 1]) > 1e-4f;
                        }
                    }
                    EXPECT_TRUE(moving_gate_flow) << "history " << HISTORY << " step " << index;
                }
                if constexpr (HISTORY == 1){
                    retained = retained_image();
                }
            }
        }
        free(host, host_states);
        free(host, host_parameters);
        free(device, observations);
        free(device, actions);
        free(device, reset_mask);
        free(device, next_states);
        free(device, states);
        free(device, parameters);
        rl_tools::free(device, world);
        free(host, shared.library);
        free(device, rng);
        return checkpoints;
    }

    template <typename BASE_SPEC, typename BASE_SPEC::TI HISTORY, bool FLOW, typename DEVICE, typename HOST>
    void compare(DEVICE& device, HOST& host, const std::string& scene_path, const std::string& asset_path){
        const auto eager = rollout<BASE_SPEC, HISTORY, FLOW>(device, host, scene_path, asset_path, true);
        const auto deferred = rollout<BASE_SPEC, HISTORY, FLOW>(device, host, scene_path, asset_path, false);
        ASSERT_EQ(eager.size(), deferred.size());
        for(size_t checkpoint = 0; checkpoint < eager.size(); checkpoint++){
            for(const auto& field : eager[checkpoint]){
                const auto& expected = field.second;
                const auto& actual = deferred[checkpoint].at(field.first);
                ASSERT_EQ(expected.size(), actual.size());
                size_t first_difference = 0;
                while(first_difference < expected.size() && expected[first_difference] == actual[first_difference]){
                    first_difference++;
                }
                EXPECT_EQ(first_difference, expected.size()) << "history " << HISTORY << " flow " << FLOW << " checkpoint " << checkpoint << " output " << field.first;
            }
        }
    }

    template <typename BASE_SPEC, typename DEVICE, typename HOST>
    void check(DEVICE& device, HOST& host, const std::string& scene_path, const std::string& asset_path){
        compare<BASE_SPEC, 1, true>(device, host, scene_path, asset_path);
        compare<BASE_SPEC, 3, true>(device, host, scene_path, asset_path);
        compare<BASE_SPEC, 1, false>(device, host, scene_path, asset_path);
    }
}
