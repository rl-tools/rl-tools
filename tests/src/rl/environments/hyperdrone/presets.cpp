#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <string>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
static const std::string DRONE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/x500.glb";
#else
static const std::string SCENE_PATH = "";
static const std::string DRONE_PATH = "";
#endif

namespace test_presets {
    struct WORLD_SPEC: rlt::rl::environments::hyperdrone::presets::X500FPV<T, TI> {
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 2;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        using SHADING = rlt::rendering::raytracing::Low;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    static_assert(WORLD_SPEC::SELF_VISIBLE);
    static_assert(rlt::rl::environments::hyperdrone::world::HasRotorPhase<typename WORLD::State>::VALUE);
    static_assert(WORLD::ACTION_DIM == 4);
}

using namespace test_presets;

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_PRESETS, X500FPV_ROLLOUT){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::init(device);
    WORLD world;
    typename WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    world.drone_asset_path = DRONE_PATH;
    shared.scene_set.paths = {SCENE_PATH};
    rlt::init(device, world, shared, 0, 1, 0);
    ASSERT_EQ(world.entity_kinds.size(), 1);
    EXPECT_GT(world.drone_rig.num_props, 0);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, WORLD::INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, WORLD::INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, WORLD::INSTANCES>>> reset_mask, terminated_flags;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, WORLD::INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, WORLD::INSTANCES, WORLD::OBSERVATION_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, WORLD::INSTANCES>>> rewards;
    rlt::malloc(device, parameters);
    rlt::malloc(device, states);
    rlt::malloc(device, next_states);
    rlt::malloc(device, reset_mask);
    rlt::malloc(device, terminated_flags);
    rlt::malloc(device, actions);
    rlt::malloc(device, observations);
    rlt::malloc(device, rewards);

    rlt::set_all(device, reset_mask, true);
    rlt::set_all(device, actions, (T)0);
    rlt::sample_initial_parameters(device, world, parameters, reset_mask, rng);
    rlt::sample_initial_state(device, world, parameters, states, reset_mask, rng);
    for(TI step_i = 0; step_i < 3; step_i++){
        rlt::render(device, world, parameters, states, reset_mask);
        rlt::observe(device, world, parameters, states, typename WORLD::Observation{}, observations, rng);
        rlt::step(device, world, parameters, states, actions, next_states, rng);
        rlt::reward(device, world, parameters, states, actions, next_states, rewards, rng);
        rlt::copy(device, device, next_states, states);
        rlt::terminated(device, world, parameters, states, terminated_flags, rng);
        rlt::set_all(device, reset_mask, false);
    }
    // rotor phase advances with rpm through the step (the preset carries the render state)
    typename WORLD::State state = rlt::get(device, states, (TI)0);
    bool phase_advanced = false;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        phase_advanced = phase_advanced || state.rotor_phase[rotor_i] != 0;
    }
    EXPECT_TRUE(phase_advanced) << "the preset's StateRenderRotorPhase should integrate rpm";
    T observation_sum = 0;
    for(TI dim = 0; dim < WORLD::OBSERVATION_DIM; dim++){
        observation_sum += rlt::get(device, observations, (TI)0, dim);
    }
    EXPECT_GT(observation_sum, (T)0) << "the rendered observation should not be blank";

    rlt::free(device, parameters);
    rlt::free(device, states);
    rlt::free(device, next_states);
    rlt::free(device, reset_mask);
    rlt::free(device, terminated_flags);
    rlt::free(device, actions);
    rlt::free(device, observations);
    rlt::free(device, rewards);
    rlt::free(device, rng);
    rlt::free(device, world);
    rlt::free(device, shared.library);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
