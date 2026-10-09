#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cuda.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cuda.h>

#include "../../../utils/utils.h"

#include "drones_test.h"

using DEVICE_GPU_SPEC = rlt::rendering::raytracing::device::Specification<rlt::devices::DefaultCUDASpecification, DEVICE>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<DEVICE_GPU_SPEC>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;

using namespace test_drones;

static bool cuda_available(){
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES_CUDA_DeathTest, REJECTS_EMPTY_CATALOG){
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_DEATH({
        DEVICE_GPU device;
        rlt::init(device.rendering);
        WORLD::SharedContext shared;
        rlt::init(device, shared);
    }, ".*");
}

// the device bank: setups draw the drone entry, resetting instances adopt the setup's entry and
// the others keep theirs
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES_CUDA, TARGET_FRAME_BANK_CARRIES_THE_DRONE){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    if(!cuda_available()){
        GTEST_SKIP() << "no CUDA device available";
    }
    using TASK_WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<WORLD>>;
    DEVICE_GPU device_gpu;
    auto& device = device_gpu.rendering;
    rlt::init(device);
    rlt::init(device_gpu);
    TASK_WORLD world;
    TASK_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device_gpu, world);
    const auto table = one_asset_per_dynamics();
    shared.drones = table.drones;
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device_gpu, world, shared, dataset, corpus, 0, 1, 0);
    InstanceTensor<TASK_WORLD::Parameters> parameters, parameters_gpu, setups;
    InstanceTensor<TASK_WORLD::State> states_gpu;
    InstanceTensor<TI> setup_index;
    MASK_TENSOR mask, mask_gpu;
    RNG_GPU rng;
    rlt::malloc(device, parameters);
    rlt::malloc(device, setups);
    rlt::malloc(device, setup_index);
    rlt::malloc(device, mask);
    rlt::malloc(device_gpu, parameters_gpu);
    rlt::malloc(device_gpu, states_gpu);
    rlt::malloc(device_gpu, mask_gpu);
    rlt::malloc(device_gpu, rng);
    rlt::init(device_gpu, rng, 912);
    rlt::set_all(device_gpu, mask_gpu, true);
    rlt::sample_initial_parameters(device_gpu, world, parameters_gpu, mask_gpu, rng);
    rlt::sample_initial_state(device_gpu, world, parameters_gpu, states_gpu, mask_gpu, rng);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        rlt::set(device, mask, instance_i % 2 != 0, instance_i);
    }
    rlt::copy(device, device_gpu, mask, mask_gpu);
    std::vector<TI> counts(table.drones.size(), 0);
    for(TI round_i = 0; round_i < 16; round_i++){
        rlt::select_scene(device_gpu, world, 0);
        rlt::copy(device_gpu, device, parameters_gpu, parameters);
        std::vector<TASK_WORLD::Parameters> before(INSTANCES);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            before[instance_i] = rlt::get(device, parameters, instance_i);
        }
        rlt::sample_initial_parameters(device_gpu, world, parameters_gpu, mask_gpu, rng);
        rlt::sample_initial_state(device_gpu, world, parameters_gpu, states_gpu, mask_gpu, rng);
        rlt::render(device_gpu, world, parameters_gpu, states_gpu, mask_gpu);
        cudaDeviceSynchronize();
        rlt::copy(device_gpu, device, parameters_gpu, parameters);
        rlt::copy(device_gpu, device, world.setups, setups);
        rlt::copy(device_gpu, device, world.setup_index, setup_index);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const auto& setup = rlt::get_ref(device, setups, instance_i);
            ASSERT_LT(setup.drone, table.drones.size());
            EXPECT_EQ(setup.camera.fov, table.drones[setup.drone].parameters.camera.fov);
            const auto& actual = rlt::get_ref(device, parameters, instance_i);
            if(instance_i % 2 == 0){
                EXPECT_EQ(actual.drone, before[instance_i].drone);
                EXPECT_EQ(actual.dynamics.dynamics.mass, before[instance_i].dynamics.dynamics.mass);
                EXPECT_EQ(actual.camera.fov, before[instance_i].camera.fov);
            }
            else{
                const auto& adopted = rlt::get_ref(device, setups, rlt::get(device, setup_index, instance_i));
                EXPECT_EQ(actual.drone, adopted.drone);
                EXPECT_EQ(actual.camera.fov, adopted.camera.fov);
                EXPECT_EQ(actual.dynamics.dynamics.mass, table.drones[actual.drone].parameters.dynamics.dynamics.mass);
                counts[actual.drone]++;
            }
        }
    }
    for(TI count : counts){
        EXPECT_GT(count, 0);
    }
    rlt::free(device_gpu, rng);
    rlt::free(device_gpu, mask_gpu);
    rlt::free(device_gpu, states_gpu);
    rlt::free(device_gpu, parameters_gpu);
    rlt::free(device, mask);
    rlt::free(device, setup_index);
    rlt::free(device, setups);
    rlt::free(device, parameters);
    rlt::free(device_gpu, world);
    rlt::free(device, shared.library);
}

// the device-resident path samples the drone entry per instance on the GPU and writes the body's
// mesh ids into the device structure rows; the CPU verbs on the same arranged state, which swap
// the host truth instead, must produce the same own-drone masks
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES_CUDA, SAMPLE_AND_RENDER_MATCH_CPU){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    if(!cuda_available()){
        GTEST_SKIP() << "no CUDA device available";
    }
    DEVICE_GPU device_gpu;
    auto& device = device_gpu.rendering;
    rlt::init(device);
    rlt::init(device_gpu);
    WORLD world_gpu, world_cpu;
    typename WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    shared.drones = one_asset_per_dynamics().drones;
    rlt::malloc(device_gpu, world_gpu);
    rlt::malloc(device, world_cpu);
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device_gpu, world_gpu, shared, dataset, corpus, 0, 1, 0);
    rlt::init(device, world_cpu, shared, dataset, corpus, 0, 1, 0);
    ASSERT_EQ(shared.num_drones, 4);
    ASSERT_EQ(shared.drone_meshes.size(), 4);
    EXPECT_EQ(world_gpu.num_drones, 4);
    EXPECT_EQ(world_cpu.num_drones, 4);

    RNG_GPU rng;
    rlt::malloc(device_gpu, rng);
    rlt::init(device_gpu, rng, 1337);
    PARAMETERS_TENSOR parameters, parameters_gpu;
    STATES_TENSOR states, states_gpu;
    MASK_TENSOR reset_mask, reset_mask_gpu;
    rlt::malloc(device, parameters);
    rlt::malloc(device, states);
    rlt::malloc(device, reset_mask);
    rlt::malloc(device_gpu, parameters_gpu);
    rlt::malloc(device_gpu, states_gpu);
    rlt::malloc(device_gpu, reset_mask_gpu);

    // GPU sampling covers the table and applies the entry's nominal
    std::vector<TI> counts(4, 0);
    rlt::set_all(device_gpu, reset_mask_gpu, true);
    for(TI round_i = 0; round_i < 32; round_i++){
        rlt::sample_initial_parameters(device_gpu, world_gpu, parameters_gpu, reset_mask_gpu, rng);
        cudaDeviceSynchronize();
        rlt::copy(device_gpu, device, parameters_gpu, parameters);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const auto instance_parameters = rlt::get(device, parameters, instance_i);
            ASSERT_LT(instance_parameters.drone, 4);
            counts[instance_parameters.drone]++;
            const auto& nominal = shared.drones[instance_parameters.drone].parameters;
            EXPECT_EQ(instance_parameters.dynamics.dynamics.mass, nominal.dynamics.dynamics.mass);
            EXPECT_EQ(instance_parameters.camera.fov, nominal.camera.fov);
        }
    }
    for(TI drone_i = 0; drone_i < 4; drone_i++){
        EXPECT_GT(counts[drone_i], 0) << "drone " << drone_i << " never sampled";
    }

    rlt::sample_initial_state(device_gpu, world_gpu, parameters_gpu, states_gpu, reset_mask_gpu, rng);
    cudaDeviceSynchronize();
    rlt::copy(device_gpu, device, parameters_gpu, parameters);
    rlt::copy(device_gpu, device, states_gpu, states);
    arrange(device, parameters, states, shared.drones, {0, 1, 2, 3}, (T)0.3);
    rlt::copy(device, device_gpu, parameters, parameters_gpu);
    rlt::copy(device, device_gpu, states, states_gpu);
    rlt::render(device_gpu, world_gpu, parameters_gpu, states_gpu, reset_mask_gpu);
    cudaDeviceSynchronize();
    const auto masks_gpu = own_masks(device, device_gpu, world_gpu, shared);
    rlt::set_all(device, reset_mask, true);
    rlt::render(device, world_cpu, parameters, states, reset_mask);
    check_placements(world_cpu, shared, {0, 1, 2, 3});
    const auto masks_cpu = own_masks(device, device, world_cpu, shared);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_TRUE(seen(masks_gpu[instance_i])) << "camera " << instance_i;
        EXPECT_EQ(masks_gpu[instance_i], masks_cpu[instance_i]) << "instance " << instance_i;
    }

    // swapping every instance onto the x500 and back on the device
    arrange(device, parameters, states, shared.drones, {3, 3, 3, 3}, (T)0.3);
    rlt::copy(device, device_gpu, parameters, parameters_gpu);
    rlt::copy(device, device_gpu, states, states_gpu);
    rlt::set_all(device_gpu, reset_mask_gpu, false);
    rlt::render(device_gpu, world_gpu, parameters_gpu, states_gpu, reset_mask_gpu);
    cudaDeviceSynchronize();
    const auto masks_x500 = own_masks(device, device_gpu, world_gpu, shared);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_EQ(masks_x500[instance_i], masks_gpu[3]) << "instance " << instance_i;
    }
    arrange(device, parameters, states, shared.drones, {0, 1, 2, 3}, (T)0.3);
    rlt::copy(device, device_gpu, parameters, parameters_gpu);
    rlt::copy(device, device_gpu, states, states_gpu);
    rlt::render(device_gpu, world_gpu, parameters_gpu, states_gpu, reset_mask_gpu);
    cudaDeviceSynchronize();
    const auto masks_back = own_masks(device, device_gpu, world_gpu, shared);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_EQ(masks_back[instance_i], masks_gpu[instance_i]) << "instance " << instance_i;
    }

    rlt::free(device, parameters);
    rlt::free(device, states);
    rlt::free(device, reset_mask);
    rlt::free(device_gpu, parameters_gpu);
    rlt::free(device_gpu, states_gpu);
    rlt::free(device_gpu, reset_mask_gpu);
    rlt::free(device_gpu, rng);
    rlt::free(device_gpu, world_gpu);
    rlt::free(device, world_cpu);
    rlt::free(device, shared.library);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
