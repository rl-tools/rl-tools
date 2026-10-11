#include <rl_tools/operations/cpu.h>
#include "../../../../src/rendering/raytracing/benchmark/simulator_matrix_physics_contract.h"

#include <gtest/gtest.h>
#include <metra/metra.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace rlt = rl_tools;
using namespace rl_tools;
namespace benchmark = rlt::rendering::raytracing::benchmark;
namespace paper = benchmark::paper;
using DEVICE = rlt::devices::DefaultCPU;
using POSE = benchmark::PhysicsCameraPose;
using CAMERA = rlt::rendering::Camera<float>;

class PaperPhysicsCPU: public ::testing::Test {
protected:
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    benchmark::PhysicsSimulation simulation;

    void SetUp() override {
        malloc(device, rng);
        init(device, rng, 123);
    }

    void TearDown() override {
        free_physics_simulation(device, simulation);
        free(device, rng);
    }
};

static double camera_error(const CAMERA& actual, const CAMERA& expected) {
    double maximum_error = 0;
    const float* actual_vectors[] = {actual.pos, actual.dir_00, actual.dir_du, actual.dir_dv};
    const float* expected_vectors[] = {expected.pos, expected.dir_00, expected.dir_du, expected.dir_dv};
    for(int vector_i = 0; vector_i < 4; vector_i++) {
        for(int axis = 0; axis < 3; axis++) {
            EXPECT_TRUE(std::isfinite(actual_vectors[vector_i][axis]));
            const auto error = std::abs(static_cast<double>(actual_vectors[vector_i][axis]) - expected_vectors[vector_i][axis]);
            maximum_error = std::max(maximum_error, error);
            EXPECT_NEAR(actual_vectors[vector_i][axis], expected_vectors[vector_i][axis], 2e-6)
                << "vector=" << vector_i << " axis=" << axis;
        }
    }
    return maximum_error;
}

TEST_F(PaperPhysicsCPU, BATCH_TRAJECTORIES_AND_SCRATCH_CAMERAS) {
    constexpr int NUM_ENVS = 4096;
    constexpr std::array<int, 4> INDICES = {0, 1, 257, 4095};
    const float position[3] = {-3.92f, -5.67f, 1.f};
    std::vector<POSE> poses(NUM_ENVS, POSE{{1, 0, 0}, {0, 0, 1}});
    poses[1] = {{-1, 0, 0}, {0, 0, 1}};
    poses[257] = {{0, 1, 0}, {0, 0, -1}};
    poses[4095] = {{0, 0, 1}, {0, -1, 0}};
    ASSERT_TRUE(init_physics_simulation(device, simulation, NUM_ENVS, position, poses.data(), 80.f, 1.f, 123));
    std::array<paper::ENVIRONMENT, INDICES.size()> environments;
    std::array<paper::STATE, INDICES.size()> initial_states;
    std::array<paper::PARAMETERS, INDICES.size()> parameters;
    for(std::size_t index = 0; index < INDICES.size(); index++) {
        environments[index].parameters = paper::NOMINAL_PARAMETERS;
        parameters[index] = paper::NOMINAL_PARAMETERS;
        initial_states[index] = paper_benchmark_initial_state(device, position, poses[INDICES[index]]);
    }
    auto states = initial_states;
    std::vector<CAMERA> cameras(NUM_ENVS);
    double maximum_error = 0;
    for(int iteration = 0; iteration <= 30; iteration++) {
        const bool scratch = iteration < 5 || (iteration >= 12 && iteration < 18);
        ASSERT_TRUE(physics_step_cameras(device, simulation, scratch ? nullptr : cameras.data(), iteration));
        ASSERT_TRUE(synchronize_physics(device, simulation));
        for(std::size_t index = 0; index < INDICES.size(); index++) {
            paper_benchmark_step(device, environments[index], parameters[index], states[index], initial_states[index], INDICES[index], iteration, rng);
            if(!scratch) {
                SCOPED_TRACE("iteration=" + std::to_string(iteration) + " environment=" + std::to_string(INDICES[index]));
                const auto expected = paper_benchmark_camera(device, states[index], 80.f, 1.f);
                maximum_error = std::max(maximum_error, camera_error(cameras[INDICES[index]], expected));
            }
        }
        if(iteration == 9) {
            EXPECT_NEAR(cameras[0].pos[2], 1.f - 0.00810400117f, 2e-6);
            EXPECT_NEAR(cameras[4095].pos[1], -5.71096754f, 2e-6);
            EXPECT_NEAR(cameras[4095].pos[2], 0.950907707f, 2e-6);
        }
        if(iteration == 10 || iteration == 20) {
            EXPECT_NEAR(cameras[0].pos[2], 1.f - 8.13336446e-05f, 2e-6);
            EXPECT_NEAR(cameras[4095].pos[1], -5.6704092f, 2e-6);
            EXPECT_NEAR(cameras[4095].pos[2], 0.999509513f, 2e-6);
        }
    }
    metra::log("raytracing/paper-physics/cpu-driver-camera-max-error", maximum_error);
}

TEST_F(PaperPhysicsCPU, REINITIALIZATION_AND_LIFECYCLE) {
    const float old_position[3] = {0, 0, 0};
    const POSE old_poses[] = {{{1, 0, 0}, {0, 0, 1}}, {{0, 0, 1}, {0, -1, 0}}};
    ASSERT_TRUE(init_physics_simulation(device, simulation, 2, old_position, old_poses, 80.f, 1.f, 123));
    for(int iteration = 0; iteration < 7; iteration++) {
        ASSERT_TRUE(physics_step_cameras(device, simulation, nullptr, iteration));
    }

    const float position[3] = {1, 2, 3};
    const POSE pose{{0, 1, 0}, {0, 0, -1}};
    ASSERT_TRUE(init_physics_simulation(device, simulation, 1, position, &pose, 65.f, 1.5f, 456));
    EXPECT_TRUE(simulation.error.empty());
    CAMERA camera;
    ASSERT_TRUE(physics_step_cameras(device, simulation, &camera, 0));
    paper::ENVIRONMENT environment;
    environment.parameters = paper::NOMINAL_PARAMETERS;
    auto parameters = paper::NOMINAL_PARAMETERS;
    const auto initial = paper_benchmark_initial_state(device, position, pose);
    auto state = initial;
    paper_benchmark_step(device, environment, parameters, state, initial, 0, 0, rng);
    camera_error(camera, paper_benchmark_camera(device, state, 65.f, 1.5f));

    EXPECT_FALSE(init_physics_simulation(device, simulation, 0, position, &pose, 80.f, 1.f, 0));
    EXPECT_FALSE(simulation.error.empty());
    EXPECT_FALSE(init_physics_simulation(device, simulation, 1, nullptr, &pose, 80.f, 1.f, 0));
    EXPECT_FALSE(init_physics_simulation(device, simulation, 1, position, nullptr, 80.f, 1.f, 0));
    ASSERT_TRUE(physics_step_cameras(device, simulation, &camera, 1));
    paper_benchmark_step(device, environment, parameters, state, initial, 0, 1, rng);
    camera_error(camera, paper_benchmark_camera(device, state, 65.f, 1.5f));

    free_physics_simulation(device, simulation);
    EXPECT_EQ(simulation.impl, nullptr);
    free_physics_simulation(device, simulation);
    EXPECT_FALSE(physics_step_cameras(device, simulation, &camera, 0));
    ASSERT_TRUE(init_physics_simulation(device, simulation, 1, position, &pose, 65.f, 1.5f, 456));
    EXPECT_TRUE(simulation.error.empty());
    ASSERT_TRUE(physics_step_cameras(device, simulation, &camera, 0));
    state = initial;
    paper_benchmark_step(device, environment, parameters, state, initial, 0, 0, rng);
    camera_error(camera, paper_benchmark_camera(device, state, 65.f, 1.5f));
}
