#include <rl_tools/operations/cpu.h>
#include "../../../src/hyperlux/benchmark/simulator_matrix_physics_contract.h"

#include <gtest/gtest.h>
#include <metra/metra.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace rlt = rl_tools;
using namespace rl_tools;
namespace paper = rlt::hyperlux::benchmark::paper;
using DEVICE = rlt::devices::DefaultCPU;
using POSE = rlt::hyperlux::benchmark::PhysicsCameraPose;

static std::array<float, 17> components(const paper::STATE& state) {
    return {state.position[0], state.position[1], state.position[2],
            state.orientation[0], state.orientation[1], state.orientation[2], state.orientation[3],
            state.linear_velocity[0], state.linear_velocity[1], state.linear_velocity[2],
            state.angular_velocity[0], state.angular_velocity[1], state.angular_velocity[2],
            state.rpm[0], state.rpm[1], state.rpm[2], state.rpm[3]};
}

TEST(HYPERLUX_PAPER_PHYSICS, MAY_25_TRAJECTORIES) {
    // Captured from 6c7fc4c6e0c912c451e40856b80740e65f129635's bridge and include/ on Apple Clang, -O2 -ffp-contract=off.
    const std::array<float, 17> expected[2][3] = {
        {
            {2.53302169e-05f, -1.48210574e-05f, -0.00810400117f, 0.999995053f, 0.00151966116f, 0.0027525709f, -0.000268011761f, 0.0012087439f, -0.000691551133f, -0.161875114f, 0.0762206316f, 0.147214413f, -0.0147278802f, 0.726459384f, 0.726035297f, 0.730308115f, 0.72448945f},
            {0.f, 0.f, -8.13336446e-05f, 1.f, 2.11095471e-06f, 3.29433169e-06f, -3.93909829e-07f, 1.40268469e-07f, -9.14398299e-08f, -0.016260989f, 0.00124654546f, 0.00195485051f, -0.000233728962f, 0.72624594f, 0.726113498f, 0.726960599f, 0.725943267f},
            {0.f, 0.f, -8.1327511e-05f, 1.f, 1.77025549e-06f, 2.76380229e-06f, -4.34207891e-07f, 1.17695535e-07f, -7.66768125e-08f, -0.0162591748f, 0.00104556687f, 0.00164001342f, -0.000257663283f, 0.726351023f, 0.726088464f, 0.726950645f, 0.725945592f}
        },
        {
            {-3.91999531f, -5.71096754f, 0.950907707f, 0.497338295f, 0.502559245f, -0.502101898f, 0.497978538f, 0.000252135564f, -0.819662094f, -0.983031392f, 0.0270614661f, -0.254897714f, 0.00482336245f, 0.728999794f, 0.7250247f, 0.724488974f, 0.730309427f},
            {-3.92000008f, -5.6704092f, 0.999509513f, 0.499996334f, 0.500003576f, -0.50000298f, 0.499997109f, 2.84063493e-08f, -0.0818483979f, -0.0981002674f, 0.000393591821f, -0.00388004724f, 5.67757525e-05f, 0.726735473f, 0.725996852f, 0.725943089f, 0.726961017f},
            {-3.92000008f, -5.6704092f, 0.999509513f, 0.499996364f, 0.500003636f, -0.500003278f, 0.499996871f, 1.86670306e-08f, -0.0818495303f, -0.0981002823f, 0.000257085427f, -0.00404222403f, 3.70642665e-05f, 0.726804316f, 0.725980461f, 0.725945413f, 0.72695154f}
        }
    };
    const float positions[2][3] = {{0, 0, 0}, {-3.92f, -5.67f, 1.f}};
    const POSE poses[2] = {{{1, 0, 0}, {0, 0, 1}}, {{0, 0, 1}, {0, -1, 0}}};
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    malloc(device, rng);
    init(device, rng, 0);
    paper::ENVIRONMENT environment;
    auto parameters = paper::NOMINAL_PARAMETERS;
    double maximum_error = 0;
    for(int case_i = 0; case_i < 2; case_i++) {
        auto initial = paper_benchmark_initial_state(device, positions[case_i], poses[case_i]);
        auto state = initial;
        int snapshot = 0;
        for(int iteration = 0; iteration <= 20; iteration++) {
            paper_benchmark_step(device, environment, parameters, state, initial, case_i == 0 ? 0 : 4095, iteration, rng);
            if(iteration == 9 || iteration == 10 || iteration == 20) {
                const auto actual = components(state);
                for(std::size_t component = 0; component < actual.size(); component++) {
                    const double error = std::abs(static_cast<double>(actual[component]) - expected[case_i][snapshot][component]);
                    maximum_error = std::max(maximum_error, error);
                    EXPECT_NEAR(actual[component], expected[case_i][snapshot][component], 2e-6)
                        << "case=" << case_i << " iteration=" << iteration << " component=" << component;
                }
                snapshot++;
            }
        }
    }
    free(device, rng);
    metra::log("hyperlux/paper-physics/historical-state-max-error", maximum_error);
}

TEST(HYPERLUX_PAPER_PHYSICS, DRIFT_RESET) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    malloc(device, rng);
    init(device, rng, 0);
    paper::ENVIRONMENT environment;
    auto parameters = paper::NOMINAL_PARAMETERS;
    const float position[3] = {0, 0, 0};
    const POSE pose{{1, 0, 0}, {0, 0, 1}};
    const auto initial = paper_benchmark_initial_state(device, position, pose);
    for(const auto displacement : {std::array<float, 3>{5, 0, 0}, std::array<float, 3>{0, 0, 3.1f}}) {
        auto state = initial;
        for(int axis = 0; axis < 3; axis++) {
            state.position[axis] += displacement[axis];
        }
        paper_benchmark_step(device, environment, parameters, state, initial, 0, 1, rng);
        EXPECT_EQ(components(state), components(initial));
    }
    free(device, rng);
}

TEST(HYPERLUX_PAPER_PHYSICS, CAMERA_POSE) {
    DEVICE device;
    const float position[3] = {-3.92f, -5.67f, 1.f};
    const POSE poses[] = {{{1, 0, 0}, {0, 0, 1}}, {{-1, 0, 0}, {0, 0, 1}}, {{0, 1, 0}, {0, 0, -1}}, {{0, 0, 1}, {0, -1, 0}}};
    for(const auto& pose : poses) {
        const auto state = paper_benchmark_initial_state(device, position, pose);
        const auto camera = paper_benchmark_camera(device, state, 80.f, 1.f);
        const float target[3] = {position[0] + pose.direction[0], position[1] + pose.direction[1], position[2] + pose.direction[2]};
        const auto expected = make_camera_data(position, target, pose.up, 80.f, 1.f);
        for(int axis = 0; axis < 3; axis++) {
            EXPECT_FLOAT_EQ(camera.pos[axis], expected.pos[axis]);
            EXPECT_NEAR(camera.dir_00[axis], expected.dir_00[axis], 2e-6);
            EXPECT_NEAR(camera.dir_du[axis], expected.dir_du[axis], 2e-6);
            EXPECT_NEAR(camera.dir_dv[axis], expected.dir_dv[axis], 2e-6);
        }
    }
}
