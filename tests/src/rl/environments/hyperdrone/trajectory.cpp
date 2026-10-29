#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/trajectory_operations_generic.h>
#include <metra/metra.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rlt = rl_tools;
namespace trajectory = rlt::rl::environments::hyperdrone::tasks::visual_inertial_localization::trajectory;
using namespace rl_tools;
using DEVICE = rlt::devices::DefaultCPU;
using T = double;
using TI = DEVICE::index_t;
using SPEC = trajectory::Specification<T, TI>;

namespace {
    T norm(const T vector[3]) {
        return std::sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
    }
    template <typename VALUE, typename BITS>
    VALUE from_bits(BITS pattern) {
        volatile BITS source = pattern;
        const BITS bits = source;
        VALUE result;
        static_assert(sizeof(VALUE) == sizeof(BITS));
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_TRAJECTORY, EndpointsAndC2Join) {
    DEVICE device;
    const T a[3] = {-1, 2, 0.3};
    const T b[3] = {2, 3, 1.2};
    const T c[3] = {3, 5, 0.7};
    const T va[3] = {0.2, -0.1, 0};
    const T vb[3] = {0.3, 0.2, 0.1};
    const T vc[3] = {0, 0, 0};
    trajectory::Segment<SPEC> first, second;
    ASSERT_TRUE(init(device, first, a, b, va, vb, T(4)));
    ASSERT_TRUE(init(device, second, b, c, vb, vc, T(5)));
    trajectory::Sample<SPEC> start, left, right, end;
    ASSERT_TRUE(evaluate(device, first, T(0), start));
    ASSERT_TRUE(evaluate(device, first, first.duration, left));
    ASSERT_TRUE(evaluate(device, second, T(0), right));
    ASSERT_TRUE(evaluate(device, second, second.duration, end));
    for (TI axis = 0; axis < 3; axis++) {
        EXPECT_DOUBLE_EQ(start.position[axis], a[axis]);
        EXPECT_DOUBLE_EQ(left.position[axis], b[axis]);
        EXPECT_DOUBLE_EQ(right.position[axis], b[axis]);
        EXPECT_DOUBLE_EQ(end.position[axis], c[axis]);
        EXPECT_NEAR(start.velocity[axis], va[axis], 1e-14);
        EXPECT_NEAR(left.velocity[axis], vb[axis], 1e-14);
        EXPECT_NEAR(right.velocity[axis], vb[axis], 1e-14);
        EXPECT_NEAR(end.velocity[axis], vc[axis], 1e-14);
        EXPECT_NEAR(start.acceleration[axis], 0, 1e-14);
        EXPECT_NEAR(left.acceleration[axis], 0, 1e-14);
        EXPECT_NEAR(right.acceleration[axis], 0, 1e-14);
        EXPECT_NEAR(end.acceleration[axis], 0, 1e-14);
    }
    ASSERT_TRUE(evaluate(device, first, T(-1), start));
    ASSERT_TRUE(evaluate(device, second, T(10), end));
    for (TI axis = 0; axis < 3; axis++) {
        EXPECT_DOUBLE_EQ(start.position[axis], a[axis]);
        EXPECT_DOUBLE_EQ(end.position[axis], c[axis]);
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_TRAJECTORY, RestToRestAnalyticMotion) {
    DEVICE device;
    const T from[3] = {};
    const T to[3] = {3, 4, 0};
    const T zero[3] = {};
    trajectory::Segment<SPEC> segment;
    ASSERT_TRUE(init(device, segment, from, to, zero, zero, T(4)));
    trajectory::Sample<SPEC> sample;
    for (TI step = 0; step <= 100; step++) {
        const T u = (T)step / 100;
        ASSERT_TRUE(evaluate(device, segment, u * segment.duration, sample));
        const T p = 10 * u * u * u - 15 * u * u * u * u + 6 * u * u * u * u * u;
        const T v = 30 * u * u * (1 - u) * (1 - u) / segment.duration;
        const T a = 60 * u * (1 - u) * (1 - 2 * u) / (segment.duration * segment.duration);
        const T j = 60 * (1 - 6 * u + 6 * u * u) / (segment.duration * segment.duration * segment.duration);
        for (TI axis = 0; axis < 3; axis++) {
            EXPECT_NEAR(sample.position[axis], to[axis] * p, 1e-13);
            EXPECT_NEAR(sample.velocity[axis], to[axis] * v, 1e-13);
            EXPECT_NEAR(sample.acceleration[axis], to[axis] * a, 1e-13);
            EXPECT_NEAR(sample.jerk[axis], to[axis] * j, 1e-13);
        }
    }
    trajectory::Bounds<SPEC> bounds;
    ASSERT_TRUE(derivative_bounds(device, segment, bounds));
    EXPECT_GE(bounds.speed, 1.875 * 5 / 4 - 1e-13);
    EXPECT_LE(bounds.speed, 1.875 * 5 / 4 * 1.01);
    EXPECT_GE(bounds.acceleration, 10 / std::sqrt(T(3)) * 5 / 16 - 1e-13);
    EXPECT_LE(bounds.acceleration, 10 / std::sqrt(T(3)) * 5 / 16 * 1.02);
    EXPECT_NEAR(bounds.jerk, T(60) * 5 / 64, 1e-13);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_TRAJECTORY, DerivativeCertificatesContainCurvedMotion) {
    DEVICE device;
    const T from[3] = {-0.2, 0.4, 1.1};
    const T to[3] = {2.7, -0.3, 1.8};
    T peak_ratio = 0;
    for (TI example = 0; example < 20; example++) {
        const T phase = (T)example * T(0.37);
        const T start_velocity[3] = {std::sin(phase), T(0.3), T(-0.1)};
        const T end_velocity[3] = {T(0.2), std::cos(phase), T(0.2)};
        trajectory::Segment<SPEC> segment;
        ASSERT_TRUE(init(device, segment, from, to, start_velocity, end_velocity, T(2) + phase));
        trajectory::Bounds<SPEC> bounds;
        ASSERT_TRUE(derivative_bounds(device, segment, bounds));
        for (TI step = 0; step <= 1000; step++) {
            trajectory::Sample<SPEC> sample;
            ASSERT_TRUE(evaluate(device, segment, segment.duration * (T)step / 1000, sample));
            EXPECT_LE(norm(sample.velocity), bounds.speed + 1e-12);
            EXPECT_LE(norm(sample.acceleration), bounds.acceleration + 1e-12);
            EXPECT_LE(norm(sample.jerk), bounds.jerk + 1e-12);
            peak_ratio = std::max(peak_ratio, norm(sample.velocity) / bounds.speed);
        }
    }
    metra::log("hyperdrone/trajectory/speed_certificate_peak_ratio", peak_ratio);
    EXPECT_GT(peak_ratio, 0.99);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_TRAJECTORY, DurationEstimateAndInvalidInputs) {
    DEVICE device;
    const T from[3] = {};
    const T to[3] = {0.3, 0.4, 0};
    const T zero[3] = {};
    const T speed = 0.6, acceleration = 0.6, jerk = 2;
    const T duration = estimate_duration(device, from, to, speed, acceleration, jerk);
    EXPECT_GT(duration, 0);
    trajectory::Segment<SPEC> segment;
    ASSERT_TRUE(init(device, segment, from, to, zero, zero, duration));
    trajectory::Sample<SPEC> sample;
    for (TI step = 0; step <= 1000; step++) {
        ASSERT_TRUE(evaluate(device, segment, duration * (T)step / 1000, sample));
        EXPECT_LE(norm(sample.velocity), speed + 1e-12);
        EXPECT_LE(norm(sample.acceleration), acceleration + 1e-12);
        EXPECT_LE(norm(sample.jerk), jerk + 1e-12);
    }
    EXPECT_EQ(estimate_duration(device, from, to, T(0), acceleration, jerk), 0);
    EXPECT_EQ(estimate_duration(device, from, to, speed, T(-1), jerk), 0);
    EXPECT_EQ(estimate_duration(device, from, from, speed, acceleration, jerk), 0);
    const T infinity = from_bits<T>(0x7ff0000000000000ull);
    const T nan = from_bits<T>(0x7ff8000000000000ull);
    for (T invalid : {T(0), T(-1), infinity, nan}) {
        EXPECT_FALSE(init(device, segment, from, to, zero, zero, invalid));
        EXPECT_EQ(segment.duration, 0);
        EXPECT_FALSE(evaluate(device, segment, T(0), sample));
        trajectory::Bounds<SPEC> bounds;
        EXPECT_FALSE(derivative_bounds(device, segment, bounds));
    }
    T bad_point[3] = {nan, 0, 0};
    EXPECT_FALSE(init(device, segment, bad_point, to, zero, zero, duration));
    bad_point[0] = infinity;
    EXPECT_FALSE(init(device, segment, bad_point, to, zero, zero, duration));
    EXPECT_EQ(estimate_duration(device, bad_point, to, speed, acceleration, jerk), 0);
    ASSERT_TRUE(init(device, segment, from, to, zero, zero, duration));
    EXPECT_FALSE(evaluate(device, segment, nan, sample));
    EXPECT_FALSE(evaluate(device, segment, infinity, sample));
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_TRAJECTORY, FloatCertificatesAndOverflowRejection) {
    DEVICE device;
    using FLOAT_SPEC = trajectory::Specification<float, TI>;
    const float from[3] = {7, 4, 1};
    const float to[3] = {9, 3, 2};
    const float velocity[3] = {0.3f, -0.1f, 0.05f};
    trajectory::Segment<FLOAT_SPEC> segment;
    ASSERT_TRUE(init(device, segment, from, to, velocity, velocity, 5.0f));
    trajectory::Bounds<FLOAT_SPEC> bounds;
    ASSERT_TRUE(derivative_bounds(device, segment, bounds));
    for (TI step = 0; step <= 1000; step++) {
        trajectory::Sample<FLOAT_SPEC> sample;
        ASSERT_TRUE(evaluate(device, segment, segment.duration * (float)step / 1000, sample));
        float speed_squared = 0, acceleration_squared = 0, jerk_squared = 0;
        for (TI axis = 0; axis < 3; axis++) {
            speed_squared += sample.velocity[axis] * sample.velocity[axis];
            acceleration_squared += sample.acceleration[axis] * sample.acceleration[axis];
            jerk_squared += sample.jerk[axis] * sample.jerk[axis];
        }
        EXPECT_LE(std::sqrt(speed_squared), bounds.speed + 1e-6f);
        EXPECT_LE(std::sqrt(acceleration_squared), bounds.acceleration + 1e-6f);
        EXPECT_LE(std::sqrt(jerk_squared), bounds.jerk + 1e-6f);
    }
    const float infinity = from_bits<float>(0x7f800000u);
    const float nan = from_bits<float>(0x7fc00000u);
    EXPECT_FALSE(init(device, segment, from, to, velocity, velocity, infinity));
    EXPECT_FALSE(init(device, segment, from, to, velocity, velocity, nan));
    ASSERT_TRUE(init(device, segment, from, to, velocity, velocity, 1e-20f));
    EXPECT_FALSE(derivative_bounds(device, segment, bounds));
    trajectory::Sample<FLOAT_SPEC> sample;
    EXPECT_FALSE(evaluate(device, segment, 0.5e-20f, sample));
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_TRAJECTORY, PhaseIsBoundedAndConvergesWithContinuousAcceleration) {
    DEVICE device;
    using FLOAT_SPEC = trajectory::Specification<float, TI>;
    trajectory::Phase<FLOAT_SPEC> phase;
    const float dt = 0.005f, max_acceleration = 0.2f, max_jerk = 0.5f;
    for (TI section = 0; section < 6; section++) {
        const float target = section % 2 == 0 ? 1.0f : 0.0f;
        for (TI step = 0; step < 3000; step++) {
            const float previous_rate = phase.rate;
            const float previous_acceleration = phase.acceleration;
            const float increment = advance(device, phase, target, dt, max_acceleration, max_jerk);
            ASSERT_GE(phase.rate, 0);
            ASSERT_LE(phase.rate, 1);
            ASSERT_LE(std::abs(phase.acceleration), max_acceleration + 1e-7f);
            ASSERT_LE(std::abs(phase.acceleration - previous_acceleration), max_jerk * dt + 1e-7f);
            ASSERT_GE(increment, 0);
            ASSERT_LE(increment, dt);
            EXPECT_NEAR(phase.rate, previous_rate + (previous_acceleration + phase.acceleration) * dt / 2, 2e-7f);
            const float inferred_jerk = (phase.acceleration - previous_acceleration) / dt;
            EXPECT_NEAR(increment, previous_rate * dt + previous_acceleration * dt * dt / 2 + inferred_jerk * dt * dt * dt / 6, 1e-8f);
        }
        EXPECT_NEAR(phase.rate, target, 1e-4f);
        EXPECT_NEAR(phase.acceleration, 0, 1e-4f);
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_TRAJECTORY, PhaseRapidTargetChangesStayInsideViabilityBounds) {
    DEVICE device;
    trajectory::Phase<SPEC> phase;
    const T dt = 0.02, max_acceleration = 0.2, max_jerk = 0.5;
    for (TI step = 0; step < 10000; step++) {
        const T target = (step / 73) % 3 == 0 ? 0 : ((step / 73) % 3 == 1 ? 1 : 0.35);
        const T previous_acceleration = phase.acceleration;
        advance(device, phase, target, dt, max_acceleration, max_jerk);
        ASSERT_GE(phase.rate, 0);
        ASSERT_LE(phase.rate, 1);
        ASSERT_LE(std::abs(phase.acceleration - previous_acceleration), max_jerk * dt + 1e-12);
        const T stopping_distance = phase.acceleration * phase.acceleration / (2 * max_jerk);
        if (phase.acceleration > 0) {
            ASSERT_LE(phase.rate + stopping_distance, 1 + 1e-12);
        } else {
            ASSERT_GE(phase.rate - stopping_distance, -1e-12);
        }
    }
}
