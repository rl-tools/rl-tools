#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/yaw/operations_generic.h>

#include <gtest/gtest.h>
#include <metra/metra.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace rl_tools::test_hyperdrone_yaw{
    using DEVICE = devices::DefaultCPU;
    using TI = DEVICE::index_t;
    using T = double;
    using STATE = rl::environments::l2f::StateBase<rl::environments::l2f::StateSpecification<T, TI>>;
    namespace l2f = rl::environments::l2f;
    namespace obs = l2f::observation;
    struct TEACHER_STATIC_PARAMETERS: l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI>::STATIC_PARAMETERS{
        using STATE_IMU = l2f::StateIMU<T, TI, STATE>;
        using STATE_MAHONY = l2f::StateMahony<l2f::StateMahonySpecification<T, TI, STATE_IMU>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, 8, false, STATE_MAHONY>>;
        using OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI, obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI, obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI, obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI, obs::ActionHistory<obs::ActionHistorySpecification<T, TI, 1>>>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = obs::Position<obs::PositionSpecificationPrivileged<T, TI, obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecificationPrivileged<T, TI, obs::LinearVelocity<obs::LinearVelocitySpecificationPrivileged<T, TI, obs::AngularVelocity<obs::AngularVelocitySpecificationPrivileged<T, TI, obs::ActionHistory<obs::ActionHistorySpecification<T, TI, 1>>>>>>>>>>;
    };
    using TEACHER_ENV = rl::environments::Multirotor<l2f::Specification<T, TI, TEACHER_STATIC_PARAMETERS>>;
    static_assert(TEACHER_ENV::OBSERVATION_DIM == 22 && TEACHER_ENV::OBSERVATION_DIM_PRIVILEGED == 22);

    STATE orientation(T yaw, T pitch, T roll){
        const T cy = std::cos(yaw / 2), sy = std::sin(yaw / 2);
        const T cp = std::cos(pitch / 2), sp = std::sin(pitch / 2);
        const T cr = std::cos(roll / 2), sr = std::sin(roll / 2);
        STATE state{};
        state.orientation[0] = cr * cp * cy + sr * sp * sy;
        state.orientation[1] = sr * cp * cy - cr * sp * sy;
        state.orientation[2] = cr * sp * cy + sr * cp * sy;
        state.orientation[3] = cr * cp * sy - sr * sp * cy;
        return state;
    }
}

using namespace rl_tools;
using namespace rl_tools::test_hyperdrone_yaw;

TEST(RL_TOOLS_HYPERDRONE_YAW, HEADING_WITH_TILT_AND_QUATERNION_SIGN){
    DEVICE device;
    T worst_error = 0;
    for(T yaw : {-3.0, -1.5, -0.2, 0.0, 0.7, 1.5, 3.0}){
        for(T pitch : {-1.2, 0.0, 1.2}){
            for(T roll : {-2.0, 0.0, 2.0}){
                auto state = orientation(yaw, pitch, roll);
                for(int sign : {1, -1}){
                    for(T& component : state.orientation){
                        component *= sign;
                    }
                    T cosine, sine;
                    yaw_target(device, state, cosine, sine);
                    EXPECT_NEAR(cosine, std::cos(yaw), 1e-12);
                    EXPECT_NEAR(sine, std::sin(yaw), 1e-12);
                    worst_error = std::max(worst_error, std::abs(std::atan2(sine, cosine) - yaw));
                }
            }
        }
    }
    metra::log("hyperdrone/yaw/label_max_error_rad", worst_error);
}

TEST(RL_TOOLS_HYPERDRONE_YAW, VERTICAL_HEADING_FALLBACK){
    DEVICE device;
    auto state = orientation(0.9, math::PI<T> / 2, -0.4);
    T cosine, sine;
    yaw_target(device, state, cosine, sine);
    EXPECT_EQ(cosine, 1);
    EXPECT_EQ(sine, 0);
}

TEST(RL_TOOLS_HYPERDRONE_YAW, ANGULAR_ERROR_WRAPS_AND_IGNORES_MAGNITUDE){
    DEVICE device;
    const T angle = math::PI<T> * 179 / 180;
    EXPECT_NEAR(yaw_error(device, std::cos(angle), std::sin(angle), std::cos(-angle), std::sin(-angle)), -math::PI<T> / 90, 1e-12);
    EXPECT_NEAR(yaw_error(device, 3 * std::cos(0.6), 3 * std::sin(0.6), std::cos(-0.2), std::sin(-0.2)), 0.8, 1e-12);
    EXPECT_NEAR(yaw_error(device, (T)0, (T)0, std::cos(0.4), std::sin(0.4)), -0.4, 1e-12);
}

TEST(RL_TOOLS_HYPERDRONE_YAW, STRIDED_BATCH_LABELS){
    DEVICE device;
    Tensor<tensor::Specification<STATE, TI, tensor::Shape<TI, 3>, false>> states;
    Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 3, 4>, false>> storage;
    set_all(device, storage, (T)17);
    for(TI i = 0; i < 3; i++){
        set(device, states, orientation((T)i - 1, 0.4, -0.3), i);
    }
    auto labels = view_range(device, storage, (TI)1, tensor::ViewSpec<1, 2>{});
    yaw_targets(device, states, labels);
    for(TI i = 0; i < 3; i++){
        EXPECT_EQ(get(device, storage, i, 0), 17);
        EXPECT_NEAR(get(device, storage, i, 1), std::cos((T)i - 1), 1e-12);
        EXPECT_NEAR(get(device, storage, i, 2), std::sin((T)i - 1), 1e-12);
        EXPECT_EQ(get(device, storage, i, 3), 17);
    }
}

TEST(RL_TOOLS_HYPERDRONE_YAW, CURRICULUM_FRONTIER_GATE){
    DEVICE device;
    EXPECT_EQ(yaw_curriculum_threshold(device, (T)0, (T)0.25, (T)2.5, (T)5), 2.5);
    EXPECT_EQ(yaw_curriculum_threshold(device, (T)12, (T)0.25, (T)2.5, (T)5), 3);
    EXPECT_EQ(yaw_curriculum_threshold(device, (T)89, (T)0.25, (T)2.5, (T)5), 5);
    EXPECT_EQ(yaw_curriculum_advance(device, (T)0, (T)90, (T)1, (T)2.5, (T)2.5), 1);
    EXPECT_EQ(yaw_curriculum_advance(device, (T)0, (T)90, (T)1, (T)2.6, (T)2.5), 0);
    EXPECT_EQ(yaw_curriculum_advance(device, (T)89.5, (T)90, (T)1, (T)5, (T)5), 90);
    EXPECT_EQ(yaw_curriculum_advance(device, (T)90, (T)90, (T)1, (T)0, (T)5), 90);
    EXPECT_EQ(yaw_curriculum_advance(device, (T)3, (T)90, (T)1, std::numeric_limits<T>::infinity(), (T)5), 3);
    EXPECT_EQ(yaw_curriculum_advance(device, (T)3, (T)90, (T)1, std::numeric_limits<T>::quiet_NaN(), (T)5), 3);
}

TEST(RL_TOOLS_HYPERDRONE_YAW, TEACHER_OBSERVATION_PRESERVES_IMU_BIAS_AND_LATEST_ACTION){
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    init(device);
    malloc(device, rng);
    init(device, rng, 0);
    TEACHER_ENV env;
    TEACHER_ENV::Parameters parameters = env.parameters;
    TEACHER_ENV::State state{};
    static_cast<STATE&>(state) = orientation(math::PI<T> / 2, 0, 0);
    const T expected[22] = {0.1, -0.2, 0.3, 0, -1, 0, 1, 0, 0, 0, 0, 1, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, -0.6, -0.2, 0.2, 0.6};
    const T bias[3] = {0.01, -0.02, 0.015};
    for(TI axis = 0; axis < 3; axis++){
        state.position[axis] = expected[axis];
        state.linear_velocity[axis] = expected[12 + axis];
        state.angular_velocity[axis] = expected[15 + axis];
        state.gyro_bias[axis] = bias[axis];
        state.gyro[axis] = 100 + axis;
    }
    state.rotor_history_step = 6;
    for(TI history = 0; history < 8; history++){
        for(TI action = 0; action < 4; action++){
            state.action_history[history][action] = history == 5 ? expected[18 + action] : -0.95;
        }
    }
    Matrix<matrix::Specification<T, TI, 1, 22, false>> teacher, privileged;
    observe(device, env, parameters, state, TEACHER_ENV::Observation{}, teacher, rng);
    parameters.mdp.observation_noise = {123, 123, 123, 123, 123};
    observe(device, env, parameters, state, TEACHER_ENV::ObservationPrivileged{}, privileged, rng);
    for(TI column = 0; column < 22; column++){
        const T expected_bias = column >= 15 && column < 18 ? bias[column - 15] : 0;
        EXPECT_NEAR(get(teacher, 0, column), expected[column] + expected_bias, 1e-12) << "teacher column " << column;
        EXPECT_NEAR(get(privileged, 0, column), expected[column], 1e-12) << "privileged column " << column;
    }
    free(device, rng);
}
