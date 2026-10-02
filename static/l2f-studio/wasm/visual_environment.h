#pragma once
#include <rl_tools/rl/environments/l2f/operations_generic_inertial_velocity.h>

namespace visual_policy {
    using namespace rl_tools::rl::environments::l2f;
    using PARAMETERS_BASE = ParametersBase<builder::PARAMETERS_SPEC>;
    using PARAMETERS_FORCE = ParametersDisturbances<ParametersSpecification<T, TI, PARAMETERS_BASE>>;
    using PARAMETERS_IMU = ParametersIMU<ParametersSpecification<T, TI, PARAMETERS_FORCE>>;
    using PARAMETERS_CTBR = ParametersCTBRController<ParametersSpecification<T, TI, PARAMETERS_IMU>>;
    using PARAMETERS_RANDOMIZATION = ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, builder::DOMAIN_RANDOMIZATION_OPTIONS, PARAMETERS_CTBR>>;
    using PARAMETERS_TRAJECTORY = ParametersTrajectory<ParametersTrajectorySpecification<T, TI, builder::TRAJECTORY_LENGTH, builder::TRAJECTORY_DT, PARAMETERS_RANDOMIZATION>>;
    using PARAMETERS = ParametersObservationDelay<ParametersObservationDelaySpecification<T, TI, PARAMETERS_TRAJECTORY>>;
    static constexpr PARAMETERS::IMU imu = {{{{0, 0}, {0, 0, 0}}}, {{{0, 0}, {0.02f, 0, 0}}}};
    static constexpr PARAMETERS::CTBRController controller = {
        0, 1, {4, 4, 2}, {216, 216, 96}, {5.76f, 5.76f, 1.6f}, {0.003f, 0.003f, 0.0012f}
    };
    static constexpr PARAMETERS nominal_parameters = []{
        PARAMETERS p{};
        p.integration = builder::integration;
        p.mdp = builder::mdp;
        p.imu = imu;
        p.ctbr_controller = controller;
        p.trajectory_parameters = builder::trajectory_parameters;
        p.dynamics = parameters::dynamics::registry<parameters::dynamics::REGISTRY::crazyflie_openmv, builder::PARAMETERS_SPEC>;
        const T hover = 2 * p.dynamics.hovering_throttle_relative - 1;
        p.mdp.init = {0.2f, 0.1f, 0.02f, 0.05f, 0.01f, true, hover, hover};
        return p;
    }();
    using BASE = StateBase<StateSpecification<T, TI>>;
    using LAST_ACTION = StateLastAction<StateSpecification<T, TI, BASE>>;
    using ACCELERATION = StateLinearAcceleration<StateSpecification<T, TI, LAST_ACTION>>;
    using IMU = StateIMU<T, TI, ACCELERATION>;
    struct MAHONY_SPEC: StateMahonySpecification<T, TI, IMU> {
        static constexpr T KI = 0.01f;
    };
    using MAHONY = StateMahony<MAHONY_SPEC>;
    using CONTROLLER = StateCTBRController<StateSpecification<T, TI, MAHONY>>;
    using ACCELERATION_HISTORY = StateLinearAccelerationHistory<StateLinearAccelerationHistorySpecification<T, TI, 8, CONTROLLER>>;
    using ROTORS = StateRotorsHistory<StateRotorsHistorySpecification<T, TI, 8, false, ACCELERATION_HISTORY>>;
    using STATE = StateRandomForce<StateSpecification<T, TI, ROTORS>>;
    using ACTION_OBSERVATION = observation::ActionHistory<observation::ActionHistorySpecification<T, TI, 8>>;
    using ACCELERATION_OBSERVATION = observation::LinearAccelerationBodyFrame<observation::LinearAccelerationBodyFrameSpecification<T, TI, ACTION_OBSERVATION>>;
    using GYRO_OBSERVATION = observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI, ACCELERATION_OBSERVATION>>;
    using VELOCITY_OBSERVATION = observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI, GYRO_OBSERVATION>>;
    using ORIENTATION_OBSERVATION = observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI, VELOCITY_OBSERVATION>>;
    using OBSERVATION = observation::Position<observation::PositionSpecification<T, TI, ORIENTATION_OBSERVATION>>;
    using ACTOR_ACCELERATION = observation::LinearAccelerationBodyFrameHistory<observation::LinearAccelerationBodyFrameHistorySpecification<T, TI, 8>>;
    using ACTOR_ACTION = observation::ActionHistory<observation::ActionHistorySpecification<T, TI, 8, ACTOR_ACCELERATION>>;
    using ACTOR_GYRO = observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI, ACTOR_ACTION>>;
    using ACTOR_OBSERVATION = observation::OrientationMahonyWorldZ<observation::OrientationMahonyWorldZSpecification<T, TI, ACTOR_GYRO>>;
    static_assert(ACTOR_OBSERVATION::DIM == 62);
    struct CONFIG: builder::ENVIRONMENT_STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = parameters::ActionInterface::CTBR;
        using STATE_TYPE = STATE;
        using OBSERVATION_TYPE = OBSERVATION;
        using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION;
        using PARAMETERS = visual_policy::PARAMETERS;
        static constexpr auto PARAMETER_VALUES = nominal_parameters;
    };
    using ENVIRONMENT = rl_tools::rl::environments::Multirotor<Specification<T, TI, CONFIG>>;
    using INERTIAL_SPEC = rl_tools::rl::environments::l2f::inertial_velocity::Specification<rl_tools::numeric_types::Policy<T>, TI>;
    template<bool ENABLED>
    struct PolicyState {};
    template<>
    struct PolicyState<true> {
        rl_tools::rl::environments::l2f::inertial_velocity::State<INERTIAL_SPEC> inertial;
        rl_tools::Matrix<rl_tools::matrix::Specification<T, TI, 1, 65, false>> observation;
        mutable T cache[65];
    };
}
