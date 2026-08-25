#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#define HYPERDRONE_DEMO_STRINGIFY_INNER(x) #x
#define HYPERDRONE_DEMO_STRINGIFY(x) HYPERDRONE_DEMO_STRINGIFY_INNER(x)

namespace hyperdrone_demo {
    namespace rlt = rl_tools;
    namespace l2f = rl_tools::rl::environments::l2f;

    template <typename T_T, typename T_TI>
    struct DynamicsStaticParameters {
        using T = T_T;
        using TI = T_TI;
        using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
        static constexpr TI EPISODE_STEP_LIMIT = 500;
        using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
        using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
        using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
                l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
                l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
                l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = PARAMETERS_TYPE;
        static constexpr auto dynamics = l2f::parameters::dynamics::registry<l2f::parameters::dynamics::REGISTRY::crazyflie, PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {(T)0.01};
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {init, REWARD_FUNCTION{}, {}, {}, {}};
        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
        static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {{dynamics, integration, mdp}, disturbances};
        static constexpr T STATE_LIMIT_POSITION_X = 100000;
        static constexpr T STATE_LIMIT_POSITION_Y = 100000;
        static constexpr T STATE_LIMIT_POSITION_Z = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
    };

    inline FILE* open_video_pipe(std::size_t width, std::size_t height, std::size_t framerate, std::size_t upscale, const std::string& output_path){
        char command[1024];
        std::snprintf(command, sizeof(command),
            "ffmpeg -y -hide_banner -loglevel error -f rawvideo -pixel_format rgb24 -video_size %zux%zu -framerate %zu -i - "
            "-c:v libx264 -pix_fmt yuv420p -crf 23 -preset fast -vf \"scale=iw*%zu:ih*%zu:flags=neighbor\" \"%s\"",
            width, height, framerate, upscale, upscale, output_path.c_str());
        return popen(command, "w");
    }
    inline bool close_video_pipe(FILE* pipe){
        return pclose(pipe) == 0;
    }

    inline std::uint8_t quantize_pixel(float value){
        float scaled = value * 255.0f;
        scaled = scaled < 0.0f ? 0.0f : (scaled > 255.0f ? 255.0f : scaled);
        return (std::uint8_t)scaled;
    }

    template <typename DEVICE, typename PARAMETERS>
    auto hover_action(DEVICE& device, const PARAMETERS& parameters, typename DEVICE::index_t rotor_i){
        using T = decltype(parameters.dynamics.mass);
        T gravity_magnitude = rlt::math::abs(device.math, parameters.dynamics.gravity[2]);
        T hover_thrust = parameters.dynamics.mass * gravity_magnitude / (T)4;
        T command = l2f::rotor_command_from_thrust(device, parameters, rotor_i, hover_thrust);
        T action_limit_min = parameters.dynamics.action_limit.min;
        T action_limit_max = parameters.dynamics.action_limit.max;
        return (T)2 * (command - action_limit_min) / (action_limit_max - action_limit_min) - (T)1;
    }
}
