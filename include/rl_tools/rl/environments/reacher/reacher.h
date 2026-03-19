#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_REACHER_REACHER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_REACHER_REACHER_H

#include "../../../math/operations_generic.h"
#include "../environments.h"
#include "../observation.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::reacher{
    template <typename T>
    struct DefaultParameters{
        static constexpr T DT = 0.1;
        static constexpr T ARENA_SIZE = 1.0;
        static constexpr T MAX_VELOCITY = 1.5;
        static constexpr T TARGET_RADIUS = 0.1;
        static constexpr T ACTION_LIMIT = 1.0;
        static constexpr auto IMAGE_HEIGHT = 32;
        static constexpr auto IMAGE_WIDTH = 32;
    };

    template <typename T_T, typename T_TI, typename T_PARAMETERS = DefaultParameters<T_T>>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
        using PARAMETERS = T_PARAMETERS;
    };

    template <typename T_T, typename T_TI>
    struct StateSpecification{
        using T = T_T;
        using TI = T_TI;
    };

    template <typename T_SPEC>
    struct State{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI DIM = 4;
        T x;
        T y;
        T target_x;
        T target_y;
    };

    template <typename T_SPEC>
    struct StateWithStep{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI DIM = 5;
        T x;
        T y;
        T target_x;
        T target_y;
        TI step;
    };

    template <typename TI>
    struct ObservationDense{
        static constexpr TI DIM = 4;
        using SHAPE = tensor::Shape<TI, DIM>;
    };

    template <typename T_TI, T_TI T_HEIGHT, T_TI T_WIDTH>
    struct ObservationImage : observation::Image<T_TI, T_HEIGHT, T_WIDTH, 3>{};

    template <typename T_TI, T_TI T_HEIGHT, T_TI T_WIDTH>
    struct ObservationImageFlat{
        using TI = T_TI;
        static constexpr TI HEIGHT = T_HEIGHT;
        static constexpr TI WIDTH = T_WIDTH;
        static constexpr TI CHANNELS = 3;
        static constexpr TI DIM = HEIGHT * WIDTH * CHANNELS;
        using SHAPE = tensor::Shape<TI, DIM>;
    };

    template <typename T_SPEC>
    struct StateSequentialTargets{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI DIM = 8;
        T x;
        T y;
        T target1_x;
        T target1_y;
        T target2_x;
        T target2_y;
        TI step;
        TI current_target;
    };

    template <typename TI>
    struct ObservationDenseSequentialTargets{
        static constexpr TI DIM = 7;
        using SHAPE = tensor::Shape<TI, DIM>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments{
    template <typename T_SPEC>
    struct Reacher: Environment<typename T_SPEC::T, typename T_SPEC::TI>{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using State = reacher::State<reacher::StateSpecification<T, TI>>;
        using Parameters = typename SPEC::PARAMETERS;
        using Observation = reacher::ObservationDense<TI>;
        using ObservationPrivileged = Observation;
        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 2;
        static constexpr TI EPISODE_STEP_LIMIT = 40;
    };
    template <typename T_SPEC>
    struct ReacherVisual: Environment<typename T_SPEC::T, typename T_SPEC::TI>{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using State = reacher::State<reacher::StateSpecification<T, TI>>;
        using Parameters = typename SPEC::PARAMETERS;
        using Observation = reacher::ObservationImage<TI, SPEC::PARAMETERS::IMAGE_HEIGHT, SPEC::PARAMETERS::IMAGE_WIDTH>;
        using ObservationPrivileged = reacher::ObservationDense<TI>;
        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 2;
        static constexpr TI EPISODE_STEP_LIMIT = 40;
    };
    template <typename T_SPEC>
    struct ReacherMemoryVisual: Environment<typename T_SPEC::T, typename T_SPEC::TI>{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using State = reacher::StateWithStep<reacher::StateSpecification<T, TI>>;
        using Parameters = typename SPEC::PARAMETERS;
        using Observation = reacher::ObservationImage<TI, SPEC::PARAMETERS::IMAGE_HEIGHT, SPEC::PARAMETERS::IMAGE_WIDTH>;
        using ObservationPrivileged = reacher::ObservationDense<TI>;
        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 2;
        static constexpr TI EPISODE_STEP_LIMIT = 40;
    };
    template <typename T_SPEC>
    struct ReacherVisualMemoryHard: Environment<typename T_SPEC::T, typename T_SPEC::TI>{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using State = reacher::StateSequentialTargets<reacher::StateSpecification<T, TI>>;
        using Parameters = typename SPEC::PARAMETERS;
        using Observation = reacher::ObservationImage<TI, SPEC::PARAMETERS::IMAGE_HEIGHT, SPEC::PARAMETERS::IMAGE_WIDTH>;
        using ObservationPrivileged = reacher::ObservationDenseSequentialTargets<TI>;
        static constexpr TI N_AGENTS = 1;
        static constexpr TI ACTION_DIM = 2;
        static constexpr TI EPISODE_STEP_LIMIT = 80;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
