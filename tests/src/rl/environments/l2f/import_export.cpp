#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/parameters/default.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic.h>

#include <nlohmann/json.hpp>
#include <fstream>

#include <gtest/gtest.h>

#include "../../../utils/utils.h"

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = double;
using TI = typename DEVICE::index_t;


constexpr T EPSILON = 1e-6;

namespace static_parameter_builder{
    // to prevent spamming the global namespace
    using namespace rl_tools::rl::environments::l2f;
    static constexpr bool CLOSED_FORM = true;
    struct ENVIRONMENT_STATIC_PARAMETERS{
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 16;
        static constexpr TI EPISODE_STEP_LIMIT = 500;
        using STATE_BASE = StateBase<StateSpecification<T, TI>>;
        using STATE_TYPE = StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>;
        using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = observation::Position<observation::PositionSpecificationPrivileged<T, TI,
                observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                observation::LinearVelocity<observation::LinearVelocitySpecificationPrivileged<T, TI,
                observation::AngularVelocity<observation::AngularVelocitySpecificationPrivileged<T, TI,
                observation::RandomForce<observation::RandomForceSpecification<T, TI,
                observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>
        >
        >
        >>
        >>
        >>
        >>;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETER_FACTORY = parameters::DefaultParameters<T, TI>;
        static constexpr auto PARAMETER_VALUES = PARAMETER_FACTORY::parameters;
        using PARAMETERS = typename PARAMETER_FACTORY::PARAMETERS_TYPE;
    };
}

using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, static_parameter_builder::ENVIRONMENT_STATIC_PARAMETERS>;
using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;


TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F, IMPORT_EXPORT){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    TI seed = 0;
    rlt::init(device, rng, seed);

    ENVIRONMENT env;
    ENVIRONMENT::Parameters params, params_reconstruct;
    ENVIRONMENT::State state, state_reconstruct;

    rlt::malloc(device, env);
    rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);
    auto params_json = rlt::json(device, env, params);

    nlohmann::json params_json_object = nlohmann::json::parse(params_json);

    unsigned char* raw = reinterpret_cast<unsigned char*>(&params_reconstruct);
    std::generate(raw, raw + sizeof(params_reconstruct), [] { return static_cast<unsigned char>(std::rand() % 256); });
    rlt::from_json(device, env, params_json, params_reconstruct);

    auto params_json_reconstruct = rlt::json(device, env, params_reconstruct);
    ASSERT_EQ(params_json_reconstruct, params_json);
}
