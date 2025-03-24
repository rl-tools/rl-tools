#include <rl_tools/operations/cpu.h>
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

using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI>;
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
