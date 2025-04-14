#ifndef RL_TOOLS_WASM
#include <rl_tools/operations/arm.h>
#else
#include <rl_tools/operations/wasm32.h>
#endif

#include <rl_tools/nn/layers/standardize/operations_generic.h>
#ifndef RL_TOOLS_WASM
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
// #include <rl_tools/nn/layers/dense/operations_generic.h>
#else
#include <rl_tools/nn/layers/dense/operations_generic.h>
#endif
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include "../../../../../tests/data/test_inference_executor_policy.h"

#ifdef RL_TOOLS_ENABLE_INFORMATIVE_STATUS_MESSAGES
#include <cstdio>
#endif

namespace rlt = rl_tools;

#ifndef RL_TOOLS_WASM
using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
#else
using DEVICE = rlt::devices::DefaultWASM32;
#endif


using TI = typename DEVICE::index_t;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
static constexpr TI TEST_SEQUENCE_LENGTH = rlt::checkpoint::example::input::SHAPE::template GET<0>;
static constexpr TI TEST_BATCH_SIZE = rlt::checkpoint::example::input::SHAPE::template GET<1>;
static constexpr TI TEST_SEQUENCE_LENGTH_ACTUAL = 5;
static constexpr TI TEST_BATCH_SIZE_ACTUAL = 2;
static_assert(TEST_BATCH_SIZE_ACTUAL <= TEST_BATCH_SIZE);
static_assert(TEST_SEQUENCE_LENGTH_ACTUAL <= TEST_SEQUENCE_LENGTH);
using ACTOR_TYPE_ORIGINAL = rlt::checkpoint::actor::TYPE;
using ACTOR_TYPE_TEST = rlt::checkpoint::actor::TYPE::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_SEQUENCE_LENGTH<TI, 1>;
using ACTOR_TYPE = ACTOR_TYPE_ORIGINAL::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_SEQUENCE_LENGTH<TI, 1>;
auto& rl_tools_inference_applications_l2f_policy = rlt::checkpoint::actor::module;

#include <rl_tools/inference/applications/l2f/c_backend.h>

