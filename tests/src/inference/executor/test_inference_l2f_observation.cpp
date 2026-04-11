#include <rl_tools/operations/cpu.h>

#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/inference/executor/executor.h>
#include <rl_tools/inference/executor/operations_generic.h>
#include <rl_tools/inference/applications/l2f/l2f.h>
#include <rl_tools/inference/applications/l2f/operations_generic.h>
#include <rl_tools/inference/applications/l2f/operations_dyn.h>

#include "../../../../tests/data/test_inference_executor_policy.h"

#include <gtest/gtest.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

namespace l2f = rlt::inference::applications::l2f;

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, EXTRACT_BASIC){
    const char* meta = R"JSON({"environment": {"name": "l2f", "observation": "Position.OrientationRotationMatrix.LinearVelocity.AngularVelocity.ActionHistory(1)"}})JSON";
    TI meta_len = 0;
    while(meta[meta_len] != '\0') meta_len++;
    char obs_buf[256];
    TI obs_len = 0;
    bool result = l2f::extract_observation_from_meta(meta, meta_len, obs_buf, (TI)256, obs_len);
    ASSERT_TRUE(result);
    ASSERT_STREQ(obs_buf, "Position.OrientationRotationMatrix.LinearVelocity.AngularVelocity.ActionHistory(1)");
    ASSERT_EQ(obs_len, (TI)82);
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, EXTRACT_NO_MATCH_OBSERVATION_NOISE){
    const char* meta = R"JSON({"observation_noise": 0.1})JSON";
    TI meta_len = 0;
    while(meta[meta_len] != '\0') meta_len++;
    char obs_buf[256];
    TI obs_len = 0;
    bool result = l2f::extract_observation_from_meta(meta, meta_len, obs_buf, (TI)256, obs_len);
    ASSERT_FALSE(result);
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, EXTRACT_OBSERVATION_AFTER_NOISE){
    const char* meta = R"JSON({"observation_noise": 0.1, "observation": "Position.LinearVelocity"})JSON";
    TI meta_len = 0;
    while(meta[meta_len] != '\0') meta_len++;
    char obs_buf[256];
    TI obs_len = 0;
    bool result = l2f::extract_observation_from_meta(meta, meta_len, obs_buf, (TI)256, obs_len);
    ASSERT_TRUE(result);
    ASSERT_STREQ(obs_buf, "Position.LinearVelocity");
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, EXTRACT_OBSERVATION_IN_VALUE_STRING){
    const char* meta = R"JSON({"description": "the observation is important", "observation": "Position"})JSON";
    TI meta_len = 0;
    while(meta[meta_len] != '\0') meta_len++;
    char obs_buf[256];
    TI obs_len = 0;
    bool result = l2f::extract_observation_from_meta(meta, meta_len, obs_buf, (TI)256, obs_len);
    ASSERT_TRUE(result);
    ASSERT_STREQ(obs_buf, "Position");
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, EXTRACT_ESCAPED_QUOTES){
    const char* meta = R"JSON({"key": "val\"observation\": \"fake\"", "observation": "AngularVelocity"})JSON";
    TI meta_len = 0;
    while(meta[meta_len] != '\0') meta_len++;
    char obs_buf[256];
    TI obs_len = 0;
    bool result = l2f::extract_observation_from_meta(meta, meta_len, obs_buf, (TI)256, obs_len);
    ASSERT_TRUE(result);
    ASSERT_STREQ(obs_buf, "AngularVelocity");
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, EXTRACT_NESTED_JSON){
    const char* meta = R"JSON({"environment": {"name": "l2f", "observation": "Position.OrientationRotationMatrix.LinearVelocity.AngularVelocityDelayed(0).ActionHistory(1)"}})JSON";
    TI meta_len = 0;
    while(meta[meta_len] != '\0') meta_len++;
    char obs_buf[256];
    TI obs_len = 0;
    bool result = l2f::extract_observation_from_meta(meta, meta_len, obs_buf, (TI)256, obs_len);
    ASSERT_TRUE(result);
    ASSERT_STREQ(obs_buf, "Position.OrientationRotationMatrix.LinearVelocity.AngularVelocityDelayed(0).ActionHistory(1)");
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, EXTRACT_NOT_FOUND){
    const char* meta = R"JSON({"name": "l2f"})JSON";
    TI meta_len = 0;
    while(meta[meta_len] != '\0') meta_len++;
    char obs_buf[256];
    TI obs_len = 0;
    bool result = l2f::extract_observation_from_meta(meta, meta_len, obs_buf, (TI)256, obs_len);
    ASSERT_FALSE(result);
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, PARSE_BASIC){
    DEVICE device;
    rlt::init(device);
    const char* obs_str = "Position.OrientationRotationMatrix.LinearVelocity.AngularVelocity.ActionHistory(1)";
    TI obs_len = 0;
    while(obs_str[obs_len] != '\0') obs_len++;
    l2f::ObservationLayout<TI> layout;
    bool result = l2f::parse_observation_string(device, obs_str, obs_len, layout, (TI)4);
    ASSERT_TRUE(result);
    ASSERT_EQ(layout.component_count, (TI)5);
    ASSERT_EQ(layout.total_dim, (TI)22); // 3+9+3+3+4
    ASSERT_EQ(layout.action_history_length, (TI)1);

    ASSERT_TRUE(layout.components[0].type == l2f::ObservationComponentType::POSITION);
    ASSERT_EQ(layout.components[0].offset, (TI)0);
    ASSERT_EQ(layout.components[0].dim, (TI)3);

    ASSERT_TRUE(layout.components[1].type == l2f::ObservationComponentType::ORIENTATION_ROTATION_MATRIX);
    ASSERT_EQ(layout.components[1].offset, (TI)3);
    ASSERT_EQ(layout.components[1].dim, (TI)9);

    ASSERT_TRUE(layout.components[2].type == l2f::ObservationComponentType::LINEAR_VELOCITY);
    ASSERT_EQ(layout.components[2].offset, (TI)12);
    ASSERT_EQ(layout.components[2].dim, (TI)3);

    ASSERT_TRUE(layout.components[3].type == l2f::ObservationComponentType::ANGULAR_VELOCITY);
    ASSERT_EQ(layout.components[3].offset, (TI)15);
    ASSERT_EQ(layout.components[3].dim, (TI)3);

    ASSERT_TRUE(layout.components[4].type == l2f::ObservationComponentType::ACTION_HISTORY);
    ASSERT_EQ(layout.components[4].offset, (TI)18);
    ASSERT_EQ(layout.components[4].dim, (TI)4);
    ASSERT_EQ(layout.components[4].parameter, (TI)1);
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, PARSE_DELAYED){
    DEVICE device;
    rlt::init(device);
    const char* obs_str = "Position.AngularVelocityDelayed(0).LinearVelocityDelayed(0)";
    TI obs_len = 0;
    while(obs_str[obs_len] != '\0') obs_len++;
    l2f::ObservationLayout<TI> layout;
    bool result = l2f::parse_observation_string(device, obs_str, obs_len, layout, (TI)4);
    ASSERT_TRUE(result);
    ASSERT_EQ(layout.component_count, (TI)3);
    ASSERT_EQ(layout.total_dim, (TI)9); // 3+3+3

    ASSERT_TRUE(layout.components[1].type == l2f::ObservationComponentType::ANGULAR_VELOCITY_DELAYED);
    ASSERT_EQ(layout.components[1].parameter, (TI)0);
    ASSERT_TRUE(layout.components[2].type == l2f::ObservationComponentType::LINEAR_VELOCITY_DELAYED);
    ASSERT_EQ(layout.components[2].parameter, (TI)0);
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, PARSE_QUATERNION){
    DEVICE device;
    rlt::init(device);
    const char* obs_str = "Position.OrientationQuaternion.LinearVelocity";
    TI obs_len = 0;
    while(obs_str[obs_len] != '\0') obs_len++;
    l2f::ObservationLayout<TI> layout;
    bool result = l2f::parse_observation_string(device, obs_str, obs_len, layout, (TI)4);
    ASSERT_TRUE(result);
    ASSERT_EQ(layout.component_count, (TI)3);
    ASSERT_EQ(layout.total_dim, (TI)10); // 3+4+3

    ASSERT_TRUE(layout.components[1].type == l2f::ObservationComponentType::ORIENTATION_QUATERNION);
    ASSERT_EQ(layout.components[1].dim, (TI)4);
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, PARSE_ACTION_HISTORY_2){
    DEVICE device;
    rlt::init(device);
    const char* obs_str = "Position.ActionHistory(2)";
    TI obs_len = 0;
    while(obs_str[obs_len] != '\0') obs_len++;
    l2f::ObservationLayout<TI> layout;
    bool result = l2f::parse_observation_string(device, obs_str, obs_len, layout, (TI)4);
    ASSERT_TRUE(result);
    ASSERT_EQ(layout.component_count, (TI)2);
    ASSERT_EQ(layout.total_dim, (TI)11); // 3+4*2
    ASSERT_EQ(layout.action_history_length, (TI)2);
    ASSERT_EQ(layout.components[1].dim, (TI)8);
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, PARSE_EMPTY){
    DEVICE device;
    rlt::init(device);
    const char* obs_str = "";
    l2f::ObservationLayout<TI> layout;
    bool result = l2f::parse_observation_string(device, obs_str, (TI)0, layout, (TI)4);
    ASSERT_FALSE(result);
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, VALIDATE_OK){
    DEVICE device;
    rlt::init(device);
    const char* obs_str = "Position.OrientationRotationMatrix.LinearVelocity.AngularVelocity.ActionHistory(1)";
    TI obs_len = 0;
    while(obs_str[obs_len] != '\0') obs_len++;
    l2f::ObservationLayout<TI> layout;
    l2f::parse_observation_string(device, obs_str, obs_len, layout, (TI)4);
    bool result = l2f::validate_observation_layout(device, layout, (TI)1);
    ASSERT_TRUE(result);
}


TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, OBSERVE_DYNAMIC_BASIC){
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TIMESTAMP = uint64_t;
    using POLICY = rlt::checkpoint::actor::TYPE;
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr TI OUTPUT_DIM = 4;
    static constexpr TIMESTAMP CONTROL_INTERVAL_INTERMEDIATE_NS = 2500 * 1000;
    static constexpr TIMESTAMP CONTROL_INTERVAL_NATIVE_NS = 10000 * 1000;
    using SPEC = rlt::inference::applications::l2f::Specification<TYPE_POLICY, TI, TIMESTAMP, ACTION_HISTORY_LENGTH, OUTPUT_DIM, POLICY, CONTROL_INTERVAL_INTERMEDIATE_NS, CONTROL_INTERVAL_NATIVE_NS>;

    DEVICE device;
    rlt::init(device);

    rlt::inference::applications::L2F<SPEC> executor;
    rlt::malloc(device, executor);

    auto& layout = executor.observation_layout;
    layout.component_count = 3;
    layout.total_dim = 10;
    layout.action_history_length = 1;
    layout.components[0] = {l2f::ObservationComponentType::POSITION, 0, 0, 3};
    layout.components[1] = {l2f::ObservationComponentType::ANGULAR_VELOCITY, 0, 3, 3};
    layout.components[2] = {l2f::ObservationComponentType::ACTION_HISTORY, 1, 6, 4};

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> obs_flat{};
    TI shape[] = {1, 10};
    rlt::dyn::set_shape(obs_flat, (TI)2, shape);
    obs_flat.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, obs_flat);

    l2f::Observation<SPEC> observation;
    observation.position[0] = 1.0f; observation.position[1] = 2.0f; observation.position[2] = 3.0f;
    observation.position_set = true;
    observation.angular_velocity[0] = 4.0f; observation.angular_velocity[1] = 5.0f; observation.angular_velocity[2] = 6.0f;
    observation.angular_velocity_set = true;
    executor.action_history[0][0] = 0.1f;
    executor.action_history[0][1] = 0.2f;
    executor.action_history[0][2] = 0.3f;
    executor.action_history[0][3] = 0.4f;

    bool result = l2f::observe(device, executor, observation, obs_flat);
    ASSERT_TRUE(result);

    ASSERT_FLOAT_EQ(rlt::get(device, obs_flat, (TI)0), 1.0f);
    ASSERT_FLOAT_EQ(rlt::get(device, obs_flat, (TI)1), 2.0f);
    ASSERT_FLOAT_EQ(rlt::get(device, obs_flat, (TI)2), 3.0f);
    ASSERT_FLOAT_EQ(rlt::get(device, obs_flat, (TI)3), 4.0f);
    ASSERT_FLOAT_EQ(rlt::get(device, obs_flat, (TI)4), 5.0f);
    ASSERT_FLOAT_EQ(rlt::get(device, obs_flat, (TI)5), 6.0f);
    ASSERT_FLOAT_EQ(rlt::get(device, obs_flat, (TI)6), 0.1f);
    ASSERT_FLOAT_EQ(rlt::get(device, obs_flat, (TI)7), 0.2f);
    ASSERT_FLOAT_EQ(rlt::get(device, obs_flat, (TI)8), 0.3f);
    ASSERT_FLOAT_EQ(rlt::get(device, obs_flat, (TI)9), 0.4f);

    rlt::free(device, obs_flat);
    rlt::free(device, executor);
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, OBSERVE_DYNAMIC_MISSING_SET){
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TIMESTAMP = uint64_t;
    using POLICY = rlt::checkpoint::actor::TYPE;
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr TI OUTPUT_DIM = 4;
    static constexpr TIMESTAMP CONTROL_INTERVAL_INTERMEDIATE_NS = 2500 * 1000;
    static constexpr TIMESTAMP CONTROL_INTERVAL_NATIVE_NS = 10000 * 1000;
    using SPEC = rlt::inference::applications::l2f::Specification<TYPE_POLICY, TI, TIMESTAMP, ACTION_HISTORY_LENGTH, OUTPUT_DIM, POLICY, CONTROL_INTERVAL_INTERMEDIATE_NS, CONTROL_INTERVAL_NATIVE_NS>;

    DEVICE device;
    rlt::init(device);

    rlt::inference::applications::L2F<SPEC> executor;
    rlt::malloc(device, executor);

    auto& layout = executor.observation_layout;
    layout.component_count = 1;
    layout.total_dim = 3;
    layout.components[0] = {l2f::ObservationComponentType::POSITION, 0, 0, 3};

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> obs_flat{};
    TI shape[] = {1, 3};
    rlt::dyn::set_shape(obs_flat, (TI)2, shape);
    obs_flat.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, obs_flat);

    l2f::Observation<SPEC> observation;
    observation.position[0] = 1.0f; observation.position[1] = 2.0f; observation.position[2] = 3.0f;

    bool result = l2f::observe(device, executor, observation, obs_flat);
    ASSERT_FALSE(result);

    rlt::free(device, obs_flat);
    rlt::free(device, executor);
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, OBSERVE_DYNAMIC_ROTATION_MATRIX){
    using T = float;
    using TYPE_POLICY = rlt::numeric_types::Policy<T>;
    using TIMESTAMP = uint64_t;
    using POLICY = rlt::checkpoint::actor::TYPE;
    static constexpr TI ACTION_HISTORY_LENGTH = 1;
    static constexpr TI OUTPUT_DIM = 4;
    static constexpr TIMESTAMP CONTROL_INTERVAL_INTERMEDIATE_NS = 2500 * 1000;
    static constexpr TIMESTAMP CONTROL_INTERVAL_NATIVE_NS = 10000 * 1000;
    using SPEC = rlt::inference::applications::l2f::Specification<TYPE_POLICY, TI, TIMESTAMP, ACTION_HISTORY_LENGTH, OUTPUT_DIM, POLICY, CONTROL_INTERVAL_INTERMEDIATE_NS, CONTROL_INTERVAL_NATIVE_NS>;

    DEVICE device;
    rlt::init(device);

    rlt::inference::applications::L2F<SPEC> executor;
    rlt::malloc(device, executor);

    auto& layout = executor.observation_layout;
    layout.component_count = 1;
    layout.total_dim = 9;
    layout.components[0] = {l2f::ObservationComponentType::ORIENTATION_ROTATION_MATRIX, 0, 0, 9};

    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> obs_flat{};
    TI shape[] = {1, 9};
    rlt::dyn::set_shape(obs_flat, (TI)2, shape);
    obs_flat.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, obs_flat);

    l2f::Observation<SPEC> observation;
    observation.orientation[0] = 1.0f;
    observation.orientation[1] = 0.0f;
    observation.orientation[2] = 0.0f;
    observation.orientation[3] = 0.0f;
    observation.orientation_set = true;

    bool result = l2f::observe(device, executor, observation, obs_flat);
    ASSERT_TRUE(result);

    ASSERT_NEAR(rlt::get(device, obs_flat, (TI)0), 1.0f, 1e-6);
    ASSERT_NEAR(rlt::get(device, obs_flat, (TI)1), 0.0f, 1e-6);
    ASSERT_NEAR(rlt::get(device, obs_flat, (TI)2), 0.0f, 1e-6);
    ASSERT_NEAR(rlt::get(device, obs_flat, (TI)3), 0.0f, 1e-6);
    ASSERT_NEAR(rlt::get(device, obs_flat, (TI)4), 1.0f, 1e-6);
    ASSERT_NEAR(rlt::get(device, obs_flat, (TI)5), 0.0f, 1e-6);
    ASSERT_NEAR(rlt::get(device, obs_flat, (TI)6), 0.0f, 1e-6);
    ASSERT_NEAR(rlt::get(device, obs_flat, (TI)7), 0.0f, 1e-6);
    ASSERT_NEAR(rlt::get(device, obs_flat, (TI)8), 1.0f, 1e-6);

    rlt::free(device, obs_flat);
    rlt::free(device, executor);
}

TEST(RL_TOOLS_INFERENCE_L2F_OBSERVATION, ROUNDTRIP_EXTRACT_AND_PARSE){
    DEVICE device;
    rlt::init(device);
    const char* meta = R"JSON({"environment": {"name": "l2f", "observation": "Position.OrientationRotationMatrix.LinearVelocity.AngularVelocityDelayed(0).ActionHistory(1)"}})JSON";
    TI meta_len = 0;
    while(meta[meta_len] != '\0') meta_len++;

    char obs_buf[256];
    TI obs_len = 0;
    ASSERT_TRUE(l2f::extract_observation_from_meta(meta, meta_len, obs_buf, (TI)256, obs_len));

    l2f::ObservationLayout<TI> layout;
    ASSERT_TRUE(l2f::parse_observation_string(device, obs_buf, obs_len, layout, (TI)4));

    ASSERT_EQ(layout.component_count, (TI)5);
    ASSERT_EQ(layout.total_dim, (TI)22); // 3+9+3+3+4
    ASSERT_EQ(layout.action_history_length, (TI)1);
    ASSERT_TRUE(layout.components[0].type == l2f::ObservationComponentType::POSITION);
    ASSERT_TRUE(layout.components[1].type == l2f::ObservationComponentType::ORIENTATION_ROTATION_MATRIX);
    ASSERT_TRUE(layout.components[2].type == l2f::ObservationComponentType::LINEAR_VELOCITY);
    ASSERT_TRUE(layout.components[3].type == l2f::ObservationComponentType::ANGULAR_VELOCITY_DELAYED);
    ASSERT_EQ(layout.components[3].parameter, (TI)0);
    ASSERT_TRUE(layout.components[4].type == l2f::ObservationComponentType::ACTION_HISTORY);

    ASSERT_TRUE(l2f::validate_observation_layout(device, layout, (TI)1));
}
