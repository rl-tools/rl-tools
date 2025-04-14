#include <rl_tools/inference/applications/l2f/c_backend.h>

#include <gtest/gtest.h>

TEST(RL_TOOLS_INFERENCE_APPLICATIONS_L2F, MAIN){
    rl_tools_inference_applications_l2f_init();
    RLtoolsInferenceApplicationL2FObservation observation;
    observation.position[0] = 0.0f;
    observation.position[1] = 0.0f;
    observation.position[2] = 0.0f;
    observation.orientation[0] = 1.0f;
    observation.orientation[1] = 0.0f;
    observation.orientation[2] = 0.0f;
    observation.orientation[3] = 0.0f;
    observation.linear_velocity[0] = 0.0f;
    observation.linear_velocity[1] = 0.0f;
    observation.linear_velocity[2] = 0.0f;
    observation.angular_velocity[0] = 0.0f;
    observation.angular_velocity[1] = 0.0f;
    observation.angular_velocity[2] = 0.0f;
    for(uint j = 0; j < OUTPUT_DIM; j++){
        observation.previous_action[j] = 0.0f;
    }
    RLtoolsInferenceTimestamp timestamp = 0;
    RLtoolsInferenceApplicationL2FAction action;
    rl_tools_inference_applications_l2f_control(timestamp, &observation, &action);
}
