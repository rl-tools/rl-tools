#include "../../../../../src/rl/environments/mujoco/ant/ppo/cpu/training.h"
#include <gtest/gtest.h>
TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, TRAINING_PPO){
    run<1, 100>();
}
