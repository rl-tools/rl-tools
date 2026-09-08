#include "collection.h"

TEST(PPO_COLLECTION, RECURRENT_BOOTSTRAP){ rl_tools::devices::DefaultCPU device; rl_tools::test_ppo_collection::check<decltype(device), true, false>(device); }
TEST(PPO_COLLECTION, RECURRENT_IGNORE_TERMINATION){ rl_tools::devices::DefaultCPU device; rl_tools::test_ppo_collection::check<decltype(device), true, true>(device); }
TEST(PPO_COLLECTION, RECURRENT_NO_BOOTSTRAP){ rl_tools::devices::DefaultCPU device; rl_tools::test_ppo_collection::check<decltype(device), false, false>(device); }
TEST(PPO_COLLECTION, RECURRENT_IGNORE_WITHOUT_TRUNCATION_BOOTSTRAP){ rl_tools::devices::DefaultCPU device; rl_tools::test_ppo_collection::check<decltype(device), false, true>(device); }
TEST(PPO_COLLECTION, RECURRENT_FINAL_BOUNDARY){ rl_tools::devices::DefaultCPU device; rl_tools::test_ppo_collection::check<decltype(device), true, false, 4>(device); }
TEST(PPO_COLLECTION, RECURRENT_ONE_STEP){ rl_tools::devices::DefaultCPU device; rl_tools::test_ppo_collection::check<decltype(device), true, false, 1>(device); }
TEST(PPO_COLLECTION, RECURRENT_ACTOR_FEEDFORWARD_CRITIC){ rl_tools::devices::DefaultCPU device; rl_tools::test_ppo_collection::check<decltype(device), true, false, 5, false>(device); }
TEST(PPO_COLLECTION, RECURRENT_ACTOR_FEEDFORWARD_CRITIC_NO_BOOTSTRAP){ rl_tools::devices::DefaultCPU device; rl_tools::test_ppo_collection::check<decltype(device), false, false, 5, false>(device); }
