#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/operations_cuda.h>
#include "../collection.h"

class PPO_COLLECTION_CUDA: public ::testing::Test{
protected:
    rl_tools::devices::DefaultCUDA device;
    void SetUp() override{ using namespace rl_tools; init(device); }
    void TearDown() override{
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
        EXPECT_EQ(cudnnDestroy(device.cudnn_handle), CUDNN_STATUS_SUCCESS);
#endif
        EXPECT_EQ(cublasDestroy(device.handle), CUBLAS_STATUS_SUCCESS);
        EXPECT_EQ(cudaStreamDestroy(device.stream), cudaSuccess);
    }
};

TEST_F(PPO_COLLECTION_CUDA, RECURRENT_BOOTSTRAP){ rl_tools::test_ppo_collection::check<decltype(device), true, false>(device); }
TEST_F(PPO_COLLECTION_CUDA, RECURRENT_IGNORE_TERMINATION){ rl_tools::test_ppo_collection::check<decltype(device), true, true>(device); }
TEST_F(PPO_COLLECTION_CUDA, RECURRENT_NO_BOOTSTRAP){ rl_tools::test_ppo_collection::check<decltype(device), false, false>(device); }
TEST_F(PPO_COLLECTION_CUDA, RECURRENT_IGNORE_WITHOUT_TRUNCATION_BOOTSTRAP){ rl_tools::test_ppo_collection::check<decltype(device), false, true>(device); }
TEST_F(PPO_COLLECTION_CUDA, RECURRENT_FINAL_BOUNDARY){ rl_tools::test_ppo_collection::check<decltype(device), true, false, 4>(device); }
TEST_F(PPO_COLLECTION_CUDA, RECURRENT_ONE_STEP){ rl_tools::test_ppo_collection::check<decltype(device), true, false, 1>(device); }
TEST_F(PPO_COLLECTION_CUDA, RECURRENT_ACTOR_FEEDFORWARD_CRITIC){ rl_tools::test_ppo_collection::check<decltype(device), true, false, 5, false>(device); }
TEST_F(PPO_COLLECTION_CUDA, RECURRENT_ACTOR_FEEDFORWARD_CRITIC_NO_BOOTSTRAP){ rl_tools::test_ppo_collection::check<decltype(device), false, false, 5, false>(device); }
