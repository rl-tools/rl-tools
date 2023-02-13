#include <layer_in_c/operations/cpu.h>

#include <layer_in_c/rl/components/replay_buffer/operations_cpu.h>
#include <layer_in_c/rl/components/replay_buffer/persist.h>

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

namespace lic = layer_in_c;


TEST(LAYER_IN_C_RL_COMPONENTS_REPLAY_BUFFER, PERSISTENCE){
std::string replay_buffer_path = "test_cuda_replay_buffer.h5";
using DEVICE = lic::devices::DefaultCPU;
using DTYPE = float;
constexpr DEVICE::index_t OBSERVATION_DIM = 2;
constexpr DEVICE::index_t ACTION_DIM = 3;
constexpr DEVICE::index_t CAPACITY = 20;
using REPLAY_BUFFER_SPEC = lic::rl::components::replay_buffer::Specification<DTYPE, DEVICE::index_t, OBSERVATION_DIM, ACTION_DIM, CAPACITY>;
using REPLAY_BUFFER = lic::rl::components::ReplayBuffer<REPLAY_BUFFER_SPEC>;
DEVICE device;
REPLAY_BUFFER replay_buffer_1;
REPLAY_BUFFER replay_buffer_2;
auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
{
lic::malloc(device, replay_buffer_1);
lic::randn(device, replay_buffer_1.observations, rng);
lic::randn(device, replay_buffer_1.actions, rng);
lic::randn(device, replay_buffer_1.next_observations, rng);
lic::randn(device, replay_buffer_1.rewards, rng);

for(typename DEVICE::index_t row_i = 0; row_i < CAPACITY; row_i++){
bool terminated = lic::random::normal_distribution(DEVICE::SPEC::RANDOM(), (DTYPE)0, (DTYPE)1, rng) > 0.5;
bool truncated = terminated || lic::random::normal_distribution(DEVICE::SPEC::RANDOM(), (DTYPE)0, (DTYPE)1, rng) > 0.5;
set(replay_buffer_1.terminated, row_i, 0, terminated);
set(replay_buffer_1.truncated, row_i, 0, truncated);
}
set(replay_buffer_1.next_observations, 7, 0, 1337);
auto data_file = HighFive::File(replay_buffer_path, HighFive::File::Overwrite);
lic::save(device, replay_buffer_1, data_file.createGroup("replay_buffer"));
}
{
lic::malloc(device, replay_buffer_2);
auto data_file = HighFive::File(replay_buffer_path, HighFive::File::ReadOnly);
lic::load(device, replay_buffer_2, data_file.getGroup("replay_buffer"));
}
{
auto abs_diff = lic::abs_diff(device, replay_buffer_1, replay_buffer_2);
ASSERT_FLOAT_EQ(abs_diff, 0);
auto v = lic::view<DEVICE, typename decltype(replay_buffer_2.next_observations)::SPEC, 3, 2>(device, replay_buffer_2.next_observations, 6, 0);
lic::print(device, v);
}
{
lic::free(device, replay_buffer_1);
lic::free(device, replay_buffer_2);
}
}
