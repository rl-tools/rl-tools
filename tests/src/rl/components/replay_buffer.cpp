#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/rl/components/replay_buffer/operations_cpu.h>
#include <backprop_tools/rl/components/replay_buffer/persist.h>

#include "replay_buffer.h"


#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

namespace bpt = backprop_tools;


TEST(BACKPROP_TOOLS_RL_COMPONENTS_REPLAY_BUFFER, PERSISTENCE) {
    std::string replay_buffer_path = "test_cuda_replay_buffer.h5";
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    constexpr DEVICE::index_t OBSERVATION_DIM = 2;
    constexpr DEVICE::index_t ACTION_DIM = 3;
    constexpr DEVICE::index_t CAPACITY = 20;
    using REPLAY_BUFFER_SPEC = bpt::rl::components::replay_buffer::Specification<DTYPE, DEVICE::index_t, OBSERVATION_DIM, ACTION_DIM, CAPACITY>;
    using REPLAY_BUFFER = bpt::rl::components::ReplayBuffer<REPLAY_BUFFER_SPEC>;
    DEVICE device;
    REPLAY_BUFFER replay_buffer_1;
    REPLAY_BUFFER replay_buffer_2;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    {
        bpt::malloc(device, replay_buffer_1);
        bpt::test::rl::components::replay_buffer::sample(device, replay_buffer_1, rng);
        set(replay_buffer_1.next_observations, 7, 0, 1337);
        auto data_file = HighFive::File(replay_buffer_path, HighFive::File::Overwrite);
        bpt::save(device, replay_buffer_1, data_file.createGroup("replay_buffer"));
    }
    {
        bpt::malloc(device, replay_buffer_2);
        auto data_file = HighFive::File(replay_buffer_path, HighFive::File::ReadOnly);
        bpt::load(device, replay_buffer_2, data_file.getGroup("replay_buffer"));
    }
    {
        auto abs_diff = bpt::abs_diff(device, replay_buffer_1, replay_buffer_2);
        ASSERT_FLOAT_EQ(abs_diff, 0);
        auto v = bpt::view<DEVICE, typename decltype(replay_buffer_2.next_observations)::SPEC, 3, 2>(device, replay_buffer_2.next_observations, 6, 0);
        bpt::print(device, v);
    }
    {
        bpt::free(device, replay_buffer_1);
        bpt::free(device, replay_buffer_2);
    }
}
