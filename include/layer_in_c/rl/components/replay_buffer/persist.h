#include "replay_buffer.h"

#include <highfive/H5Group.hpp>
#include <vector>

namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void save(DEVICE& device, rl::components::ReplayBuffer<SPEC>& rb, HighFive::Group group) {
        save(device, rb.observations, group, "observations");
        save(device, rb.actions, group, "actions");
        save(device, rb.rewards, group, "rewards");
        save(device, rb.next_observations, group, "next_observations");
        {
            lic::Matrix<lic::matrix::Specification<typename SPEC::T, typename DEVICE::index_t, 1, SPEC::CAPACITY>> terminated;
            lic::malloc(device, terminated);
            copy(device, device, terminated, rb.terminated);
            save(device, terminated, group, "terminated");
            lic::free(device, terminated);
        }
        {
            lic::Matrix<lic::matrix::Specification<typename SPEC::T, typename DEVICE::index_t, 1, SPEC::CAPACITY>> truncated;
            lic::malloc(device, truncated);
            copy(device, device, truncated, rb.truncated);
            save(device, truncated, group, "truncated");
            lic::free(device, truncated);
        }

        std::vector<typeof(rb.position)> position;
        position.push_back(rb.position);
        group.createDataSet("position", position);

        std::vector<typeof(rb.position)> full;
        position.push_back(rb.full);
        group.createDataSet("full", full);
    }
}
