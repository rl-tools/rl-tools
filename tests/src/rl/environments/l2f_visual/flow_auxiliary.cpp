#include <rl_tools/operations/cpu.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/rl/environments/l2f_visual/operations_generic_flow_auxiliary.h>
#include <gtest/gtest.h>

using rl_tools::pool_flow_component;

TEST(L2F_VISUAL_FLOW_AUXILIARY, UNEVEN_CELLS_AND_CAMERA_LAYOUT){
    namespace rlt = rl_tools;
    using DEVICE = rlt::devices::DefaultCPU;
    using TI = DEVICE::index_t;
    using SPEC = rlt::rl::environments::l2f_visual::flow_auxiliary::Specification<rlt::numeric_types::Policy<double>, TI, 5, 7, 3>;
    DEVICE device;
    double flow[2 * SPEC::HEIGHT * SPEC::WIDTH * 2];
    for(TI camera = 0; camera < 2; camera++){
        for(TI y = 0; y < SPEC::HEIGHT; y++){
            for(TI x = 0; x < SPEC::WIDTH; x++){
                const TI offset = ((camera * SPEC::HEIGHT + y) * SPEC::WIDTH + x) * 2;
                flow[offset] = SPEC::WIDTH * (0.01 * x + 0.02 * y + 0.1 * camera);
                flow[offset + 1] = SPEC::HEIGHT * (-0.02 * x + 0.01 * y - 0.1 * camera);
            }
        }
    }
    const double centers_x[] = {0.5, 2.5, 5};
    const double centers_y[] = {0, 1.5, 3.5};
    for(TI camera = 0; camera < 2; camera++){
        for(TI y = 0; y < SPEC::GRID; y++){
            for(TI x = 0; x < SPEC::GRID; x++){
                const TI component = (y * SPEC::GRID + x) * 2;
                EXPECT_NEAR(pool_flow_component(device, SPEC{}, flow, camera, component), SPEC::SCALE * (0.01 * centers_x[x] + 0.02 * centers_y[y] + 0.1 * camera), 1e-12);
                EXPECT_NEAR(pool_flow_component(device, SPEC{}, flow, camera, component + 1), SPEC::SCALE * (-0.02 * centers_x[x] + 0.01 * centers_y[y] - 0.1 * camera), 1e-12);
            }
        }
    }
}

TEST(L2F_VISUAL_FLOW_AUXILIARY, CLIPS_PROJECTIONS_OUTSIDE_IMAGE){
    namespace rlt = rl_tools;
    using DEVICE = rlt::devices::DefaultCPU;
    using TI = DEVICE::index_t;
    using SPEC = rlt::rl::environments::l2f_visual::flow_auxiliary::Specification<rlt::numeric_types::Policy<float>, TI, 2, 3, 1>;
    DEVICE device;
    const float flow[] = {6, -4, 6, -4, 6, -4, 6, -4, 6, -4, 6, -4};
    EXPECT_FLOAT_EQ(pool_flow_component(device, SPEC{}, flow, 0, 0), SPEC::SCALE);
    EXPECT_FLOAT_EQ(pool_flow_component(device, SPEC{}, flow, 0, 1), -SPEC::SCALE);
}
