#include <rl_tools/operations/cpu.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/rl/environments/l2f/operations_generic_inertial_velocity.h>
#include <gtest/gtest.h>
#include <cmath>

using DEVICE = rl_tools::devices::DefaultCPU;
using BASE = rl_tools::rl::environments::l2f::inertial_velocity::Specification<rl_tools::numeric_types::Policy<double>, DEVICE::index_t>;
struct SPEC: BASE{
    static constexpr double TIME_CONSTANT = 0;
};
TEST(L2F_INERTIAL_VELOCITY, INERTIAL_VELOCITY){
    using rl_tools::init;
    using rl_tools::step;
    DEVICE device;
    rl_tools::rl::environments::l2f::inertial_velocity::State<SPEC> state;
    double gyro[3] = {}, up[3] = {0,0,1}, acceleration[3] = {0,0,SPEC::GRAVITY};
    init(device, state, gyro);
    for(int i=0;i<10000;i++){
        step(device, state, up, acceleration, gyro, 0.01);
    }
    for(double v:state.velocity){
        EXPECT_NEAR(v, 0, 1e-12);
    }
    acceleration[0]=2;
    for(int i=0;i<100;i++){
        step(device, state, up, acceleration, gyro, 0.01);
    }
    EXPECT_NEAR(state.velocity[0], 2, 1e-10);
    acceleration[0]=0;
    gyro[2]=0.5;
    state.previous_gyro[2]=0.5;
    for(int i=0;i<100;i++){
        step(device,state,up,acceleration,gyro,0.01);
    }
    double rotation_error=std::abs(state.velocity[0]-2*std::cos(0.5))+std::abs(state.velocity[1]+2*std::sin(0.5));
    EXPECT_LT(rotation_error, 1e-10);
    rl_tools::rl::environments::l2f::inertial_velocity::State<BASE> leaky;
    init(device,leaky,gyro);
    leaky.velocity[2]=1;
    for(int i=0;i<500;i++){
        step(device,leaky,up,acceleration,gyro,0.01);
    }
    EXPECT_NEAR(leaky.velocity[2], std::exp(-1.0), 1e-10);
}
