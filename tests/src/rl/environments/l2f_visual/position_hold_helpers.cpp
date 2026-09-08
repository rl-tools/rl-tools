#include <rl_tools/operations/cpu.h>
#include <rl_tools/numeric_types/policy.h>
#include <rl_tools/rl/environments/l2f_visual/operations_generic_inertial_velocity.h>
#include <rl_tools/rl/environments/l2f_visual/operations_generic_level_camera.h>
#include <gtest/gtest.h>
#include <cmath>

using DEVICE = rl_tools::devices::DefaultCPU;
using BASE = rl_tools::rl::environments::l2f_visual::inertial_velocity::Specification<rl_tools::numeric_types::Policy<double>, DEVICE::index_t>;
struct SPEC: BASE{
    static constexpr double TIME_CONSTANT = 0;
};
TEST(L2F_VISUAL_POSITION_HOLD, INERTIAL_VELOCITY){
    using rl_tools::init;
    using rl_tools::step;
    DEVICE device;
    rl_tools::rl::environments::l2f_visual::inertial_velocity::State<SPEC> state;
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
    rl_tools::rl::environments::l2f_visual::inertial_velocity::State<BASE> leaky;
    init(device,leaky,gyro);
    leaky.velocity[2]=1;
    for(int i=0;i<500;i++){
        step(device,leaky,up,acceleration,gyro,0.01);
    }
    EXPECT_NEAR(leaky.velocity[2], std::exp(-1.0), 1e-10);
}

TEST(L2F_VISUAL_POSITION_HOLD, LEVEL_CAMERA_RAYS){
    using namespace rl_tools;
    devices::DefaultCPU device;
    using SPEC = rl::environments::l2f_visual::level_camera::Specification<numeric_types::Policy<double>, devices::DefaultCPU::index_t, 80, 50>;
    rl::environments::l2f_visual::level_camera::Camera<SPEC> camera;
    double maximum = 0;
    for(int sample = 0; sample < 100; sample++){
        const double roll = (sample%5-2)*.12;
        const double pitch = (sample%7-3)*.08;
        const double yaw = sample*.1;
        const double r = .02*(sample%3-1), p = .03*(sample%4-2), z = .01*(sample%5-2);
        auto rotation = [](double r, double p, double z, double out[3][3]){
            const double cr=std::cos(r),sr=std::sin(r),cp=std::cos(p),sp=std::sin(p),cz=std::cos(z),sz=std::sin(z);
            const double matrix[3][3] = {{cz*cp,cz*sp*sr-sz*cr,cz*sp*cr+sz*sr},{sz*cp,sz*sp*sr+cz*cr,sz*sp*cr-cz*sr},{-sp,cp*sr,cp*cr}};
            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++){
                    out[i][j]=matrix[i][j];
                }
            }
        };
        double attitude[3][3],mount[3][3],level[3][3];
        rotation(roll,pitch,yaw,attitude);
        rotation(r,p,z,mount);
        rotation(0,0,yaw,level);
        const double gravity[3]={attitude[2][0],attitude[2][1],attitude[2][2]};
        const double forward[3]={mount[0][0],mount[1][0],mount[2][0]};
        const double up[3]={mount[0][2],mount[1][2],mount[2][2]};
        init(device,camera,gravity,forward,up,79.6);
        for(int row=4;row<46;row+=7){
            for(int column=4;column<76;column+=7){
                double sx,sy;
                project(device,camera,double(column),double(row),sx,sy);
                const double target_ray[3]={1,-(column-39.5)/camera.focal,-(row-24.5)/camera.focal};
                const double source_ray[3]={1,-(sx-39.5)/camera.focal,-(sy-24.5)/camera.focal};
                double desired[3]={},actual[3]={};
                for(int world=0;world<3;world++){
                    for(int body=0;body<3;body++){
                        for(int channel=0;channel<3;channel++){
                            desired[world]+=level[world][body]*mount[body][channel]*target_ray[channel];
                            actual[world]+=attitude[world][body]*mount[body][channel]*source_ray[channel];
                        }
                    }
                }
                double a=0,d=0;
                for(int axis=0;axis<3;axis++){
                    a+=actual[axis]*actual[axis];
                    d+=desired[axis]*desired[axis];
                }
                for(int axis=0;axis<3;axis++){
                    maximum=std::max(maximum,std::abs(actual[axis]/std::sqrt(a)-desired[axis]/std::sqrt(d)));
                }
            }
        }
    }
    EXPECT_LT(maximum, 1e-12);
}
