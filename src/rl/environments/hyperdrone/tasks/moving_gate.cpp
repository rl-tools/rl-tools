#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cpu.h>

#include "../demo_common.h"

#include <cstdio>
#include <string>
#include <vector>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

using DYNAMICS_STATIC_PARAMETERS = hyperdrone_demo::DynamicsStaticParameters<T, TI>;
struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, DYNAMICS_STATIC_PARAMETERS> {
    static constexpr TI INSTANCES_PER_ENVIRONMENT = 2;
    static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 8;
};
using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
using TASK_SPEC = rlt::rl::environments::hyperdrone::tasks::moving_gate::Specification<BASE_WORLD>;
using WORLD = rlt::rl::environments::hyperdrone::tasks::moving_gate::World<TASK_SPEC>;

constexpr TI INSTANCES = WORLD::INSTANCES;
constexpr TI OBSERVATION_DIM = BASE_WORLD::OBSERVATION_DIM;
constexpr TI CAM_WIDTH = WORLD_SPEC::CAM_WIDTH;
constexpr TI CAM_HEIGHT = WORLD_SPEC::CAM_HEIGHT;
constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
constexpr TI IMAGE_CHANNELS = BASE_WORLD::IMAGE_CHANNELS;
static_assert(IMAGE_CHANNELS == 3);
constexpr TI STEPS = 600;
constexpr TI EPISODE_TRUNCATION_INTERVAL = 150;
constexpr TI FRAMERATE = 100;
constexpr TI UPSCALE = 8;
constexpr T ACTION_EXCITATION_AMPLITUDE = 0.05;
constexpr T ACTION_EXCITATION_FREQUENCY = 0.5;

struct Tensors {
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, OBSERVATION_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> terminated_flags;
    void allocate(DEVICE& device){
        rlt::malloc(device, parameters);
        rlt::malloc(device, states);
        rlt::malloc(device, next_states);
        rlt::malloc(device, reset_mask);
        rlt::malloc(device, actions);
        rlt::malloc(device, observations);
        rlt::malloc(device, terminated_flags);
    }
    void deallocate(DEVICE& device){
        rlt::free(device, parameters);
        rlt::free(device, states);
        rlt::free(device, next_states);
        rlt::free(device, reset_mask);
        rlt::free(device, actions);
        rlt::free(device, observations);
        rlt::free(device, terminated_flags);
    }
};

int main(int argc, char** argv){
#ifdef RL_TOOLS_TEST_DATA_PATH
    std::string scene_path = std::string(HYPERDRONE_DEMO_STRINGIFY(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
    std::string gate_path = std::string(HYPERDRONE_DEMO_STRINGIFY(RL_TOOLS_TEST_DATA_PATH)) + "/x500.glb";
#else
    std::string scene_path = "";
    std::string gate_path = "";
#endif
    std::string output_path = "moving_gate.mp4";
    TI seed = 0;
    if(argc > 1){
        scene_path = argv[1];
    }
    if(argc > 2){
        gate_path = argv[2];
    }
    if(argc > 3){
        output_path = argv[3];
    }
    if(argc > 4){
        seed = std::stoul(argv[4]);
    }
    if(scene_path.empty() || gate_path.empty()){
        std::fprintf(stderr, "usage: %s <scene.glb> <gate.glb> [output.mp4] [seed]\n", argv[0]);
        return 1;
    }

    DEVICE device;
    rlt::init(device);
    WORLD world;
    typename BASE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    world.gate_asset_path = gate_path;
    shared.scene_set.paths = {scene_path};
    rlt::init(device, world, shared, 0, 1, 0);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);

    Tensors tensors;
    tensors.allocate(device);
    rlt::set_all(device, tensors.reset_mask, true);

    FILE* video_pipe = hyperdrone_demo::open_video_pipe(CAM_WIDTH, CAM_HEIGHT * INSTANCES, FRAMERATE, UPSCALE, output_path);
    if(video_pipe == nullptr){
        std::fprintf(stderr, "failed to open ffmpeg pipe for %s\n", output_path.c_str());
        return 1;
    }
    std::vector<std::uint8_t> frame(INSTANCES * CAM_PIXELS * 3);

    for(TI step_i = 0; step_i < STEPS; step_i++){
        rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
        rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
        rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
        rlt::observe(device, world, tensors.parameters, tensors.states, typename BASE_WORLD::Observation{}, tensors.observations, rng);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
                for(TI channel_i = 0; channel_i < 3; channel_i++){
                    frame[(instance_i * CAM_PIXELS + pixel_i) * 3 + channel_i] = hyperdrone_demo::quantize_pixel(rlt::get(device, tensors.observations, instance_i, pixel_i * 3 + channel_i));
                }
            }
        }
        std::fwrite(frame.data(), 1, frame.size(), video_pipe);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, instance_i);
            T excitation = ACTION_EXCITATION_AMPLITUDE * rlt::math::sin(device.math, (T)2 * rlt::math::PI<T> * ACTION_EXCITATION_FREQUENCY * (T)step_i / (T)FRAMERATE + (T)instance_i);
            for(TI action_i = 0; action_i < WORLD::ACTION_DIM; action_i++){
                T action = hyperdrone_demo::hover_action(device, instance_parameters.dynamics, action_i) + excitation;
                rlt::set(device, tensors.actions, action, instance_i, action_i);
            }
        }
        rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
        rlt::copy(device, device, tensors.next_states, tensors.states);
        rlt::terminated(device, world, tensors.parameters, tensors.states, tensors.terminated_flags, rng);
        rlt::copy(device, device, tensors.terminated_flags, tensors.reset_mask);
        if((step_i + 1) % EPISODE_TRUNCATION_INTERVAL == 0){
            rlt::set_all(device, tensors.reset_mask, true);
        }
    }

    bool video_ok = hyperdrone_demo::close_video_pipe(video_pipe);
    if(!video_ok){
        std::fprintf(stderr, "ffmpeg exited with an error for %s\n", output_path.c_str());
    }
    else{
        std::printf("wrote %s (%lu frames, %lux%lu upscaled %lux)\n", output_path.c_str(), (unsigned long)STEPS, (unsigned long)CAM_WIDTH, (unsigned long)(CAM_HEIGHT * INSTANCES), (unsigned long)UPSCALE);
    }

    tensors.deallocate(device);
    rlt::free(device, world);
    rlt::free(device, shared.library);
    rlt::free(device, rng);
    return video_ok ? 0 : 1;
}
