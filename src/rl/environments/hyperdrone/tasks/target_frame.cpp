#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>

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
    static constexpr TI HISTORY_LENGTH = 4;
    static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0.25;
};
using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
struct TASK_SPEC: rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<BASE_WORLD> {
    static constexpr TI IMAGE_STACK_N = 2;
    static constexpr TI IMAGE_STACK_STRIDE = 1;
    static constexpr T TARGET_FRAME_ROLL_PITCH_RANDOMIZATION_RANGE = 0.1;
    static constexpr T TARGET_FRAME_BRIGHTNESS_MISMATCH_RANGE = 0.25;
};
using WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
constexpr TI NUMBER_OF_ENVIRONMENTS = 2;
using ENVIRONMENT = rlt::rl::environments::hyperdrone::MultiEnvironment<WORLD, NUMBER_OF_ENVIRONMENTS>;

constexpr TI INSTANCES = ENVIRONMENT::INSTANCES;
constexpr TI CAM_WIDTH = WORLD_SPEC::CAM_WIDTH;
constexpr TI CAM_HEIGHT = WORLD_SPEC::CAM_HEIGHT;
constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
constexpr TI IMAGE_CHANNELS = BASE_WORLD::IMAGE_CHANNELS;
static_assert(IMAGE_CHANNELS == 3);
// video tiles: the stacked frames followed by the target frame, in observation channel order
constexpr TI TILES = TASK_SPEC::IMAGE_STACK_N + 1;
constexpr TI FRAME_WIDTH = TILES * CAM_WIDTH;
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
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM>>> observations;
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

int main(){
    const std::string output_path = "target_frame.mp4";
    constexpr TI SEED = 0;

    DEVICE device;
    rlt::init(device);
    // dataset configuration: one scene per World, from the conta-published ProcTHOR pair
    // (resolved from the content-addressed store, downloaded on first use)
    rlt::rendering::datasets::procthor::GLB dataset{{}, {
        "conta:a8fda3e9872e29994a425bd35c425159598623ce", // ai2thor-hab/glb/ProcTHOR-Train-0.glb
        "conta:7f1c9129532798e0b63bc41edb6b4c09251cf8a0", // ai2thor-hab/glb/ProcTHOR-Train-1.glb
    }};

    ENVIRONMENT env;
    rlt::malloc(device, env);
    // the annotation cache persists the free-space scans across runs
    env.shared.annotation_cache.directory = rlt::rendering::datasets::annotations::default_cache_directory();
    rlt::init(device, env, dataset);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, SEED);

    Tensors tensors;
    tensors.allocate(device);
    rlt::set_all(device, tensors.reset_mask, true);

    FILE* video_pipe = hyperdrone_demo::open_video_pipe(FRAME_WIDTH, CAM_HEIGHT * INSTANCES, FRAMERATE, UPSCALE, output_path);
    if(video_pipe == nullptr){
        std::fprintf(stderr, "failed to open ffmpeg pipe for %s\n", output_path.c_str());
        return 1;
    }
    std::vector<std::uint8_t> frame(INSTANCES * CAM_HEIGHT * FRAME_WIDTH * 3);

    for(TI step_i = 0; step_i < STEPS; step_i++){
        rlt::sample_initial_parameters(device, env, tensors.parameters, tensors.reset_mask, rng);
        rlt::sample_initial_state(device, env, tensors.parameters, tensors.states, tensors.reset_mask, rng);
        rlt::render(device, env, tensors.parameters, tensors.states, tensors.reset_mask);
        rlt::observe(device, env, tensors.parameters, tensors.states, typename ENVIRONMENT::Observation{}, tensors.observations, rng);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            for(TI y = 0; y < CAM_HEIGHT; y++){
                for(TI x = 0; x < CAM_WIDTH; x++){
                    TI pixel_i = y * CAM_WIDTH + x;
                    for(TI tile_i = 0; tile_i < TILES; tile_i++){
                        for(TI channel_i = 0; channel_i < 3; channel_i++){
                            T value = rlt::get(device, tensors.observations, instance_i, pixel_i * WORLD::OBSERVATION_CHANNELS + tile_i * IMAGE_CHANNELS + channel_i);
                            frame[((instance_i * CAM_HEIGHT + y) * FRAME_WIDTH + tile_i * CAM_WIDTH + x) * 3 + channel_i] = hyperdrone_demo::quantize_pixel(value);
                        }
                    }
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
        rlt::step(device, env, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
        rlt::copy(device, device, tensors.next_states, tensors.states);
        rlt::terminated(device, env, tensors.parameters, tensors.states, tensors.terminated_flags, rng);
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
        std::printf("wrote %s (%lu frames, %lux%lu upscaled %lux)\n", output_path.c_str(), (unsigned long)STEPS, (unsigned long)FRAME_WIDTH, (unsigned long)(CAM_HEIGHT * INSTANCES), (unsigned long)UPSCALE);
    }

    tensors.deallocate(device);
    rlt::free(device, env);
    rlt::free(device, rng);
    return video_ok ? 0 : 1;
}
