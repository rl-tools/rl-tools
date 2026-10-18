#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/spy/operations_cpu.h>
#include <metra/metra.h>

#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

namespace rlt = rl_tools;
namespace hd = rlt::rl::environments::hyperdrone;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = DEVICE::index_t;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
constexpr bool DYNAMIC_ALLOCATION = true;
constexpr TI SEED = 0;
struct WORLD_SPEC: hd::presets::X500FPV<T, TI> {
    static constexpr TI N_CAMERA_AGENTS = 1;
    static constexpr TI INSTANCES_PER_WORLD = 4;
    static constexpr TI CAM_WIDTH = 64;
    static constexpr TI CAM_HEIGHT = 64;
    static constexpr bool SELF_VISIBLE = false;
};
using WORLD = hd::World<WORLD_SPEC>;
using ENVIRONMENT = rlt::rl::environments::HyperDrone<WORLD, 1>;
using SPY = hd::Spy<hd::spy::Config<ENVIRONMENT>>;
template <typename ELEMENT, TI... DIMS>
using InstanceTensor = rlt::Tensor<rlt::tensor::Specification<ELEMENT, TI, rlt::tensor::Shape<TI, ENVIRONMENT::INSTANCES, DIMS...>, DYNAMIC_ALLOCATION>>;

int main(int argc, char** argv){
    const bool kitty = argc == 3 && std::string(argv[2]) == "--kitty";
    if((argc != 2 && !kitty) || !std::filesystem::is_regular_file(argv[1])){
        std::fprintf(stderr, "usage: %s /path/to/local/scene.glb [--kitty]\n", argv[0]);
        return 1;
    }
    using namespace rl_tools;
    DEVICE device{};
    ENVIRONMENT environment{};
    SPY spy{};
    RNG rng{};
    InstanceTensor<WORLD::Parameters> parameters{};
    InstanceTensor<WORLD::State> states{}, next_states{};
    InstanceTensor<bool> reset_mask{};
    InstanceTensor<T, WORLD::ACTION_DIM> actions{};
    InstanceTensor<T, WORLD::OBSERVATION_DIM> observations{};

    init(device);
    malloc(device, environment);
    malloc(device, rng);
    malloc(device, parameters);
    malloc(device, states);
    malloc(device, next_states);
    malloc(device, reset_mask);
    malloc(device, actions);
    malloc(device, observations);

    int result = 0;
    TI step_i = 0;
    try{
        rlt::hyperlux::datasets::glb::Dataset dataset{};
        dataset.references = {argv[1]};
        init(device, environment, dataset);
        init(device, rng, SEED);
        set_all(device, reset_mask, true);
        set_all(device, actions, (T)0);
        malloc(device, spy);
        init(device, spy, environment, kitty ? rlt::hyperlux::spy::ui::Frontend::KITTY : rlt::hyperlux::spy::ui::Frontend::WINDOW);
        while(true){
            sample_initial_parameters(device, environment, parameters, reset_mask, rng);
            sample_initial_state(device, environment, parameters, states, reset_mask, rng);
            observe(device, environment, parameters, states, WORLD::Observation{}, observations, rng);
            if(!update(device, spy, environment)){
                break;
            }
            step(device, environment, parameters, states, actions, next_states, rng);
            copy(device, device, next_states, states);
            terminated(device, environment, parameters, states, reset_mask, rng);
            if(++step_i % WORLD::EPISODE_STEP_LIMIT == 0){
                set_all(device, reset_mask, true);
            }
        }
    }
    catch(const std::exception& error){
        std::fprintf(stderr, "HyperDrone spy: %s\n", error.what());
        result = 1;
    }

    free(device, spy);
    free(device, observations);
    free(device, actions);
    free(device, reset_mask);
    free(device, next_states);
    free(device, states);
    free(device, parameters);
    free(device, rng);
    free(device, environment);
    free(device);
    metra::log("hyperdrone/spy/example/steps", double(step_i));
    return result;
}
