#include <rl_tools/operations/cpu.h>

#include <rl_tools/rl/environments/car/operations_cpu.h>
#include <rl_tools/rl/environments/car/ui.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <SDL2/SDL.h>

int main(){
    if (SDL_Init(SDL_INIT_JOYSTICK) != 0) {
        fprintf(stderr, "Unable to initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    if (SDL_NumJoysticks() < 1) {
        printf("No joysticks connected!\n");
        return 1;
    }

    SDL_Joystick* joystick = SDL_JoystickOpen(0);
    if (joystick == NULL) {
        printf("Could not open joystick 0: %s\n", SDL_GetError());
        return 1;
    }

    printf("Opened joystick %s\n", SDL_JoystickName(joystick));


    using DEV_SPEC = rlt::devices::DefaultCPUSpecification;
    using DEVICE = rlt::devices::CPU<DEV_SPEC>;
    using T = float;
    using TI = typename DEVICE::index_t;
//    using ENV_SPEC = rlt::rl::environments::car::SpecificationTrack<T, DEVICE::index_t>;
    using ENV_SPEC = rlt::rl::environments::car::SpecificationTrack<T, DEVICE::index_t, 100, 100, 20>;
    using ENVIRONMENT = rlt::rl::environments::CarTrack<ENV_SPEC>;

    using UI_SPEC = rlt::rl::environments::car::ui::Specification<T, TI, ENVIRONMENT, 200, 60>;
    using UI = rlt::rl::environments::car::UI<UI_SPEC>;

    DEVICE device;
    ENVIRONMENT env;
    ENVIRONMENT::State state, next_state;
    UI ui;
    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 0);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    rlt::malloc(device, action);
    rlt::malloc(device, observation);

    SDL_Event event;

    T color = 0;
    bool forward = true;

    rlt::init(device, env);
    rlt::init(device, env, ui);
//    rlt::initial_state(device, env, state);
    T steering = 0, throttle = 0;
    rlt::sample_initial_state(device, env, state, rng);
    while(true){
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) return 0;
            if (event.type == SDL_JOYAXISMOTION) {
                if(event.jaxis.axis == 0){
                    steering = -event.jaxis.value / 32768.0 * 60.0/180.0*rlt::math::PI<T>;
                }
                if(event.jaxis.axis == 3){
                    throttle = -event.jaxis.value / 32768.0;
                }
//                printf("Joystick %d axis %d value: %d\n", event.jaxis.which, event.jaxis.axis, event.jaxis.value);
            }
//            if (event.type == SDL_JOYBUTTONDOWN || event.type == SDL_JOYBUTTONUP) {
//                printf("Joystick %d button %d state: %d\n", event.jbutton.which, event.jbutton.button, event.jbutton.state);
//            }
        }

//        std::cout << "throttle " << throttle << " steering " << steering << std::endl;
        set(action, 0, 0, throttle);
        set(action, 0, 1, steering);
        rlt::step(device, env, state, action, next_state, rng);
        state = next_state;
        rlt::set_action(device, env, ui, action);
        rlt::set_state(device, env, ui, state);
        rlt::render(device, env, ui);
//        std::cout << "terminated: " << rlt::terminated(device, env, state, rng) << std::endl;

        rlt::observe(device, env, state, observation, rng);
        std::cout << "lidar: " << get(observation, 0, 6) << ", " << get(observation, 0, 7) << ", " << get(observation, 0, 8) << std::endl;
        if(rlt::terminated(device, env, state, rng)){
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            rlt::sample_initial_state(device, env, state, rng);
        }

    }

    SDL_JoystickClose(joystick);
    SDL_Quit();
    return 0;
}


