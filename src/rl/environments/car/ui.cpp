#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/rl/environments/car/operations_generic.h>
#include <backprop_tools/rl/environments/car/ui.h>
namespace bpt = backprop_tools;

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


    using DEV_SPEC = bpt::devices::DefaultCPUSpecification;
    using DEVICE = bpt::devices::CPU<DEV_SPEC>;
    using T = float;
    using TI = typename DEVICE::index_t;
    using ENV_SPEC = bpt::rl::environments::car::Specification<T, DEVICE::index_t>;
    using ENVIRONMENT = bpt::rl::environments::Car<ENV_SPEC>;

    using UI_SPEC = bpt::rl::environments::car::ui::Specification<T, TI, 200, 60>;
    using UI = bpt::rl::environments::car::UI<UI_SPEC>;

    DEVICE device;
    ENVIRONMENT env;
    ENVIRONMENT::State state, next_state;
    UI ui;
    auto rng = bpt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, 0);

    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    bpt::malloc(device, action);

    SDL_Event event;

    T color = 0;
    bool forward = true;

    bpt::init(device, env, ui);
    bpt::initial_state(device, env, state);
    T steering = 0, throttle = 0;
    bpt::initial_state(device, env, state);
    while(true){
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) return 0;
            if (event.type == SDL_JOYAXISMOTION) {
                if(event.jaxis.axis == 0){
                    steering = -event.jaxis.value / 32768.0 * 60.0/180.0*bpt::math::PI<T>;
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

        std::cout << "throttle " << throttle << " steering " << steering << std::endl;
        set(action, 0, 0, throttle);
        set(action, 0, 1, steering);
        bpt::step(device, env, state, action, next_state, rng);
        state = next_state;
        bpt::set_action(device, env, ui, action);
        bpt::set_state(device, env, ui, state);
        bpt::render(device, env, ui);
    }

    SDL_JoystickClose(joystick);
    SDL_Quit();
    return 0;
}


