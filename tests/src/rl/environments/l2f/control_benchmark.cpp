#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <metra/metra.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <type_traits>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;
using DEVICE = rlt::devices::DefaultCPU;
using TI = DEVICE::index_t;
using T = float;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using FACTORY = l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI, 5>;

using Interface = l2f::parameters::ActionInterface;
template<Interface MODE>
struct StaticParameters: FACTORY::STATIC_PARAMETERS {
    static constexpr auto ACTION_INTERFACE = MODE;
    static constexpr TI N_SUBSTEPS = MODE == Interface::CTBR ? 10 : 1;
    using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_PD = l2f::StateCTBRController<l2f::StateSpecification<T, TI, STATE_BASE>>;
    using INNER_STATE = std::conditional_t<MODE == Interface::CTBR, STATE_PD, STATE_BASE>;
    using STATE_TYPE = l2f::StateRotors<l2f::StateRotorsSpecification<T, TI, false, l2f::StateRandomForce<l2f::StateSpecification<T, TI, INNER_STATE>>>>;
    using PARAMETERS_BASE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<FACTORY::PARAMETERS_SPEC>>>;
    using PARAMETERS_PD = l2f::ParametersCTBRController<l2f::ParametersSpecification<T, TI, PARAMETERS_BASE>>;
    using PARAMETERS = std::conditional_t<MODE == Interface::CTBR, PARAMETERS_PD, PARAMETERS_BASE>;
    static constexpr PARAMETERS PARAMETER_VALUES = [](){
        if constexpr(MODE == Interface::CTBR){
            constexpr typename PARAMETERS_PD::CTBRController pd = {
                T(0), T(1), {T(4), T(4), T(2)}, {T(216), T(216), T(96)},
                {T(5.76), T(5.76), T(1.6)}, {T(0.003), T(0.003), T(0.0012)}
            };
            return PARAMETERS{{{FACTORY::dynamics, {T(0.1)}, FACTORY::mdp}, FACTORY::disturbances}, pd};
        }
        else{
            return PARAMETERS{{FACTORY::dynamics, FACTORY::integration, FACTORY::mdp}, FACTORY::disturbances};
        }
    }();
    static constexpr T STATE_LIMIT_POSITION_X = 100000;
    static constexpr T STATE_LIMIT_POSITION_Y = 100000;
    static constexpr T STATE_LIMIT_POSITION_Z = 100000;
};
using ACTION = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4, false>>;
constexpr TI STEPS = 512;
constexpr TI REPEATS = 7;

template<typename STATE>
void hash_state(std::uint64_t& hash, const STATE& state) {
    const auto append = [&hash](const auto& values) {
        for(T value: values) {
            std::uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            for(unsigned int byte = 0; byte < sizeof(bits); byte++) {
                hash ^= (bits >> (byte * 8)) & 0xff;
                hash *= UINT64_C(1099511628211);
            }
        }
    };
    append(state.position);
    append(state.orientation);
    append(state.linear_velocity);
    append(state.angular_velocity);
    append(state.rpm);
    append(state.force);
    append(state.torque);
}

template<typename STATIC_PARAMETERS, bool VERIFY>
void measure(const std::string& label, TI count) {
    using ENVIRONMENT = rlt::rl::environments::Multirotor<l2f::Specification<T, TI, STATIC_PARAMETERS>>;
    constexpr bool IDEAL = std::is_same_v<typename ENVIRONMENT::ACTUATION, l2f::actuation::IdealRates>;
    using rlt::init;
    using rlt::initial_state;
    using rlt::step;
    using rlt::malloc;
    using rlt::free;
    using rlt::set;
    DEVICE device;
    init(device);
    ENVIRONMENT environment;
    init(device, environment);
    auto parameters = environment.parameters;
    std::vector<typename ENVIRONMENT::State> initial(count), states(count);
    std::vector<RNG> engines(count);
    std::vector<ACTION> commands(STEPS);
    for(auto& engine: engines){
        malloc(device, engine);
    }
    for(TI row = 0; row < count; row++) {
        initial_state(device, environment, parameters, initial[row]);
        initial[row].position[0] = T(row % 11) * T(0.01);
        initial[row].angular_velocity[2] = T(row % 7) * T(0.01);
        if constexpr(ENVIRONMENT::RATE_CONTROLLER){
            initial[row].previous_angular_velocity[2] = initial[row].angular_velocity[2];
        }
    }
    for(TI tick = 0; tick < STEPS; tick++) {
        const T phase = T(tick % 64) * T(6.283185307179586 / 64);
        if constexpr(IDEAL){
            set(commands[tick], 0, 0, T(2) * (1 + T(0.05) * std::sin(phase)) / parameters.ideal_rates.thrust_to_weight - T(1));
            set(commands[tick], 0, 1, T(0.2) * std::sin(phase) / parameters.ideal_rates.rate_limit[0]);
            set(commands[tick], 0, 2, T(0.2) * std::cos(phase) / parameters.ideal_rates.rate_limit[1]);
            set(commands[tick], 0, 3, T(0.1) / parameters.ideal_rates.rate_limit[2]);
        }
        else if constexpr(ENVIRONMENT::RATE_CONTROLLER){
            set(commands[tick], 0, 0, parameters.dynamics.hovering_throttle_relative * T(2) - T(1) + T(0.002) * std::sin(phase));
            set(commands[tick], 0, 1, T(0.2) * std::sin(phase) / parameters.ctbr_controller.rate_limit[0]);
            set(commands[tick], 0, 2, T(0.2) * std::cos(phase) / parameters.ctbr_controller.rate_limit[1]);
            set(commands[tick], 0, 3, T(0.1) / parameters.ctbr_controller.rate_limit[2]);
        }
        else{
            for(TI motor = 0; motor < 4; motor++) {
                const T perturbation = T(0.002) * std::sin(phase + T(motor) * T(1.5707963267948966));
                set(commands[tick], 0, motor, parameters.dynamics.hovering_throttle_relative * 2 - 1 + perturbation);
            }
        }
    }
    if constexpr(VERIFY){
        states = initial;
        for(TI row = 0; row < count; row++) {
            init(device, engines[row], TI(1337) + row);
        }
        std::uint64_t hash = UINT64_C(14695981039346656037);
        for(TI tick = 0; tick < STEPS; tick++) {
            for(TI row = 0; row < count; row++) {
                typename ENVIRONMENT::State next;
                step(device, environment, parameters, states[row], commands[tick], next, engines[row]);
                states[row] = next;
                hash_state(hash, states[row]);
            }
        }
        std::cout << label << " batch=" << count << " trajectory_hash=" << std::hex << hash << std::dec << '\n';
    }
    else{
        std::vector<double> samples;
        double checksum = 0;
        for(TI repeat = 0; repeat <= REPEATS; repeat++) {
            states = initial;
            for(TI row = 0; row < count; row++) {
                init(device, engines[row], TI(1337) + row);
            }
            const auto begin = std::chrono::steady_clock::now();
            for(TI tick = 0; tick < STEPS; tick++) {
                for(TI row = 0; row < count; row++) {
                    typename ENVIRONMENT::State next;
                    step(device, environment, parameters, states[row], commands[tick], next, engines[row]);
                    states[row] = next;
                }
            }
            const auto end = std::chrono::steady_clock::now();
            if(repeat > 0) {
                samples.push_back(std::chrono::duration<double, std::nano>(end - begin).count() / double(STEPS * count));
            }
            checksum = 0;
            for(const auto& state: states) {
                for(TI axis = 0; axis < 3; axis++) {
                    checksum += state.position[axis] + state.linear_velocity[axis] + state.angular_velocity[axis];
                }
                for(TI motor = 0; motor < 4; motor++) {
                    checksum += state.orientation[motor] + state.rpm[motor];
                }
            }
        }
        const std::string metric = "hyperdrone/control/physics/" + label + "/batch-" + std::to_string(count);
        metra::log(metric + "/ns_per_drone_step", samples);
        metra::log(metric + "/checksum", checksum);
        std::cout << label << " batch=" << count << " checksum=" << std::setprecision(17) << checksum << " ns_per_drone_step=";
        for(double sample: samples) {
            std::cout << ' ' << sample;
        }
        std::cout << '\n';
    }
    for(auto& engine: engines){
        free(device, engine);
    }
}

int main(){
    for(TI count: {TI(1), TI(32), TI(4096)}){
        measure<StaticParameters<Interface::DIRECT_MOTOR>, true>("motor", count);
        measure<StaticParameters<Interface::CTBR>, true>("ctbr_pd", count);
        measure<l2f::WithIdealRates<StaticParameters<Interface::DIRECT_MOTOR>>, true>("ctbr_ideal", count);
        measure<StaticParameters<Interface::DIRECT_MOTOR>, false>("motor", count);
        measure<StaticParameters<Interface::CTBR>, false>("ctbr_pd", count);
        measure<l2f::WithIdealRates<StaticParameters<Interface::DIRECT_MOTOR>>, false>("ctbr_ideal", count);
    }
}
