#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers/matrix/persist_code.h>
#include <rl_tools/numeric_types/categories.h>
#include <rl_tools/numeric_types/policy.h>

#include "environment_attitude_setpoint_ctbr.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;
using PARAMETER_POLICY = rlt::numeric_types::UseCase<rlt::numeric_types::categories::Parameter, float>;
using TYPE_POLICY = rlt::numeric_types::Policy<float, PARAMETER_POLICY>;
using ENVIRONMENT = typename rlt::rl::zoo::l2f::ENVIRONMENT_ATTITUDE_SETPOINT_CTBR_FACTORY<DEVICE, TYPE_POLICY, TI>::ENVIRONMENT;
using T = typename ENVIRONMENT::T;
using CTBR_CONTROLLER = typename ENVIRONMENT::Parameters::CTBRController;
using ACTION = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM, false>>;

constexpr TI STEPS = 160;
constexpr TI STEP_AT = 20;
constexpr TI SCORE_FROM = 40;
constexpr TI N_MOTOR_DELAY_SCALES = 4;
constexpr T MOTOR_DELAY_SCALES[N_MOTOR_DELAY_SCALES] = {(T)0.125, (T)0.25, (T)0.5, (T)1};
constexpr const char* MOTOR_DELAY_SCALE_LABELS[N_MOTOR_DELAY_SCALES] = {"delay_0p125", "delay_0p25", "delay_0p5", "delay_1"};
constexpr const char* MOTOR_DELAY_SCALE_PLOT_LABELS[N_MOTOR_DELAY_SCALES] = {"0.125x delay", "0.25x delay", "0.5x delay", "1x delay"};
constexpr T RATE_REFERENCES[3] = {(T)5, (T)5, (T)1};

struct Result{
    T score = 0;
    T mean_abs_error[3] = {0, 0, 0};
    T max_abs_error[3] = {0, 0, 0};
    T final_abs_error[3] = {0, 0, 0};
    bool invalid = false;
};

struct Candidate{
    CTBR_CONTROLLER controller;
    Result result;
    T motor_delay_scale = 1;
    T kp_scale = 1;
    T kd_scale = 1;
};

template <typename V>
V clamp_value(V value, V low, V high){
    return std::max(low, std::min(high, value));
}

bool finite_state(const typename ENVIRONMENT::State& state){
    bool finite = true;
    for(TI i = 0; i < 3; i++){
        finite = finite && std::isfinite(state.angular_velocity[i]);
        finite = finite && std::isfinite(state.previous_angular_velocity[i]);
    }
    for(TI i = 0; i < 4; i++){
        finite = finite && std::isfinite(state.orientation[i]);
        finite = finite && std::isfinite(state.rpm[i]);
    }
    return finite;
}

template <typename PARAMETERS>
void scale_motor_delay(PARAMETERS& parameters, T scale){
    for(TI rotor_i = 0; rotor_i < PARAMETERS::N; rotor_i++){
        parameters.dynamics.rotor_time_constants_rising[rotor_i] *= scale;
        parameters.dynamics.rotor_time_constants_falling[rotor_i] *= scale;
    }
}

template <typename PARAMETERS>
void set_rate_action(const PARAMETERS& parameters, TI axis, T rate_reference, ACTION& action){
    T thrust_range = parameters.ctbr_controller.thrust_max - parameters.ctbr_controller.thrust_min;
    T collective = parameters.dynamics.hovering_throttle_relative;
    T normalized_collective = thrust_range > (T)1e-6
        ? (T)2 * (collective - parameters.ctbr_controller.thrust_min) / thrust_range - (T)1
        : (T)0;
    rlt::set(action, 0, 0, clamp_value(normalized_collective, (T)-1, (T)1));
    for(TI axis_i = 0; axis_i < 3; axis_i++){
        T value = axis_i == axis ? rate_reference : (T)0;
        T limit = parameters.ctbr_controller.rate_limit[axis_i];
        T normalized = limit > (T)1e-6 ? value / limit : (T)0;
        rlt::set(action, 0, axis_i + 1, clamp_value(normalized, (T)-1, (T)1));
    }
}

Result evaluate_axis(DEVICE& device, ENVIRONMENT& env, const CTBR_CONTROLLER& controller, T motor_delay_scale, TI axis, T rate_reference){
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    typename ENVIRONMENT::Parameters parameters;
    typename ENVIRONMENT::State state;
    typename ENVIRONMENT::State next_state;
    ACTION action;

    rlt::initial_parameters(device, env, parameters);
    scale_motor_delay(parameters, motor_delay_scale);
    parameters.ctbr_controller = controller;
    rlt::initial_state(device, env, parameters, state);

    Result result;
    TI count = 0;
    for(TI step_i = 0; step_i < STEPS; step_i++){
        state.target_roll = 0;
        state.target_pitch = 0;
        state.target_yaw_rate = 0;
        state.target_thrust_g = 1;
        state.target_steps_remaining = STEPS + 1;

        T reference = step_i >= STEP_AT ? rate_reference : (T)0;
        set_rate_action(parameters, axis, reference, action);
        rlt::step(device, env, parameters, state, action, next_state, rng);

        if(!finite_state(next_state)){
            result.invalid = true;
            break;
        }

        T error = std::abs(next_state.angular_velocity[axis] - reference);
        if(step_i >= SCORE_FROM){
            result.mean_abs_error[axis] += error;
            result.max_abs_error[axis] = std::max(result.max_abs_error[axis], error);
            count++;
        }
        if(step_i == STEPS - 1){
            result.final_abs_error[axis] = error;
        }
        state = next_state;
    }
    rlt::free(device, rng);

    if(count > 0){
        result.mean_abs_error[axis] /= (T)count;
    }
    result.score =
        result.mean_abs_error[axis] +
        (T)0.25 * result.max_abs_error[axis] +
        (T)0.5 * result.final_abs_error[axis];
    if(result.invalid){
        result.score += (T)1e6;
    }
    return result;
}

Result evaluate_controller(DEVICE& device, ENVIRONMENT& env, const CTBR_CONTROLLER& controller, T motor_delay_scale){
    Result result;
    for(TI axis = 0; axis < 3; axis++){
        for(TI sign_i = 0; sign_i < 2; sign_i++){
            T sign = sign_i == 0 ? (T)1 : (T)-1;
            Result axis_result = evaluate_axis(device, env, controller, motor_delay_scale, axis, sign * RATE_REFERENCES[axis]);
            result.invalid = result.invalid || axis_result.invalid;
            result.score += axis_result.score;
            result.mean_abs_error[axis] += axis_result.mean_abs_error[axis] / (T)2;
            result.max_abs_error[axis] = std::max(result.max_abs_error[axis], axis_result.max_abs_error[axis]);
            result.final_abs_error[axis] += axis_result.final_abs_error[axis] / (T)2;
        }
    }
    return result;
}

CTBR_CONTROLLER scale_gains(CTBR_CONTROLLER controller, T kp_scale, T kd_scale){
    for(TI axis = 0; axis < 3; axis++){
        controller.kp[axis] *= kp_scale;
        controller.kd[axis] *= kd_scale;
    }
    return controller;
}

void print_candidate(const Candidate& candidate){
    const Result& result = candidate.result;
    const CTBR_CONTROLLER& controller = candidate.controller;
    std::cout
        << "score=" << std::setw(9) << result.score
        << " motor_delay_scale=" << std::setw(5) << candidate.motor_delay_scale
        << " kp_scale=" << std::setw(5) << candidate.kp_scale
        << " kd_scale=" << std::setw(5) << candidate.kd_scale
        << " mean_abs_rate=[" << result.mean_abs_error[0] << ", " << result.mean_abs_error[1] << ", " << result.mean_abs_error[2] << "]"
        << " kp=[" << controller.kp[0] << ", " << controller.kp[1] << ", " << controller.kp[2] << "]"
        << " kd=[" << controller.kd[0] << ", " << controller.kd[1] << ", " << controller.kd[2] << "]"
        << "\n";
}

void write_summary_header(std::ofstream& output){
    output
        << "motor_delay_scale,kp_scale,kd_scale,score,invalid,"
        << "mean_abs_rate_x,mean_abs_rate_y,mean_abs_rate_z,"
        << "max_abs_rate_x,max_abs_rate_y,max_abs_rate_z,"
        << "final_abs_rate_x,final_abs_rate_y,final_abs_rate_z,"
        << "kp_x,kp_y,kp_z,kd_x,kd_y,kd_z\n";
}

void write_summary_row(std::ofstream& output, const Candidate& candidate){
    const Result& result = candidate.result;
    const CTBR_CONTROLLER& controller = candidate.controller;
    output
        << candidate.motor_delay_scale << ","
        << candidate.kp_scale << ","
        << candidate.kd_scale << ","
        << result.score << ","
        << (result.invalid ? 1 : 0) << ","
        << result.mean_abs_error[0] << ","
        << result.mean_abs_error[1] << ","
        << result.mean_abs_error[2] << ","
        << result.max_abs_error[0] << ","
        << result.max_abs_error[1] << ","
        << result.max_abs_error[2] << ","
        << result.final_abs_error[0] << ","
        << result.final_abs_error[1] << ","
        << result.final_abs_error[2] << ","
        << controller.kp[0] << ","
        << controller.kp[1] << ","
        << controller.kp[2] << ","
        << controller.kd[0] << ","
        << controller.kd[1] << ","
        << controller.kd[2] << "\n";
}

void write_trace_csv(DEVICE& device, ENVIRONMENT& env, const Candidate candidates[N_MOTOR_DELAY_SCALES], const std::string& path){
    constexpr TI N_CASES = 6;
    const char* names[N_CASES] = {"roll_pos", "roll_neg", "pitch_pos", "pitch_neg", "yaw_pos", "yaw_neg"};
    TI axes[N_CASES] = {0, 0, 1, 1, 2, 2};
    T references[N_CASES] = {RATE_REFERENCES[0], -RATE_REFERENCES[0], RATE_REFERENCES[1], -RATE_REFERENCES[1], RATE_REFERENCES[2], -RATE_REFERENCES[2]};

    typename ENVIRONMENT::Parameters parameters[N_MOTOR_DELAY_SCALES];

    typename ENVIRONMENT::State states[N_MOTOR_DELAY_SCALES][N_CASES];
    typename ENVIRONMENT::State next_states[N_MOTOR_DELAY_SCALES][N_CASES];
    ACTION actions[N_MOTOR_DELAY_SCALES][N_CASES];
    RNG rngs[N_MOTOR_DELAY_SCALES][N_CASES];
    for(TI scale_i = 0; scale_i < N_MOTOR_DELAY_SCALES; scale_i++){
        rlt::initial_parameters(device, env, parameters[scale_i]);
        scale_motor_delay(parameters[scale_i], candidates[scale_i].motor_delay_scale);
        parameters[scale_i].ctbr_controller = candidates[scale_i].controller;
        for(TI case_i = 0; case_i < N_CASES; case_i++){
            rlt::malloc(device, rngs[scale_i][case_i]);
            rlt::init(device, rngs[scale_i][case_i], scale_i * N_CASES + case_i);
            rlt::initial_state(device, env, parameters[scale_i], states[scale_i][case_i]);
        }
    }

    std::ofstream output(path);
    output << std::setprecision(9);
    output << "time";
    for(TI scale_i = 0; scale_i < N_MOTOR_DELAY_SCALES; scale_i++){
        for(TI case_i = 0; case_i < N_CASES; case_i++){
            output
                << "," << MOTOR_DELAY_SCALE_LABELS[scale_i] << "/" << names[case_i] << "/reference"
                << "," << MOTOR_DELAY_SCALE_LABELS[scale_i] << "/" << names[case_i] << "/omega_x"
                << "," << MOTOR_DELAY_SCALE_LABELS[scale_i] << "/" << names[case_i] << "/omega_y"
                << "," << MOTOR_DELAY_SCALE_LABELS[scale_i] << "/" << names[case_i] << "/omega_z"
                << "," << MOTOR_DELAY_SCALE_LABELS[scale_i] << "/" << names[case_i] << "/error";
        }
    }
    output << "\n";

    for(TI step_i = 0; step_i < STEPS; step_i++){
        T time = (T)step_i * parameters[0].integration.dt;
        output << time;
        for(TI scale_i = 0; scale_i < N_MOTOR_DELAY_SCALES; scale_i++){
            for(TI case_i = 0; case_i < N_CASES; case_i++){
                auto& state = states[scale_i][case_i];
                state.target_roll = 0;
                state.target_pitch = 0;
                state.target_yaw_rate = 0;
                state.target_thrust_g = 1;
                state.target_steps_remaining = STEPS + 1;

                T reference = step_i >= STEP_AT ? references[case_i] : (T)0;
                set_rate_action(parameters[scale_i], axes[case_i], reference, actions[scale_i][case_i]);
                rlt::step(device, env, parameters[scale_i], state, actions[scale_i][case_i], next_states[scale_i][case_i], rngs[scale_i][case_i]);

                T error = next_states[scale_i][case_i].angular_velocity[axes[case_i]] - reference;
                output
                    << "," << reference
                    << "," << next_states[scale_i][case_i].angular_velocity[0]
                    << "," << next_states[scale_i][case_i].angular_velocity[1]
                    << "," << next_states[scale_i][case_i].angular_velocity[2]
                    << "," << error;
                state = next_states[scale_i][case_i];
            }
        }
        output << "\n";
    }

    for(TI scale_i = 0; scale_i < N_MOTOR_DELAY_SCALES; scale_i++){
        for(TI case_i = 0; case_i < N_CASES; case_i++){
            rlt::free(device, rngs[scale_i][case_i]);
        }
    }
}

std::string gnuplot_string(const std::string& value){
    std::string result = "\"";
    for(char c: value){
        if(c == '\\' || c == '"'){
            result += '\\';
        }
        result += c;
    }
    result += "\"";
    return result;
}

void write_pdf_plots(const std::string& output_prefix, const std::string& trace_path){
    constexpr TI N_CASES = 6;
    const char* names[N_CASES] = {"roll_pos", "roll_neg", "pitch_pos", "pitch_neg", "yaw_pos", "yaw_neg"};
    const char* titles[N_CASES] = {"Roll + step", "Roll - step", "Pitch + step", "Pitch - step", "Yaw + step", "Yaw - step"};
    TI axes[N_CASES] = {0, 0, 1, 1, 2, 2};

    const char* path_env = std::getenv("PATH");
    std::string path = std::string("/opt/homebrew/bin:/usr/local/bin:") + (path_env == nullptr ? "" : path_env);
    setenv("PATH", path.c_str(), 1);

    FILE* gnuplot = popen("gnuplot -persist", "w");
    if(gnuplot == nullptr){
        std::cerr << "Failed to start gnuplot\n";
        return;
    }

    std::string trace = gnuplot_string(trace_path);
    std::fprintf(gnuplot, "set datafile separator comma\n");
    std::fprintf(gnuplot, "set terminal pdfcairo noenhanced color size 10,6 font ',10'\n");
    std::fprintf(gnuplot, "set grid\n");
    std::fprintf(gnuplot, "set key outside right top vertical Left reverse samplen 2 spacing 1.2\n");
    std::fprintf(gnuplot, "set xlabel 'time [s]'\n");
    std::fprintf(gnuplot, "set ylabel 'angular rate [rad/s]'\n");

    for(TI case_i = 0; case_i < N_CASES; case_i++){
        std::string output = gnuplot_string(output_prefix + "_" + names[case_i] + ".pdf");
        TI reference_col = 2 + case_i * 5;

        std::fprintf(gnuplot, "set output %s\n", output.c_str());
        std::fprintf(gnuplot, "set title '%s'\n", titles[case_i]);
        std::fprintf(gnuplot, "plot %s using 1:%d with lines lw 3 lc rgb 'black' dt 2 title 'reference'", trace.c_str(), (int)reference_col);
        for(TI scale_i = 0; scale_i < N_MOTOR_DELAY_SCALES; scale_i++){
            TI omega_col = 2 + (scale_i * N_CASES + case_i) * 5 + 1 + axes[case_i];
            std::fprintf(gnuplot, ", \\\n     %s using 1:%d with lines lw 2 title '%s'", trace.c_str(), (int)omega_col, MOTOR_DELAY_SCALE_PLOT_LABELS[scale_i]);
        }
        std::fprintf(gnuplot, "\n");
    }
    std::fprintf(gnuplot, "set output\n");
    std::fflush(gnuplot);
    int status = pclose(gnuplot);
    if(status != 0){
        std::cerr << "gnuplot exited with status " << status << "\n";
    }
}

int main(int argc, char** argv){
    std::string output_prefix = argc > 1 ? argv[1] : "ctbr_rate_step";
    std::string summary_path = output_prefix + "_summary.csv";
    std::string trace_path = output_prefix + "_trace.csv";

    DEVICE device;
    rlt::init(device);

    ENVIRONMENT env;
    rlt::init(device, env);

    constexpr T GAIN_SCALES[] = {
        (T)0.015625,
        (T)0.03125,
        (T)0.0625,
        (T)0.125,
        (T)0.25,
        (T)0.5,
        (T)0.75,
        (T)1.0,
        (T)1.5,
        (T)2.0,
        (T)4.0,
        (T)8.0,
        (T)16.0,
        (T)32.0,
        (T)64.0
    };

    Candidate best_by_motor_delay_scale[N_MOTOR_DELAY_SCALES];
    for(TI scale_i = 0; scale_i < N_MOTOR_DELAY_SCALES; scale_i++){
        best_by_motor_delay_scale[scale_i].motor_delay_scale = MOTOR_DELAY_SCALES[scale_i];
        best_by_motor_delay_scale[scale_i].result.score = (T)1e9;
    }

    std::cout << std::setprecision(6);
    std::ofstream summary(summary_path);
    summary << std::setprecision(9);
    write_summary_header(summary);
    for(TI scale_i = 0; scale_i < N_MOTOR_DELAY_SCALES; scale_i++){
        std::cout << "\nMotor delay scale: " << MOTOR_DELAY_SCALES[scale_i] << "\n";
        for(T kp_scale: GAIN_SCALES){
            for(T kd_scale: GAIN_SCALES){
                Candidate candidate;
                candidate.motor_delay_scale = MOTOR_DELAY_SCALES[scale_i];
                candidate.kp_scale = kp_scale;
                candidate.kd_scale = kd_scale;
                candidate.controller = scale_gains(env.parameters.ctbr_controller, kp_scale, kd_scale);
                candidate.result = evaluate_controller(device, env, candidate.controller, candidate.motor_delay_scale);
                print_candidate(candidate);
                write_summary_row(summary, candidate);
                if(candidate.result.score < best_by_motor_delay_scale[scale_i].result.score){
                    best_by_motor_delay_scale[scale_i] = candidate;
                }
            }
        }
    }
    summary.close();
    write_trace_csv(device, env, best_by_motor_delay_scale, trace_path);
    write_pdf_plots(output_prefix, trace_path);

    std::cout << "\nBest angular-rate step candidates by motor delay scale:\n";
    bool invalid = false;
    for(TI scale_i = 0; scale_i < N_MOTOR_DELAY_SCALES; scale_i++){
        print_candidate(best_by_motor_delay_scale[scale_i]);
        invalid = invalid || best_by_motor_delay_scale[scale_i].result.invalid;
    }
    std::cout
        << "\nWrote summary CSV: " << summary_path << "\n"
        << "Wrote PlotJuggler trace CSV: " << trace_path << "\n"
        << "Wrote PDF plots: " << output_prefix << "_{roll_pos,roll_neg,pitch_pos,pitch_neg,yaw_pos,yaw_neg}.pdf\n";

    return invalid ? 1 : 0;
}
