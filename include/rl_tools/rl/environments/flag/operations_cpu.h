#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_FLAG_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_FLAG_OPERATIONS_CPU_H

#include "environment.h"
#include "operations_generic.h"

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE&, rl::environments::Flag<SPEC>& env, typename rl::environments::Flag<SPEC>::Parameters& parameters){
        using PARAMS = typename rl::environments::Flag<SPEC>::Parameters;
        std::string json = "{";
        json += "\"BOARD_SIZE\":" + std::to_string(PARAMS::BOARD_SIZE) + ",";
        json += "\"DT\":" + std::to_string(PARAMS::DT) + ",";
        json += "\"MAX_ACCELERATION\":" + std::to_string(PARAMS::MAX_ACCELERATION) + ",";
        json += "\"MAX_VELOCITY\":" + std::to_string(PARAMS::MAX_VELOCITY) + ",";
        json += "\"FLAG_DISTANCE_THRESHOLD\":" + std::to_string(PARAMS::FLAG_DISTANCE_THRESHOLD) + ",";
        json += "\"flag_position\":[" + std::to_string(parameters.flag_position[0]) + "," + std::to_string(parameters.flag_position[1]) + "]";
        json += "}";
        return json;
    }
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE&, rl::environments::Flag<SPEC>& env, typename rl::environments::Flag<SPEC>::Parameters& parameters, typename rl::environments::Flag<SPEC>::State& state){
        using TI = typename SPEC::TI;
        std::string json = "{";
        json += "\"position\":[" + std::to_string(state.position[0]) + "," + std::to_string(state.position[1]) + "],";
        json += "\"velocity\":[" + std::to_string(state.velocity[0]) + "," + std::to_string(state.velocity[1]) + "],";
        json += "\"state_machine\":" + std::to_string(static_cast<TI>(state.state_machine));
        json += "}";
        return json;
    }

    template <typename DEVICE, typename SPEC>
    std::string get_ui(DEVICE& device, rl::environments::Flag<SPEC>& env){
        // Implement the functions `export async function render(ui_state, parameters, state, action)` and `export async function init(canvas, parameters, options)` and `export` them so that they are available as ES6 imports
        // Please have a look at https://studio.rl.tools which helps you create render functions interactively
        std::string ui = R"RL_TOOLS_LITERAL(


export async function init(canvas, options) {
    return {
        ctx: canvas.getContext('2d'),
        trace: []
    }
}

export async function episode_init(ui_state, parameters) {
    // Clear previous traces
    ui_state.trace = [];
}

export async function render(ui_state, parameters, state, action) {
    const ctx = ui_state.ctx;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const canvasWidth = ctx.canvas.width;
    const scale = canvasWidth / parameters.BOARD_SIZE;

    const agentX = state.position[0] * scale;
    const agentY = state.position[1] * scale;

    // Record trace
    ui_state.trace.push({
        x: agentX,
        y: agentY,
        sm: state.state_machine
    });

//    // Draw board
//    ctx.strokeStyle = 'black';
//    ctx.lineWidth = 2;
//    ctx.strokeRect(0, 0, parameters.BOARD_SIZE * scale, parameters.BOARD_SIZE * scale);

    // Draw flag
    const flagX = parameters.flag_position[0] * scale;
    const flagY = parameters.flag_position[1] * scale;
    ctx.beginPath();
    ctx.arc(flagX, flagY, 5, 0, 2 * Math.PI);
    ctx.fillStyle = 'red';
    ctx.fill();
    ctx.stroke();

    // Transparent area around flag
    ctx.beginPath();
    ctx.arc(flagX, flagY, parameters.FLAG_DISTANCE_THRESHOLD * scale, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
    ctx.fill();

    // Draw origin
    ctx.beginPath();
    ctx.arc(0, 0, 5, 0, 2 * Math.PI);
    ctx.fillStyle = 'black';
    ctx.fill();
    ctx.stroke();

    // Transparent area around origin
    ctx.beginPath();
    ctx.arc(0, 0, parameters.FLAG_DISTANCE_THRESHOLD * scale, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.fill();

    // Determine current target and highlight it with the threshold radius
    let targetX, targetY;
    if (state.state_machine === 0) {
        // INITIAL: target is flag
        targetX = flagX;
        targetY = flagY;
    } else {
        // FLAG_VISITED or ORIGIN_VISITED: target is origin
        targetX = 0;
        targetY = 0;
    }
    ctx.beginPath();
    ctx.arc(targetX, targetY, parameters.FLAG_DISTANCE_THRESHOLD * scale, 0, 2 * Math.PI);
    ctx.strokeStyle = '#c93c52';
    ctx.lineWidth = 5;
    ctx.stroke();

    // Draw trace with different colors depending on state machine
    for (let i = 1; i < ui_state.trace.length; i++) {
        const prev = ui_state.trace[i-1];
        const curr = ui_state.trace[i];

        let color = 'gray';
        if (prev.sm === 1) color = 'purple';
        if (prev.sm === 2) color = 'orange';

        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(curr.x, curr.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 5;
        ctx.stroke();
    }

    // Draw agent
    ctx.beginPath();
    ctx.arc(agentX, agentY, 35, 0, 2 * Math.PI);
    ctx.fillStyle = '#7DB9B6';
    ctx.fill();

//    // Draw velocity vector
//    const velX = state.velocity[0] * scale * 0.5;
//    const velY = state.velocity[1] * scale * 0.5;
//    ctx.beginPath();
//    ctx.moveTo(agentX, agentY);
//    ctx.lineTo(agentX + velX, agentY + velY);
//    ctx.strokeStyle = 'blue';
//    ctx.lineWidth = 5;
//    ctx.stroke();
//
//    // Draw action vector (scaled by MAX_ACCELERATION)
//    const accX = action[0] * parameters.MAX_ACCELERATION/(parameters.BOARD_SIZE*3) * scale * 0.5;
//    const accY = action[1] * parameters.MAX_ACCELERATION/(parameters.BOARD_SIZE*3) * scale * 0.5;
//    ctx.beginPath();
//    ctx.moveTo(agentX, agentY);
//    ctx.lineTo(agentX + accX, agentY + accY);
//    ctx.strokeStyle = 'green';
//    ctx.lineWidth = 5;
//    ctx.stroke();

}


        )RL_TOOLS_LITERAL";
        return ui;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
