#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_MULTI_AGENT_BOTTLENECK_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_MULTI_AGENT_BOTTLENECK_OPERATIONS_CPU_H

#include "bottleneck.h"
#include "operations_generic.h"

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE&, rl::environments::multi_agent::Bottleneck<SPEC>& env, const typename rl::environments::multi_agent::Bottleneck<SPEC>::Parameters& parameters){
        std::string json = "{";
        json += "\"N_AGENTS\":" + std::to_string(SPEC::PARAMETERS::N_AGENTS) + ",";
        json += "\"LIDAR_RESOLUTION\":" + std::to_string(SPEC::PARAMETERS::LIDAR_RESOLUTION) + ",";
        json += "\"LIDAR_FOV\":" + std::to_string(SPEC::PARAMETERS::LIDAR_FOV) + ",";
        json += "\"LIDAR_RANGE\":" + std::to_string(SPEC::PARAMETERS::LIDAR_RANGE) + ",";
        json += "\"DT\":" + std::to_string(SPEC::PARAMETERS::DT) + ",";
        json += "\"ARENA_WIDTH\":" + std::to_string(SPEC::PARAMETERS::ARENA_WIDTH) + ",";
        json += "\"ARENA_HEIGHT\":" + std::to_string(SPEC::PARAMETERS::ARENA_HEIGHT) + ",";
        json += "\"AGENT_DIAMETER\":" + std::to_string(SPEC::PARAMETERS::AGENT_DIAMETER) + ",";
        json += "\"AGENT_MAX_SPEED\":" + std::to_string(SPEC::PARAMETERS::AGENT_MAX_SPEED) + ",";
        json += "\"AGENT_MAX_ACCELERATION\":" + std::to_string(SPEC::PARAMETERS::AGENT_MAX_ACCELERATION) + ",";
        json += "\"AGENT_MAX_ANGULAR_VELOCITY\":" + std::to_string(SPEC::PARAMETERS::AGENT_MAX_ANGULAR_VELOCITY) + ",";
        json += "\"AGENT_MAX_ANGULAR_ACCELERATION\":" + std::to_string(SPEC::PARAMETERS::AGENT_MAX_ANGULAR_ACCELERATION) + ",";
        json += "\"BOTTLENECK_POSITION\":" + std::to_string(SPEC::PARAMETERS::BOTTLENECK_POSITION) + ",";
        json += "\"BOTTLENECK_WIDTH\":" + std::to_string(SPEC::PARAMETERS::BOTTLENECK_WIDTH) + ",";
        json += "\"BARRIER_WIDTH\":" + std::to_string(SPEC::PARAMETERS::BARRIER_WIDTH);
        json += "}";
        return json;
    }
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE&, rl::environments::multi_agent::Bottleneck<SPEC>& env, const typename rl::environments::multi_agent::Bottleneck<SPEC>::Parameters& parameters, const typename rl::environments::multi_agent::Bottleneck<SPEC>::State& state){
        using TI = typename DEVICE::index_t;
        std::string agent_states = "[";
        for(TI agent_i = 0; agent_i < SPEC::PARAMETERS::N_AGENTS; agent_i++){
            std::string agent_state = "{";
            agent_state += "\"position\": [" + std::to_string(state.agent_states[agent_i].position[0]) + "," + std::to_string(state.agent_states[agent_i].position[1]) + "],";
            agent_state += "\"orientation\": " + std::to_string(state.agent_states[agent_i].orientation) + ",";
            agent_state += "\"velocity\": [" + std::to_string(state.agent_states[agent_i].velocity[0]) + "," + std::to_string(state.agent_states[agent_i].velocity[1]) + "],";
            agent_state += "\"angular_velocity\": " + std::to_string(state.agent_states[agent_i].angular_velocity);
            agent_state += "}";
            agent_states += agent_state;
            if(agent_i < parameters.N_AGENTS - 1){
                agent_states += ",";
            }
        }
        agent_states += "]";
        std::string json = "{";
        json += "\"agent_states\": " + agent_states;
        json += "}";
        return json;
    }

    template <typename DEVICE, typename SPEC>
    std::string get_ui(DEVICE& device, rl::environments::multi_agent::Bottleneck<SPEC>& env){
        // just the body of `function render(ctx, state, action) {` (so that it can be easily processed by `new Function("ctx", "state", "action", body)`
        std::string ui = R"RL_TOOLS_LITERAL(
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    const canvasWidth = ctx.canvas.width;
    const canvasHeight = ctx.canvas.height;
    const scaleX = canvasWidth / parameters.ARENA_WIDTH;
    const scaleY = canvasHeight / parameters.ARENA_HEIGHT;
    // Draw the bottleneck barrier
    const barrierX = (parameters.ARENA_WIDTH / 2 - parameters.BARRIER_WIDTH / 2) * scaleX;
    const barrierWidth = parameters.BARRIER_WIDTH * scaleX;
    const bottleneckTopY = (parameters.BOTTLENECK_POSITION - parameters.BOTTLENECK_WIDTH / 2) * scaleY;
    const bottleneckBottomY = (parameters.BOTTLENECK_POSITION + parameters.BOTTLENECK_WIDTH / 2) * scaleY;
    ctx.fillStyle = 'gray';
    ctx.fillRect(barrierX, 0, barrierWidth, bottleneckTopY);
    ctx.fillRect(barrierX, bottleneckBottomY, barrierWidth, canvasHeight - bottleneckBottomY);

    const agentRadius = parameters.AGENT_DIAMETER * scaleX / 2;

    // Draw agents and their actions
    for (let i = 0; i < parameters.N_AGENTS; i++) {
        const agent = state.agent_states[i];
        const posX = agent.position[0] * scaleX;
        const posY = agent.position[1] * scaleY;
        const orientation = agent.orientation;

        // Draw agent body
        ctx.beginPath();
        ctx.arc(posX, posY, agentRadius, 0, 2 * Math.PI);
        ctx.fillStyle = 'blue';
        ctx.fill();
        ctx.stroke();

        // Draw agent orientation
        const endX = posX + agentRadius * Math.cos(orientation);
        const endY = posY + agentRadius * Math.sin(orientation);
        ctx.beginPath();
        ctx.moveTo(posX, posY);
        ctx.lineTo(endX, endY);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw actions (acceleration vectors)
        const agent_action = action[i];

        // Linear acceleration in the direction of orientation
        const accelMagnitude = agent_action[0] * scaleX;
        const accelX = accelMagnitude * Math.cos(orientation);
        const accelY = accelMagnitude * Math.sin(orientation);
        ctx.beginPath();
        ctx.moveTo(posX, posY);
        ctx.lineTo(posX + accelX, posY + accelY);
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw arrowhead for linear acceleration
        const angle = Math.atan2(accelY, accelX);
        const headlen = 10;
        ctx.beginPath();
        ctx.moveTo(posX + accelX, posY + accelY);
        ctx.lineTo(posX + accelX - headlen * Math.cos(angle - Math.PI / 6), posY + accelY - headlen * Math.sin(angle - Math.PI / 6));
        ctx.moveTo(posX + accelX, posY + accelY);
        ctx.lineTo(posX + accelX - headlen * Math.cos(angle + Math.PI / 6), posY + accelY - headlen * Math.sin(angle + Math.PI / 6));
        ctx.stroke();

        // Draw circular arrow for angular acceleration
        const angularAccel = Math.max(-1, Math.min(1, agent_action[1])); // Negative sign to match the canvas coordinate system
        const direction = Math.sign(angularAccel);
        const arrowRadius = agentRadius * 1.5;
        const arrowAngle = Math.PI / 3;
        const startAngle = orientation;
        const endAngle = orientation + arrowAngle * angularAccel;
        const arrowHeadSize = 10 * Math.abs(angularAccel);

        ctx.beginPath();
        ctx.arc(posX, posY, arrowRadius, startAngle, endAngle, angularAccel < 0);
        ctx.strokeStyle = 'green';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw arrowhead for angular acceleration
        const arrowHeadAngle = endAngle - direction* arrowAngle * 0.05 - Math.PI / 2;
        const arrowHeadX = posX + arrowRadius * Math.cos(endAngle);
        const arrowHeadY = posY + arrowRadius * Math.sin(endAngle);

        ctx.beginPath();
        ctx.moveTo(arrowHeadX, arrowHeadY);
        ctx.lineTo(
            arrowHeadX + direction * arrowHeadSize * Math.cos(arrowHeadAngle - Math.PI / 6),
            arrowHeadY + direction * arrowHeadSize * Math.sin(arrowHeadAngle - Math.PI / 6)
        );
        ctx.moveTo(arrowHeadX, arrowHeadY);
        ctx.lineTo(
            arrowHeadX + direction * arrowHeadSize * Math.cos(arrowHeadAngle + Math.PI / 6),
            arrowHeadY + direction * arrowHeadSize * Math.sin(arrowHeadAngle + Math.PI / 6)
        );
        ctx.strokeStyle = 'green';
        ctx.lineWidth = 2;
        ctx.stroke();
    }
        )RL_TOOLS_LITERAL";
        return ui;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
