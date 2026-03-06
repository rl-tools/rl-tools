#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_REACHER_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_REACHER_OPERATIONS_CPU_H

#include "reacher.h"
#include "operations_generic.h"

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE&, rl::environments::Reacher<SPEC>& env){
        return "{}";
    }
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE&, rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters){
        std::string json = "{";
        json += "\"arena_size\":" + std::to_string(SPEC::PARAMETERS::ARENA_SIZE) + ",";
        json += "\"target_radius\":" + std::to_string(SPEC::PARAMETERS::TARGET_RADIUS);
        json += "}";
        return json;
    }
    template <typename DEVICE, typename SPEC, typename STATE_SPEC>
    std::string json(DEVICE&, rl::environments::Reacher<SPEC>& env, typename rl::environments::Reacher<SPEC>::Parameters& parameters, typename rl::environments::reacher::State<STATE_SPEC>& state){
        std::string json = "{";
        json += "\"x\":" + std::to_string(state.x) + ",";
        json += "\"y\":" + std::to_string(state.y) + ",";
        json += "\"target_x\":" + std::to_string(state.target_x) + ",";
        json += "\"target_y\":" + std::to_string(state.target_y);
        json += "}";
        return json;
    }
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE&, rl::environments::ReacherVisual<SPEC>& env){
        return "{}";
    }
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE&, rl::environments::ReacherVisual<SPEC>& env, typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters){
        std::string json = "{";
        json += "\"arena_size\":" + std::to_string(SPEC::PARAMETERS::ARENA_SIZE) + ",";
        json += "\"target_radius\":" + std::to_string(SPEC::PARAMETERS::TARGET_RADIUS);
        json += "}";
        return json;
    }
    template <typename DEVICE, typename SPEC, typename STATE_SPEC>
    std::string json(DEVICE&, rl::environments::ReacherVisual<SPEC>& env, typename rl::environments::ReacherVisual<SPEC>::Parameters& parameters, typename rl::environments::reacher::State<STATE_SPEC>& state){
        std::string json = "{";
        json += "\"x\":" + std::to_string(state.x) + ",";
        json += "\"y\":" + std::to_string(state.y) + ",";
        json += "\"target_x\":" + std::to_string(state.target_x) + ",";
        json += "\"target_y\":" + std::to_string(state.target_y);
        json += "}";
        return json;
    }

    template <typename DEVICE, typename SPEC>
    std::string get_ui(DEVICE& device, rl::environments::Reacher<SPEC>& env){
        std::string ui = R"RL_TOOLS_LITERAL(
export async function init(canvas, options){
    return {
        ctx: canvas.getContext('2d')
    }
}
export async function render(ui_state, parameters, state, action) {
    const ctx = ui_state.ctx;
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    ctx.clearRect(0, 0, w, h);

    const arenaSize = parameters.arena_size;
    const targetRadius = parameters.target_radius;
    const margin = w * 0.05;
    const drawSize = w - 2 * margin;

    function toCanvasX(x) { return margin + (x + arenaSize) / (2 * arenaSize) * drawSize; }
    function toCanvasY(y) { return margin + (y + arenaSize) / (2 * arenaSize) * drawSize; }
    function toCanvasLen(l) { return l / (2 * arenaSize) * drawSize; }

    // Arena background
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(margin, margin, drawSize, drawSize);
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    ctx.strokeRect(margin, margin, drawSize, drawSize);

    // Grid lines
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.5;
    for (let i = -arenaSize; i <= arenaSize; i += arenaSize / 4) {
        ctx.beginPath();
        ctx.moveTo(toCanvasX(i), margin);
        ctx.lineTo(toCanvasX(i), margin + drawSize);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(margin, toCanvasY(i));
        ctx.lineTo(margin + drawSize, toCanvasY(i));
        ctx.stroke();
    }

    // Target radius circle
    const targetCanvasR = toCanvasLen(targetRadius);
    ctx.beginPath();
    ctx.arc(toCanvasX(state.target_x), toCanvasY(state.target_y), targetCanvasR, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255, 100, 100, 0.2)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Target
    const targetR = w * 0.02;
    ctx.beginPath();
    ctx.arc(toCanvasX(state.target_x), toCanvasY(state.target_y), targetR, 0, 2 * Math.PI);
    ctx.fillStyle = '#e74c3c';
    ctx.fill();
    ctx.strokeStyle = '#c0392b';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Action arrow
    if (action && action.length >= 2) {
        const ax = toCanvasX(state.x);
        const ay = toCanvasY(state.y);
        const arrowScale = toCanvasLen(arenaSize * 0.3);
        const dx = action[0] * arrowScale;
        const dy = action[1] * arrowScale;
        const mag = Math.sqrt(dx * dx + dy * dy);
        if (mag > 1) {
            ctx.beginPath();
            ctx.moveTo(ax, ay);
            ctx.lineTo(ax + dx, ay + dy);
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.lineWidth = 2;
            ctx.stroke();

            const headLen = Math.min(mag * 0.3, w * 0.02);
            const angle = Math.atan2(dy, dx);
            ctx.beginPath();
            ctx.moveTo(ax + dx, ay + dy);
            ctx.lineTo(ax + dx - headLen * Math.cos(angle - Math.PI / 6), ay + dy - headLen * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(ax + dx - headLen * Math.cos(angle + Math.PI / 6), ay + dy - headLen * Math.sin(angle + Math.PI / 6));
            ctx.closePath();
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.fill();
        }
    }

    // Agent
    const agentR = w * 0.025;
    ctx.beginPath();
    ctx.arc(toCanvasX(state.x), toCanvasY(state.y), agentR, 0, 2 * Math.PI);
    ctx.fillStyle = '#3498db';
    ctx.fill();
    ctx.strokeStyle = '#2980b9';
    ctx.lineWidth = 1.5;
    ctx.stroke();
}
        )RL_TOOLS_LITERAL";
        return ui;
    }
    template <typename DEVICE, typename SPEC>
    std::string get_ui(DEVICE& device, rl::environments::ReacherVisual<SPEC>& env){
        rl::environments::Reacher<SPEC> reacher;
        return get_ui(device, reacher);
    }
    template <typename DEVICE, typename SPEC>
    std::string get_description(DEVICE& device, rl::environments::Reacher<SPEC>& env){
        return "2D point reacher: navigate an agent to a target position within a bounded arena.";
    }
    template <typename DEVICE, typename SPEC>
    std::string get_description(DEVICE& device, rl::environments::ReacherVisual<SPEC>& env){
        return "2D point reacher (visual): navigate an agent to a target position within a bounded arena using image observations.";
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
