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
        json += "\"target_radius\":" + std::to_string(SPEC::PARAMETERS::TARGET_RADIUS) + ",";
        json += "\"image_height\":" + std::to_string(SPEC::PARAMETERS::IMAGE_HEIGHT) + ",";
        json += "\"image_width\":" + std::to_string(SPEC::PARAMETERS::IMAGE_WIDTH);
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

    namespace rl::environments::reacher{
        inline std::string get_ui_js(){
            return R"RL_TOOLS_LITERAL(
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
    const hasImage = (parameters.image_height !== undefined);
    const minimapSize = hasImage ? Math.min(w, h) * 0.3 : 0;
    const margin = hasImage ? w * 0.03 : w * 0.05;
    const mainSize = w - minimapSize - (hasImage ? w * 0.05 : 0);
    const drawSize = mainSize - 2 * margin;

    function toCanvasX(x) { return margin + (x + arenaSize) / (2 * arenaSize) * drawSize; }
    function toCanvasY(y) { return margin + (y + arenaSize) / (2 * arenaSize) * drawSize; }
    function toCanvasLen(l) { return l / (2 * arenaSize) * drawSize; }

    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(margin, margin, drawSize, drawSize);
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    ctx.strokeRect(margin, margin, drawSize, drawSize);

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

    const targetCanvasR = toCanvasLen(targetRadius);
    ctx.beginPath();
    ctx.arc(toCanvasX(state.target_x), toCanvasY(state.target_y), targetCanvasR, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255, 100, 100, 0.2)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255, 100, 100, 0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();

    const targetR = w * 0.02;
    ctx.beginPath();
    ctx.arc(toCanvasX(state.target_x), toCanvasY(state.target_y), targetR, 0, 2 * Math.PI);
    ctx.fillStyle = '#e74c3c';
    ctx.fill();
    ctx.strokeStyle = '#c0392b';
    ctx.lineWidth = 1.5;
    ctx.stroke();

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

    const agentR = w * 0.025;
    ctx.beginPath();
    ctx.arc(toCanvasX(state.x), toCanvasY(state.y), agentR, 0, 2 * Math.PI);
    ctx.fillStyle = '#3498db';
    ctx.fill();
    ctx.strokeStyle = '#2980b9';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    if (hasImage) {
        const imgH = parameters.image_height;
        const imgW = parameters.image_width;
        const showTarget = (state.step === undefined || state.step === 0);
        const mmX = mainSize + margin;
        const mmY = margin;
        const pixelSize = minimapSize / Math.max(imgH, imgW);
        const mmW = imgW * pixelSize;
        const mmH = imgH * pixelSize;
        const sigma_sq = (imgH * imgH) / 64.0;
        const agent_px = (state.x + arenaSize) / (2 * arenaSize) * imgH;
        const agent_py = (state.y + arenaSize) / (2 * arenaSize) * imgW;
        const target_px = (state.target_x + arenaSize) / (2 * arenaSize) * imgH;
        const target_py = (state.target_y + arenaSize) / (2 * arenaSize) * imgW;

        for (let row = 0; row < imgH; row++) {
            for (let col = 0; col < imgW; col++) {
                const dh_agent = row + 0.5 - agent_px;
                const dw_agent = col + 0.5 - agent_py;
                const agent_intensity = Math.exp(-(dh_agent * dh_agent + dw_agent * dw_agent) / sigma_sq);
                let target_intensity = 0;
                if (showTarget) {
                    const dh_target = row + 0.5 - target_px;
                    const dw_target = col + 0.5 - target_py;
                    target_intensity = Math.exp(-(dh_target * dh_target + dw_target * dw_target) / sigma_sq);
                }
                const r = Math.round(target_intensity * 255);
                const g = 0;
                const b = Math.round(agent_intensity * 255);
                ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
                ctx.fillRect(mmX + col * pixelSize, mmY + row * pixelSize, pixelSize, pixelSize);
            }
        }

        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.strokeRect(mmX, mmY, mmW, mmH);

        ctx.fillStyle = '#666';
        ctx.font = Math.round(w * 0.03) + 'px monospace';
        ctx.fillText(imgW + 'x' + imgH + ' obs' + (state.step !== undefined ? ' (step=' + state.step + ')' : ''), mmX, mmY + mmH + w * 0.04);
    }
}
            )RL_TOOLS_LITERAL";
        }
    }
    template <typename DEVICE, typename SPEC>
    std::string get_ui(DEVICE& device, rl::environments::Reacher<SPEC>& env){
        return rl::environments::reacher::get_ui_js();
    }
    template <typename DEVICE, typename SPEC>
    std::string get_ui(DEVICE& device, rl::environments::ReacherVisual<SPEC>& env){
        return rl::environments::reacher::get_ui_js();
    }
    template <typename DEVICE, typename SPEC>
    std::string get_description(DEVICE& device, rl::environments::Reacher<SPEC>& env){
        return "2D point reacher: navigate an agent to a target position within a bounded arena.";
    }
    template <typename DEVICE, typename SPEC>
    std::string get_description(DEVICE& device, rl::environments::ReacherVisual<SPEC>& env){
        return "2D point reacher (visual): navigate an agent to a target position within a bounded arena using image observations.";
    }
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE&, rl::environments::ReacherVisualMemory<SPEC>& env){
        return "{}";
    }
    template <typename DEVICE, typename SPEC>
    std::string json(DEVICE&, rl::environments::ReacherVisualMemory<SPEC>& env, typename rl::environments::ReacherVisualMemory<SPEC>::Parameters& parameters){
        std::string json = "{";
        json += "\"arena_size\":" + std::to_string(SPEC::PARAMETERS::ARENA_SIZE) + ",";
        json += "\"target_radius\":" + std::to_string(SPEC::PARAMETERS::TARGET_RADIUS) + ",";
        json += "\"image_height\":" + std::to_string(SPEC::PARAMETERS::IMAGE_HEIGHT) + ",";
        json += "\"image_width\":" + std::to_string(SPEC::PARAMETERS::IMAGE_WIDTH) + ",";
        json += "\"num_targets\":" + std::to_string(SPEC::PARAMETERS::NUM_TARGETS);
        json += "}";
        return json;
    }
    template <typename DEVICE, typename SPEC, typename STATE_SPEC>
    std::string json(DEVICE&, rl::environments::ReacherVisualMemory<SPEC>& env, typename rl::environments::ReacherVisualMemory<SPEC>::Parameters& parameters, typename rl::environments::reacher::StateSequentialTargets<STATE_SPEC>& state){
        std::string json = "{";
        json += "\"x\":" + std::to_string(state.x) + ",";
        json += "\"y\":" + std::to_string(state.y) + ",";
        json += "\"target1_x\":" + std::to_string(state.target1_x) + ",";
        json += "\"target1_y\":" + std::to_string(state.target1_y) + ",";
        json += "\"target2_x\":" + std::to_string(state.target2_x) + ",";
        json += "\"target2_y\":" + std::to_string(state.target2_y) + ",";
        json += "\"step\":" + std::to_string(state.step) + ",";
        json += "\"current_target\":" + std::to_string(state.current_target);
        json += "}";
        return json;
    }
    namespace rl::environments::reacher{
        inline std::string get_ui_memory_js(){
            return R"RL_TOOLS_LITERAL(
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
    const minimapSize = Math.min(w, h) * 0.3;
    const margin = w * 0.03;
    const mainSize = w - minimapSize - w * 0.05;
    const drawSize = mainSize - 2 * margin;

    function toCanvasX(x) { return margin + (x + arenaSize) / (2 * arenaSize) * drawSize; }
    function toCanvasY(y) { return margin + (y + arenaSize) / (2 * arenaSize) * drawSize; }
    function toCanvasLen(l) { return l / (2 * arenaSize) * drawSize; }

    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(margin, margin, drawSize, drawSize);
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    ctx.strokeRect(margin, margin, drawSize, drawSize);

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

    const targetCanvasR = toCanvasLen(targetRadius);
    const targetR = w * 0.02;

    // Target 1 (red)
    ctx.beginPath();
    ctx.arc(toCanvasX(state.target1_x), toCanvasY(state.target1_y), targetCanvasR, 0, 2 * Math.PI);
    ctx.fillStyle = state.current_target === 0 ? 'rgba(255, 100, 100, 0.2)' : 'rgba(255, 100, 100, 0.05)';
    ctx.fill();
    ctx.strokeStyle = state.current_target === 0 ? 'rgba(255, 100, 100, 0.5)' : 'rgba(255, 100, 100, 0.15)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(toCanvasX(state.target1_x), toCanvasY(state.target1_y), targetR, 0, 2 * Math.PI);
    ctx.fillStyle = state.current_target === 0 ? '#e74c3c' : '#e74c3c44';
    ctx.fill();

    // Target 2 (green) - only for multi-target
    if (parameters.num_targets > 1) {
        ctx.beginPath();
        ctx.arc(toCanvasX(state.target2_x), toCanvasY(state.target2_y), targetCanvasR, 0, 2 * Math.PI);
        ctx.fillStyle = state.current_target === 1 ? 'rgba(100, 255, 100, 0.2)' : 'rgba(100, 255, 100, 0.05)';
        ctx.fill();
        ctx.strokeStyle = state.current_target === 1 ? 'rgba(100, 255, 100, 0.5)' : 'rgba(100, 255, 100, 0.15)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(toCanvasX(state.target2_x), toCanvasY(state.target2_y), targetR, 0, 2 * Math.PI);
        ctx.fillStyle = state.current_target === 1 ? '#2ecc71' : '#2ecc7144';
        ctx.fill();
    }

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

    const agentR = w * 0.025;
    ctx.beginPath();
    ctx.arc(toCanvasX(state.x), toCanvasY(state.y), agentR, 0, 2 * Math.PI);
    ctx.fillStyle = '#3498db';
    ctx.fill();
    ctx.strokeStyle = '#2980b9';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    const imgH = parameters.image_height;
    const imgW = parameters.image_width;
    const showTargets = (state.step === 0);
    const mmX = mainSize + margin;
    const mmY = margin;
    const pixelSize = minimapSize / Math.max(imgH, imgW);
    const mmW = imgW * pixelSize;
    const mmH = imgH * pixelSize;
    const sigma_sq = (imgH * imgH) / 64.0;
    const agent_px = (state.x + arenaSize) / (2 * arenaSize) * imgH;
    const agent_py = (state.y + arenaSize) / (2 * arenaSize) * imgW;
    const target1_px = (state.target1_x + arenaSize) / (2 * arenaSize) * imgH;
    const target1_py = (state.target1_y + arenaSize) / (2 * arenaSize) * imgW;
    const hasTarget2 = parameters.num_targets > 1;
    const target2_px = hasTarget2 ? (state.target2_x + arenaSize) / (2 * arenaSize) * imgH : 0;
    const target2_py = hasTarget2 ? (state.target2_y + arenaSize) / (2 * arenaSize) * imgW : 0;

    for (let row = 0; row < imgH; row++) {
        for (let col = 0; col < imgW; col++) {
            const dh_agent = row + 0.5 - agent_px;
            const dw_agent = col + 0.5 - agent_py;
            const agent_intensity = Math.exp(-(dh_agent * dh_agent + dw_agent * dw_agent) / sigma_sq);
            let target1_intensity = 0;
            let target2_intensity = 0;
            if (showTargets) {
                const dh_t1 = row + 0.5 - target1_px;
                const dw_t1 = col + 0.5 - target1_py;
                target1_intensity = Math.exp(-(dh_t1 * dh_t1 + dw_t1 * dw_t1) / sigma_sq);
                if (hasTarget2) {
                    const dh_t2 = row + 0.5 - target2_px;
                    const dw_t2 = col + 0.5 - target2_py;
                    target2_intensity = Math.exp(-(dh_t2 * dh_t2 + dw_t2 * dw_t2) / sigma_sq);
                }
            }
            const r = Math.round(target1_intensity * 255);
            const g = Math.round(target2_intensity * 255);
            const b = Math.round(agent_intensity * 255);
            ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
            ctx.fillRect(mmX + col * pixelSize, mmY + row * pixelSize, pixelSize, pixelSize);
        }
    }

    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    ctx.strokeRect(mmX, mmY, mmW, mmH);

    ctx.fillStyle = '#666';
    ctx.font = Math.round(w * 0.03) + 'px monospace';
    ctx.fillText(imgW + 'x' + imgH + ' obs (step=' + state.step + (parameters.num_targets > 1 ? ' target=' + (state.current_target === 0 ? 'red' : 'green') : '') + ')', mmX, mmY + mmH + w * 0.04);
}
            )RL_TOOLS_LITERAL";
        }
    }
    template <typename DEVICE, typename SPEC>
    std::string get_ui(DEVICE& device, rl::environments::ReacherVisualMemory<SPEC>& env){
        return rl::environments::reacher::get_ui_memory_js();
    }
    template <typename DEVICE, typename SPEC>
    std::string get_description(DEVICE& device, rl::environments::ReacherVisualMemory<SPEC>& env){
        if(SPEC::PARAMETERS::NUM_TARGETS > 1){
            return "2D point reacher (visual memory): " + std::to_string(SPEC::PARAMETERS::NUM_TARGETS) + " targets shown at step 0, agent must visit them sequentially.";
        }
        return "2D point reacher (visual memory): target shown only on first step, agent must remember target location.";
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
