import * as THREE from "three"
import { L2F } from "./l2f.js"
import { SimControls } from "./sim_controls.js";
import { ParameterManager } from "./parameter_manager.js";
import * as rlt from "./dyn_inference_wrapper.js"
import { Gamepad } from "./gamepad.js"
import { GamepadController } from "./gamepad_controller.js"
import { AttitudeSetpointGamepadInput, ATTITUDE_SETPOINT_GAMEPAD_INTERFACE, ATTITUDE_SETPOINT_GAMEPAD_STORAGE_KEY } from "./attitude_setpoint_gamepad.js"
import { Position } from "./trajectories/position.js"
import { Lissajous } from "./trajectories/lissajous.js"
import { SecondOrderLangevin } from "./trajectories/langevin.js"
import { PingPong } from "./trajectories/ping_pong.js"
// import Controller from  "./controller.js"

fetch("./git_hash.txt", { cache: "no-store" })
    .then(async (response) => {
        if(!response.ok){
            return "local"
        }
        return (await response.text()).trim() || "local"
    })
    .catch(() => "local")
    .then(commit => console.log(`RLtools commit: ${commit}`))

// check url for "file" parameter
const urlParams = new URLSearchParams(window.location.search);
const file = urlParams.get('file');
const file_url = file ? file : "./external/blob/checkpoint.h5"

let proxy_controller = null
let l2f = null
let direct_gamepad = null
let direct_gamepad_controller = null
let attitude_setpoint_gamepad = null
let attitude_setpoint_input = null

const DIRECT_GAMEPAD_INTERFACE = {
    "thrust": {
        type: "axis",
        positive_direction: "Up"
    },
    "roll": {
        type: "axis",
        positive_direction: "Right"
    },
    "pitch": {
        type: "axis",
        positive_direction: "Forward"
    },
    "yaw": {
        type: "axis",
        positive_direction: "Clockwise"
    },
    "reset": {
        type: "button"
    },
}

async function sleep(ms){
    return new Promise(resolve => setTimeout(resolve, ms));
}

function observation_includes_attitude_setpoint(obs_string){
    if(!obs_string) return false
    return obs_string.split(";").flatMap(branch => branch.split(".")).some(component => component.trim() === "AttitudeSetpoint")
}

function selected_controller_mode(){
    return document.querySelector('#controller-selector-container input[name="choice"]:checked')?.value
}

function ensure_direct_gamepad(){
    if(direct_gamepad === null){
        const parent = document.getElementById("gamepad-container")
        direct_gamepad = new Gamepad(parent, DIRECT_GAMEPAD_INTERFACE, { storageKey: "gamepad_config", enabled: false })
        direct_gamepad.addListener((output) => {
            if (output["reset"] === true) {
                const button = document.getElementById("initial-states")
                button.dispatchEvent(new Event('click'));
            }
        })
    }
    return direct_gamepad
}

function ensure_direct_gamepad_controller(){
    const gamepad = ensure_direct_gamepad()
    if(direct_gamepad_controller === null){
        direct_gamepad_controller = new GamepadController(gamepad)
    }
    return direct_gamepad_controller
}

function ensure_attitude_setpoint_gamepad(){
    if(attitude_setpoint_gamepad === null){
        const parent = document.getElementById("attitude-setpoint-gamepad-container")
        attitude_setpoint_gamepad = new Gamepad(parent, ATTITUDE_SETPOINT_GAMEPAD_INTERFACE, {
            storageKey: ATTITUDE_SETPOINT_GAMEPAD_STORAGE_KEY,
            enabled: false,
        })
        attitude_setpoint_input = new AttitudeSetpointGamepadInput(attitude_setpoint_gamepad)
    }
    return attitude_setpoint_gamepad
}

function update_attitude_setpoint_gamepad(){
    const container = document.getElementById("attitude-setpoint-gamepad-container")
    const obs = document.getElementById("observations")?.observation
    const visible = observation_includes_attitude_setpoint(obs)
    if(container){
        container.classList.toggle("active", visible)
    }
    if(visible){
        ensure_attitude_setpoint_gamepad()
    }
    if(attitude_setpoint_gamepad){
        attitude_setpoint_gamepad.setEnabled(visible && selected_controller_mode() === "policy")
    }
}

// Platform mesh hashes and parameter history management
const PLATFORM_MESH_MAP = {
    "x500": "9602ffc2ffb77f62c4cf6fdc78fe67d32088870d",
    "crazyflie": "b75f5120e17783744a8fac5e1ab69c2dce10f0e3",
    "arpl": "775ba8559aeed800dbcdab93806601e39d84fede"
}

const SCENE_REGISTRY = {
    "ProcTHOR-Train-1": {
        hash: "7f1c9129532798e0b63bc41edb6b4c09251cf8a0",
        offset: [-3.92, -5.67, 1.0],
        rotation: [0, 0, 0],
    },
    "ProcTHOR-Train-26": {
        hash: "44b5c137aaf4d6b4ed6a0d8ce57fa315c2331e67",
        offset: [-5.2, -3.97, 1.64],
        rotation: [0, 0, 0],
    },
}

const addMeshToParameters = (params, platform) => 
    (!params.ui && PLATFORM_MESH_MAP[platform]) ? Object.assign(params, { ui: { enable: true, model: PLATFORM_MESH_MAP[platform] }}) : params

const paramStore = {
    historyKey: "l2f_parameter_history",
    lastSelectedKey: "l2f_last_selected_parameters",
    maxEntries: 10,
    getHistory: () => JSON.parse(localStorage.getItem(paramStore.historyKey) || "[]"),
    getLastSelected: () => localStorage.getItem(paramStore.lastSelectedKey),
    setLastSelected: (v) => localStorage.setItem(paramStore.lastSelectedKey, v),
    addToHistory(name, parameters) {
        const history = [{ name, parameters, timestamp: Date.now() }, 
            ...this.getHistory().filter(e => e.name !== name)].slice(0, this.maxEntries)
        localStorage.setItem(this.historyKey, JSON.stringify(history))
        return history
    }
}

function populateParameterDropdown(select, platforms, history) {
    const makeOpt = (value, text, title) => Object.assign(document.createElement("option"), { value, textContent: text, title: title || "" })
    const makeGroup = (label, items) => {
        const g = Object.assign(document.createElement("optgroup"), { label })
        items.forEach(i => g.appendChild(i))
        return g
    }
    select.innerHTML = ""
    select.appendChild(makeGroup("Presets", platforms.map(p => makeOpt(p, p))))
    select.appendChild(makeOpt("file", "📂 From file..."))
    if (history.length) select.appendChild(makeGroup("History", 
        history.map(e => makeOpt(`custom:${e.name}`, `📁 ${e.name}`, new Date(e.timestamp).toLocaleString()))))
}

class ProxyController {
    constructor(current_policy) {
        this.policy = current_policy
    }
    evaluate_step(...args) {
        return this.policy.evaluate_step(...args)
    }
    reset() {
        this.policy.reset()
    }
    get_reference(states){
        return this.policy.get_reference(states)
    }
    get_reference_index(trajectory, offset){
        return this.policy.get_reference_index(trajectory, offset)
    }
}
class MultiController {
    constructor(Controller) {
        this.Controller = Controller
        this.controllers = null
    }
    evaluate_step(state) {
        if (this.controllers === null || this.controllers.length !== state.length) {
            this.controllers = state.map(() => new this.Controller())
        }
        return state.map((state, i) => this.controllers[i].evaluate_step(state))
    }
    reset() {
        if (this.controllers === null) {
            return
        }
        this.controllers.forEach(controller => controller.reset())
    }
    get_reference(states){
        // console.assert(this.controllers.length == states.length)
        // return this.controllers.map((c) => c.get_reference())
        return null
    }
}

let model = null; Object.defineProperty(window, '_model', { get: () => model })
let trajectory = null
let trajectory_offset = 0
let trajectory_offset_axis = 0

function parse_camera_spec(obs_desc){
    if(!obs_desc) return null
    const branches = obs_desc.split(";").flatMap(b => {
        const m = b.match(/^(CameraRGB\w*\([^)]+\)),\s*(.+)$/)
        return m ? [m[1], m[2]] : [b]
    })
    const visual = branches.find(b => b.startsWith("CameraRGB") || b.startsWith("Visual("))
    if(!visual) return null
    const inner_match = visual.match(/\((.+)\)/)
    if(!inner_match) return null
    const inner = inner_match[1].split(",").map(s => Number(s.trim()))
    if(visual.startsWith("CameraRGB")){
        return { fov: inner[0], cam_h: inner[1], cam_w: inner[2] }
    }
    // legacy Visual(w, h, c, fov) carried the horizontal FOV in radians; visual.fov is degrees
    const fov = inner[3] !== undefined ? inner[3] * 180 / Math.PI : 63.78166175396324
    return { cam_w: inner[0], cam_h: inner[1], fov }
}

// Single entry point for "the observation string changed". Mirrors the string
// into the DOM, parses the camera spec, writes the dims into l2f.parameters[*].visual,
// and (on the first transition into visual mode) auto-reduces drones to 1 and loads
// the first registered scene. Returns the parsed spec (or null if no visual branch).
let last_visual_active = false
async function commit_observation(obs_string){
    const obs_input = document.getElementById("observations")
    obs_input.observation = obs_string
    obs_input.value = obs_string
    update_attitude_setpoint_gamepad()
    const spec = parse_camera_spec(obs_string)
    if(spec && l2f && l2f.parameters){
        for(const p of l2f.parameters){
            p.visual = p.visual || {}
            p.visual.cam_width = spec.cam_w
            p.visual.cam_height = spec.cam_h
            p.visual.fov = spec.fov
        }
    }
    if(spec){
        const preview_cb = document.getElementById("scene-preview-checkbox")
        if(preview_cb) preview_cb.checked = true
        if(l2f && l2f.ui_state) l2f.ui_state.show_onboard_preview = true
    }

    if(spec && !last_visual_active){
        // Only latch if auto-setup actually ran. If l2f isn't ready yet
        // (commit_observation fired from load_model before main() constructed L2F),
        // leave the flag false so a subsequent commit retries.
        const ok = await auto_setup_visual_mode()
        if(ok) last_visual_active = true
    } else if(!spec){
        last_visual_active = false
    }

    return spec
}

// On first transition into visual mode: single-drone (onboard preview only renders drone 0)
// and kick off the first registered scene load so the camera has something to look at.
// Respects user overrides — skips drone resize if already at 1, skips scene load if a scene is loaded.
// Returns true if it actually ran, false if l2f wasn't ready.
async function auto_setup_visual_mode(){
    if(!l2f) return false
    try { await l2f.initialized } catch(e) { return false }

    if(l2f.parameters && l2f.parameters.length > 1){
        await l2f.change_num_quadrotors(1, l2f.parameters[0])
    }

    if(!l2f.ui_state || !l2f.ui_state.onboard_scene){
        const first_entry = Object.values(SCENE_REGISTRY)[0]
        if(first_entry){
            const scene_select = document.getElementById("scene-selector")
            if(scene_select) scene_select.value = first_entry.hash
            // Mirror set_scene_defaults: populate offset/rotation inputs so the scene-load
            // handler's apply_scene_transform picks them up.
            const off = first_entry.offset || [0, 0, 0]
            const rot = first_entry.rotation || [0, 0, 0]
            const off_ids = ["scene-offset-x", "scene-offset-y", "scene-offset-z"]
            const rot_ids = ["scene-rot-r", "scene-rot-p", "scene-rot-y"]
            for(let i = 0; i < 3; i++){
                const e_off = document.getElementById(off_ids[i])
                const e_rot = document.getElementById(rot_ids[i])
                if(e_off) e_off.value = off[i]
                if(e_rot) e_rot.value = rot[i]
            }
            const btn = document.getElementById("scene-load-btn")
            if(btn) btn.click()  // triggers reload_onboard_from_obs; async, fire-and-forget
        }
    }
    return true
}

class Policy{
    constructor() {
        this.step = 0
        this.policy_states = null
        this.frame_buffers = null
        this.frame_buffer_head = null
        this.frame_buffer_episode_start = null
        this.frame_stack_config = null
        this.frame_buffer_capacity = 0
        this._last_obs_desc = null
        this.last_stacked_frames = null
        this.target_buffers = null
    }
    _parse_frame_stack_config(obs_desc) {
        if(this._last_obs_desc === obs_desc) return
        this._last_obs_desc = obs_desc
        const branches = obs_desc.split(";").flatMap(b => {
            const m = b.match(/^(CameraRGB\w*\([^)]+\)),\s*(.+)$/)
            return m ? [m[1], m[2]] : [b]
        })
        const visual = branches.find(b => b.startsWith("CameraRGBStacked"))
        if(visual){
            const with_target = visual.startsWith("CameraRGBStackedWithTarget")
            const inner = visual.match(/\((.+)\)/)[1].split(",").map(s => Number(s.trim()))
            this.frame_stack_config = { h: inner[1], w: inner[2], stride: inner[3], n_frames: inner[4], with_target }
            this.frame_buffer_capacity = (this.frame_stack_config.n_frames - 1) * this.frame_stack_config.stride + 1
        } else {
            this.frame_stack_config = null
        }
        this.frame_buffers = null
        this.frame_buffer_head = null
        this.frame_buffer_episode_start = null
        this.target_buffers = null
    }
    _ensure_frame_buffers(n_drones) {
        if(this.frame_buffers && this.frame_buffers.length === n_drones) return
        const cfg = this.frame_stack_config
        const buf_size = this.frame_buffer_capacity * cfg.w * cfg.h * 3
        this.frame_buffers = Array.from({length: n_drones}, () => new Float32Array(buf_size))
        this.frame_buffer_head = new Array(n_drones).fill(0)
        this.frame_buffer_episode_start = new Array(n_drones).fill(0)
        this.target_buffers = new Array(n_drones).fill(null)
    }
    get_observation(state, obs, trajectory) {
        let vehicle_state = null
        const full_observation = Array.from(state.get_observation())
        console.assert(full_observation.length > 21, "Observation is smaller than base observation")
        const get_state = () => {
            if (vehicle_state === null) {
                vehicle_state = JSON.parse(state.get_state())
            }
            return vehicle_state
        }
        const clip = (x, min, max) => x  < min ? min : (x > max ? max : x);
        const position_clip = x => clip(x, -1, 1)
        const velocity_clip = x => clip(x, -2, 2)
        const current_position = full_observation.slice(0, 3)
        const current_velocity = full_observation.slice(12, 15)
        const force_trajectory_tracking = document.getElementById("force-trajectory-tracking-observations").checked
        switch (true) {
            case obs === "Position" && !force_trajectory_tracking :
                return current_position
            case obs === "TrajectoryTrackingPosition" || (obs === "Position" && force_trajectory_tracking) :{
                const ref = this.get_reference_point(trajectory, 0)
                return current_position.map((x, axis_i) => position_clip(x - ref[axis_i]))
            }
            case obs === "OrientationRotationMatrix":
                return full_observation.slice(3, 12)
            case obs === "OrientationBodyZ": {
                const q = get_state().orientation
                return [
                    2 * q[1] * q[3] + 2 * q[0] * q[2],
                    2 * q[2] * q[3] - 2 * q[0] * q[1],
                    1 - 2 * q[1] * q[1] - 2 * q[2] * q[2],
                ]
            }
            case obs === "OrientationWorldZ" || obs === "OrientationMahonyWorldZ": {
                const q = get_state().orientation
                return [
                    2 * q[1] * q[3] - 2 * q[0] * q[2],
                    2 * q[2] * q[3] + 2 * q[0] * q[1],
                    1 - 2 * q[1] * q[1] - 2 * q[2] * q[2],
                ]
            }
            case obs === "AttitudeSetpoint":
                return attitude_setpoint_input?.getObservation(JSON.parse(state.get_parameters())) ?? [0, 0, 0, 1.0]
            case obs === "LinearVelocity" && !force_trajectory_tracking :
                return current_velocity
            case obs === "TrajectoryTrackingLinearVelocity" || (obs === "LinearVelocity" && force_trajectory_tracking) :{
                const ref = this.get_reference_point(trajectory, 0)
                return current_velocity.map((x, axis_i) => velocity_clip(x - ref[3 + axis_i]))
            }
            case obs.startsWith("LinearVelocityDelayed"):{
                const delay_string = obs.split("(")[1].split(")")[0]
                const delay = parseInt(delay_string)
                if (delay === 0) {
                    return full_observation.slice(12, 15)
                } else {
                    const s = get_state()
                    return s["linear_velocity_history"][s["linear_velocity_history"].length - delay]
                }
            }
            case obs.startsWith("TrajectoryTrackingLookahead"):
                const parameters_string = obs.split("(")[1].split(")")[0]
                const parameters_split = parameters_string.split(",")
                const num_steps = parseInt(parameters_split[0])
                const step_interval = parseInt(parameters_split[1])
                const flat_observation = new Array(num_steps).fill(0).map((_, step_i) => {
                    const ref = this.get_reference_point(trajectory, step_i * step_interval)
                    return [...current_position.map((x, axis_i) => position_clip(x - ref[axis_i])),
                            ...current_velocity.map((x, axis_i) => velocity_clip(x - ref[3 + axis_i]))]
                }).flat()
                return flat_observation
            case obs === "AngularVelocity":
                return full_observation.slice(15, 18)
            case obs.startsWith("AngularVelocityDelayed"):{
                const delay_string = obs.split("(")[1].split(")")[0]
                const delay = parseInt(delay_string)
                if (delay === 0) {
                    return full_observation.slice(15, 18)
                } else {
                    const s = get_state()
                    return s["angular_velocity_history"][s["angular_velocity_history"].length - delay]
                }
            }
            case obs === "LinearAccelerationBodyFrame" || obs === "IMUAccelerometer":
                return full_observation.slice(18, 21)
            case obs.startsWith("LinearAccelerationBodyFrameHistory"):{
                const N = parseInt(obs.split("(")[1].split(")")[0])
                if (N === 0) return []
                const s = get_state()
                const buf = s["linear_acceleration_body_history"]
                if (!buf) { console.error("WASM state missing linear_acceleration_body_history — rebuild WASM with StateLinearAccelerationHistory"); return new Array(3*N).fill(0) }
                const H = buf.length
                let step = (s["acceleration_history_step"] - 1 + H) % H
                const out = []
                for (let i = 0; i < N; i++) {
                    out.push(buf[step][0], buf[step][1], buf[step][2])
                    step = (step - 1 + H) % H
                }
                return out
            }
            case obs.startsWith("ActionHistory"):
                const history_length_string = obs.split("(")[1].split(")")[0]
                const history_length = parseInt(history_length_string)
                return full_observation.slice(21, 21 + history_length * 4)
            case obs === "RotorSpeeds":
                const parameters = JSON.parse(state.get_parameters())
                const min_action = parameters.dynamics.action_limit.min
                const max_action = parameters.dynamics.action_limit.max
                return get_state()["rpm"].map(x => (x - min_action) / (max_action - min_action) * 2 - 1)
            default:
                console.error("Unknown observation: ", obs)
                return null
        }
    }
    get_visual_observation(state, branch_string, ui_state, ui, parameters, drone_index, branch_index) {
        if(!ui || !ui.render_onboard_pixels || !ui_state) return []
        const srgb = x => x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1.0 / 2.4) - 0.055
        const cfg = this.frame_stack_config
        // Render target first if needed (so the camera ends up at the drone pose after the regular render below)
        if(cfg && cfg.with_target && this.target_buffers && this.target_buffers[drone_index] == null){
            const target_render_state = { position: [0, 0, 0], orientation: [1, 0, 0, 0] }
            const raw_target = ui.render_onboard_pixels(ui_state, target_render_state, parameters)
            if(raw_target){
                const target_buf = new Float32Array(raw_target.length)
                for(let i = 0; i < raw_target.length; i++) target_buf[i] = srgb(raw_target[i])
                this.target_buffers[drone_index] = target_buf
            }
        }
        const render_state = {
            position: Array.from(state.get_observation()).slice(0, 3),
            orientation: JSON.parse(state.get_state()).orientation,
        }
        const raw = ui.render_onboard_pixels(ui_state, render_state, parameters)
        if(!raw) return []
        if(!cfg || !this.frame_buffers || drone_index === undefined){
            for(let i = 0; i < raw.length; i++) raw[i] = srgb(raw[i])
            return Array.from(raw)
        }
        const img_c = 3
        const pixel_count = cfg.w * cfg.h * img_c
        const capacity = this.frame_buffer_capacity
        const buf = this.frame_buffers[drone_index]
        const head = this.frame_buffer_head[drone_index]
        buf.set(raw, (head % capacity) * pixel_count)
        this.frame_buffer_head[drone_index] = head + 1
        const n_pixels = cfg.w * cfg.h
        const data_channels = img_c * cfg.n_frames + (cfg.with_target ? img_c : 0)
        let stacked_c = data_channels
        if(model && model.input_dims && branch_index !== undefined && model.input_dims[branch_index]){
            const expected = Math.floor(model.input_dims[branch_index] / n_pixels)
            if(expected >= data_channels) stacked_c = expected
        }
        const output = new Float32Array(n_pixels * stacked_c)
        const ep_start = this.frame_buffer_episode_start[drone_index]
        for(let f = 0; f < cfg.n_frames; f++){
            let desired = head - f * cfg.stride
            if(desired < ep_start) desired = ep_start
            const read_offset = (desired % capacity) * pixel_count
            for(let p = 0; p < n_pixels; p++){
                for(let c = 0; c < img_c; c++){
                    output[p * stacked_c + f * img_c + c] = srgb(buf[read_offset + p * img_c + c])
                }
            }
        }
        if(cfg.with_target && this.target_buffers && this.target_buffers[drone_index]){
            const target = this.target_buffers[drone_index]
            const target_offset = cfg.n_frames * img_c
            for(let p = 0; p < n_pixels; p++){
                for(let c = 0; c < img_c; c++){
                    output[p * stacked_c + target_offset + c] = target[p * img_c + c]
                }
            }
        }
        if(drone_index === 0) this.last_stacked_frames = { data: output, cfg: { ...cfg, channels: stacked_c } }
        return Array.from(output)
    }
    evaluate_step(states, ui_state, ui, parameters) {
        if (!this.policy_states || this.policy_states.length !== states.length) {
            this.policy_states = states.map(() => model ? model.create_state() : null)
        }
        const observation_description = document.getElementById("observations").observation
        this._parse_frame_stack_config(observation_description)
        if(this.frame_stack_config) this._ensure_frame_buffers(states.length)
        const references = this.get_reference(states)
        const actions = states.map((state, i) => {
            state.observe()
            const reference = references[i]
            const branches = observation_description.split(";").flatMap(b => {
                const m = b.match(/^(CameraRGB\w*\([^)]+\)),\s*(.+)$/)
                return m ? [m[1], m[2]] : [b]
            })
            const branch_observations = branches.map((branch, branch_index) => {
                if(branch.startsWith("Visual(") || branch.startsWith("CameraRGB")) return this.get_visual_observation(state, branch, ui_state, ui, parameters?.[i], i, branch_index)
                return branch.split(".").map(x => this.get_observation(state, x, reference)).flat()
            })
            if(branch_observations.some(b => b === null || b.length === 0)) return new Float32Array(state.action_dim)
            let output
            if(model.num_branches > 1 && model.evaluate_tuple){
                output = model.evaluate_tuple(branch_observations.map(b => new Float32Array(b)))
            } else {
                const input = new Float32Array(branch_observations.flat())
                output = model.evaluate_step(input, this.policy_states[i])
            }
            return output
        })
        this.step += 1
        return actions
    }
    reset() {
        this.step = 0
        if(this.policy_states && model){
            this.policy_states.forEach(id => { if(id !== null) model.reset_state(id) })
        }
        this.policy_states = null
        this._last_obs_desc = null
        if(this.frame_buffer_head && this.frame_buffer_episode_start){
            for(let i = 0; i < this.frame_buffer_head.length; i++){
                this.frame_buffer_episode_start[i] = this.frame_buffer_head[i]
            }
        }
        if(this.target_buffers){
            for(let i = 0; i < this.target_buffers.length; i++) this.target_buffers[i] = null
        }
    }
    _get_reference(){
        return trajectory.trajectory
    }
    get_reference(states){
        const ref = this._get_reference()
        return states.map((_, i) => {
            return ref.map((step, j) => {
                const step_copy = step.slice()
                step_copy[trajectory_offset_axis] += trajectory_offset * i
                return step_copy
            })
        })
    }
    get_reference_index(reference, offset){
        // Use step - 1 because this is called after evaluate_step has already incremented
        const real_step = this.step - 1 + offset
        const cycle = Math.floor(real_step / reference.length)
        const phase = real_step % reference.length
        return cycle % 2 === 0 ? phase : reference.length - 1 - phase
    }
    get_reference_point(reference, offset){
        const real_step = this.step + offset
        const cycle = Math.floor(real_step / reference.length)
        const phase = real_step % reference.length
        const forward = cycle % 2 === 0
        const idx = forward ? phase : reference.length - 1 - phase
        const p = reference[idx]
        const dir = forward ? 1 : -1
        return [p[0], p[1], p[2], p[3] * dir, p[4] * dir, p[5] * dir]
    }
}


const CHECKPOINT_DB_NAME = "l2f-studio"
const CHECKPOINT_STORE_NAME = "checkpoints"
const CHECKPOINT_KEY = "checkpoint"

function openCheckpointDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(CHECKPOINT_DB_NAME, 1)
        request.onupgradeneeded = () => {
            const db = request.result
            if (!db.objectStoreNames.contains(CHECKPOINT_STORE_NAME)) {
                db.createObjectStore(CHECKPOINT_STORE_NAME)
            }
        }
        request.onsuccess = () => resolve(request.result)
        request.onerror = () => reject(request.error)
    })
}

async function getCheckpoint() {
    const db = await openCheckpointDB()
    try {
        return await new Promise((resolve, reject) => {
            const tx = db.transaction(CHECKPOINT_STORE_NAME, "readonly")
            const request = tx.objectStore(CHECKPOINT_STORE_NAME).get(CHECKPOINT_KEY)
            request.onsuccess = () => resolve(request.result || null)
            request.onerror = () => reject(request.error)
        })
    } finally {
        db.close()
    }
}

async function setCheckpoint(arrayBuffer) {
    const db = await openCheckpointDB()
    try {
        await new Promise((resolve, reject) => {
            const tx = db.transaction(CHECKPOINT_STORE_NAME, "readwrite")
            tx.objectStore(CHECKPOINT_STORE_NAME).put(arrayBuffer, CHECKPOINT_KEY)
            tx.oncomplete = () => resolve()
            tx.onerror = () => reject(tx.error)
            tx.onabort = () => reject(tx.error)
        })
    } finally {
        db.close()
    }
}

function showStatus(message, isError = false) {
    const status = document.getElementById('status');
    status.textContent = message;
    status.style.display = 'block';
    status.className = `status ${isError ? 'error' : 'success'}`;
    setTimeout(() => status.style.display = 'none', 3000);
}

async function load_model(checkpoint) {
    if (typeof checkpoint === "string") {
        checkpoint = await (await fetch(checkpoint)).arrayBuffer()
    }
    try {
        await setCheckpoint(checkpoint)
    } catch (e) {
        console.warn("Failed to persist checkpoint to IndexedDB: ", e)
    }
    // Before destroying the old model: pause the control loop immediately (so it
    // can't call evaluate_step on the destroyed inference during subsequent awaits)
    // and drop policy state IDs (which reference the about-to-be-freed WASM state vector).
    const was_paused = l2f ? l2f.pause : null
    if(l2f) l2f.pause = true
    if(proxy_controller && proxy_controller.policy){
        proxy_controller.policy.policy_states = null
    }
    const old_model = model
    model = null
    if(old_model){
        try { old_model.destroy() }
        catch(e){ console.error("Failed to destroy previous model (continuing): ", e) }
    }
    model = await rlt.load(checkpoint)
    if(model.verify){
        const check = model.verify()
        if(!check.pass){
            const expected = check.expected ? Array.from(check.expected) : []
            const actual = check.actual ? Array.from(check.actual) : []
            console.error("Model verification FAILED for " + model.checkpoint_name + ": max_diff=" + check.max_diff + " expected=" + expected + " actual=" + actual)
        }
        else console.log("Model verification passed for " + model.checkpoint_name + ": max_diff=" + check.max_diff)
    }
    const checkpoint_span = document.getElementById("checkpoint-name")
    checkpoint_span.textContent = model.checkpoint_name
    checkpoint_span.title = model.description()
    await commit_observation(model.meta.environment.observation)
    if(proxy_controller) proxy_controller.reset()
    // Restore prior pause state (if the loop was running before, resume it).
    if(l2f && was_paused === false) l2f.pause = false
}

async function main() {
    const trajectory_offset_container = document.getElementById("reference-trajectory-offset-container")
    const trajectory_offset_slider = trajectory_offset_container.querySelector("input[type=range]")
    const trajectory_offset_label = trajectory_offset_container.querySelectorAll(".control-container-label")[0]
    trajectory_offset_label.addEventListener("click", (event) => {
        switch(trajectory_offset_label.textContent){
            case "":
                trajectory_offset_label.textContent = "X Offset"
                trajectory_offset_axis = 0
                break;
            case "X Offset":
                trajectory_offset_label.textContent = "Y Offset"
                trajectory_offset_axis = 1
                break;
            case "Y Offset":
                trajectory_offset_label.textContent = "Z Offset"
                trajectory_offset_axis = 2
                break;
            case "Z Offset":
                trajectory_offset_label.textContent = "X Offset"
                trajectory_offset_axis = 0
                break;
        }
    })
    trajectory_offset_label.dispatchEvent(new Event("click"))
    const trajectory_offset_value = trajectory_offset_container.querySelectorAll(".control-container-label")[1]
    trajectory_offset_slider.addEventListener("input", (event) => {
        trajectory_offset = parseFloat(event.target.value)
        trajectory_offset_value.textContent = trajectory_offset.toFixed(2)
    })
    const trajectory_select = document.getElementById("reference-trajectory")
    const trajectories = { "Position": Position, "Lissajous": Lissajous, "Langevin": SecondOrderLangevin, "Ping Pong": PingPong }
    trajectory_select.innerHTML = ""
    for (const name in trajectories) {
        trajectory_select.innerHTML += `<option value="${name}">${name}</option>`
    }
    trajectory_select.addEventListener("change", (event) => {
        const trajectory_class = trajectories[event.target.value]
        trajectory = new trajectory_class()


        const trajectory_options_container = document.getElementById("reference-trajectory-options")
        trajectory_options_container.innerHTML = ""
        const trajectory_option_template = document.getElementById("reference-trajectory-option-template")

        for (const [key, config] of Object.entries(trajectory.parameters)) {
            const template = trajectory_option_template.content.cloneNode(true)

            const labels = template.querySelectorAll(".control-container-label")
            labels[0].textContent = key
            labels[1].textContent = config.default

            const slider = template.querySelector("input[type=range]")
            slider.min = config.range[0]
            slider.max = config.range[1]
            slider.step = config.step ?? 0.01
            slider.value = config.default

            slider.addEventListener("input", () => {
                labels[1].textContent = slider.value
                trajectory.set_parameter(key, parseFloat(slider.value))
                trajectory.parameters_updated()
            })

            trajectory_options_container.appendChild(template)
        }
    })
    document.getElementById("reference-trajectory-reset").addEventListener("click", () => {
        trajectory_select.dispatchEvent(new Event("change"))
    })
    trajectory_select.value = "Lissajous"
    document.getElementById("reference-trajectory-reset").dispatchEvent(new Event("click"))


    document.getElementById("default-checkpoint-btn").addEventListener("click", async () => {
        load_model(file_url)
    })
    document.getElementById("load-checkpoint-btn").addEventListener("click", async () => {
        document.getElementById("load-checkpoint-btn-backend").click();
    })
    document.getElementById("load-checkpoint-btn-backend").addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = async function (e) {
                const array_buffer = e.target.result;
                await load_model(array_buffer)
                console.log("loaded model: ", model.checkpoint_name)
                showStatus(`Loaded model: ${model.checkpoint_name}`);
            };
            reader.readAsArrayBuffer(file);
        }
        event.target.value = "";
    })
    document.getElementById("observations").addEventListener("keydown", async (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            await commit_observation(e.target.value)
            await reload_onboard_from_obs(scene_select.value)
        }
    })
    const controller_code_loaded = fetch("./controller.js").then(async (response) => {
        if (response.status !== 200) {
            console.error("Error loading controller.js: ", response.status)
        }
        document.getElementById("controller-code").value = await response.text()
        document.getElementById("controller-selector-container").querySelectorAll('input[name="choice"]').forEach(radio => {
            radio.addEventListener("change", (event) => {
                if (event.target.value === "policy") {
                    document.getElementById("policy-container").style.display = "block"
                    document.getElementById("controller-container").style.display = "none"
                    document.getElementById("gamepad-container").style.display = "none"
                    if(direct_gamepad) direct_gamepad.setEnabled(false)
                    update_attitude_setpoint_gamepad()
                    proxy_controller.policy = new Policy(model)
                }
                else if (event.target.value === "controller") {
                    document.getElementById("policy-container").style.display = "none"
                    document.getElementById("controller-container").style.display = "block"
                    document.getElementById("gamepad-container").style.display = "none"
                    if(direct_gamepad) direct_gamepad.setEnabled(false)
                    update_attitude_setpoint_gamepad()
                    const event = new KeyboardEvent("keydown", { key: "Enter" });
                    document.getElementById("controller-code").dispatchEvent(event);
                }
                else if (event.target.value === "gamepad") {
                    document.getElementById("policy-container").style.display = "none"
                    document.getElementById("controller-container").style.display = "none"
                    document.getElementById("gamepad-container").style.display = "block"
                    update_attitude_setpoint_gamepad()
                    const gamepad = ensure_direct_gamepad()
                    gamepad.setEnabled(true)
                    proxy_controller.policy = ensure_direct_gamepad_controller()
                }
            });
        });
        // const gamepadRadio = document.querySelector('input[name="choice"][value="gamepad"]');
        // gamepadRadio.checked = true;
        // gamepadRadio.dispatchEvent(new Event('change'));
    })
    document.getElementById("controller-code").addEventListener("keydown", async (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            const code = document.getElementById("controller-code").value
            const blob = new Blob([code], { type: 'application/javascript' });
            const url = URL.createObjectURL(blob);
            const Controller = (await import(url)).default
            URL.revokeObjectURL(url);
            proxy_controller.policy = new MultiController(Controller)
            window.controller = proxy_controller.policy
        }
    })

    document.getElementById("vehicle-select-all-btn").addEventListener("click", () => {
        const vehicle_container = document.getElementById("vehicle-list")
        const elements = Array.from(vehicle_container.querySelectorAll(":scope .vehicle"))
        const checkboxes = elements.map(vehicle => vehicle.querySelector(".vehicle-checkbox"))
        let all_checked = checkboxes.every(checkbox => checkbox.checked)
        checkboxes.forEach(checkbox => {
            checkbox.checked = !all_checked
        })
    })


    const seed = 12

    let checkpoint = null
    try {
        checkpoint = await getCheckpoint()
    } catch (e) {
        console.warn("Failed to read checkpoint from IndexedDB: ", e)
    }
    if (checkpoint !== null) {
        console.log("loading checkpoint from IndexedDB")
    }
    else {
        console.log(`Loading checkpoint from ${file_url}`)
        checkpoint = await (await fetch(file_url)).arrayBuffer()
    }
    await load_model(checkpoint)

    const sim_container = document.getElementById("sim-container")
    proxy_controller = new ProxyController(new Policy(model))


    const platforms_text = await (await fetch("./external/blob/registry/index.json")).text()
    const platforms = platforms_text.split("\n").filter(line => line.trim() !== "").sort()
    const platform_select = document.getElementById("vehicle-load-dynamics-selector")
    
    // Populate dropdown with presets and custom history
    const paramHistory = paramStore.getHistory()
    populateParameterDropdown(platform_select, platforms, paramHistory)

    // Determine which parameters to load (URL override > last selected > default)
    const param_override = urlParams.get('parameters');
    const lastSelected = paramStore.getLastSelected()
    let selectedValue = param_override ? param_override : (lastSelected || "x500")
    
    // Validate that the selected value exists in the dropdown
    const validOptions = Array.from(platform_select.options).map(o => o.value)
    if (!validOptions.includes(selectedValue)) {
        selectedValue = "x500"
    }
    platform_select.value = selectedValue

    // Load the selected parameters
    let default_parameters
    if (selectedValue.startsWith("custom:")) {
        const customName = selectedValue.replace("custom:", "")
        const historyEntry = paramHistory.find(e => e.name === customName)
        if (historyEntry) {
            default_parameters = structuredClone(historyEntry.parameters)
        } else {
            // Fallback to x500 if custom entry not found
            default_parameters = await (await fetch(`./external/blob/registry/x500.json`)).json()
            addMeshToParameters(default_parameters, "x500")
        }
    } else if (selectedValue !== "file") {
        default_parameters = await (await fetch(`./external/blob/registry/${selectedValue}.json`)).json()
        addMeshToParameters(default_parameters, selectedValue)
    } else {
        // Fallback for "file" option
        default_parameters = await (await fetch(`./external/blob/registry/x500.json`)).json()
        addMeshToParameters(default_parameters, "x500")
    }
    
    // Reset trajectory steps to all zeros
    if (default_parameters.trajectory && default_parameters.trajectory.steps) {
        default_parameters.trajectory.steps = default_parameters.trajectory.steps.map(() => ({
            position: [0, 0, 0],
            yaw: 0,
            linear_velocity: [0, 0, 0],
            yaw_velocity: 0
        }))
    }

    console.log("Waiting for trajectory to be initialized")

    l2f = new L2F(sim_container, Array(10).fill(default_parameters), proxy_controller, seed)
    window._l2f = l2f
    
    // Wire trajectory updates to invalidate rendered trajectory lines
    const wireTrajectoryCallback = () => { trajectory.onUpdate = () => l2f.remove_trajectory_lines() }
    wireTrajectoryCallback()
    trajectory_select.addEventListener("change", wireTrajectoryCallback)

    l2f.state_update_callbacks.push((states) => {
        const vehicle_container = document.getElementById("vehicle-list")
        if (vehicle_container.children.length != states.length) {
            const vehicle_template = document.getElementById("vehicle-template")
            vehicle_container.innerHTML = ""
            states.forEach((state, i) => {
                const vehicle_pre = vehicle_template.content.cloneNode(true)
                const vehicle = vehicle_pre.querySelector(".vehicle")
                vehicle.dataset.vehicleId = i
                vehicle.querySelector(".vehicle-title").textContent = `Vehicle ${i}`
                // vehicle.querySelector(".vehicle-id").textContent = JSON.stringify(state.parameters.dynamics, null, 2)
                vehicle_container.appendChild(vehicle)
                // on hover
                vehicle.addEventListener("mouseenter", (event) => {
                    console.log(`hovering over vehicle ${i}`)
                    vehicle.classList.add("vehicle-hover")
                    event.stopPropagation()
                })
                vehicle.addEventListener("mouseleave", (event) => {
                    vehicle.classList.remove("vehicle-hover")
                    event.stopPropagation()
                })
            })
        }
        states.forEach((state, i) => {
            const vehicle = vehicle_container.children[i]
            const fixed = (x, n) => {
                const y = x.toFixed(n);
                return y >= 0 ? `+${y}` : y;
            }
            vehicle.querySelector(".vehicle-position").textContent = state.state.position.map(x => fixed(x, 3)).join(",")
            vehicle.querySelector(".vehicle-action").textContent = state.action.map(x => fixed(x, 2)).join(",")
            const thrusts = state.parameters.dynamics.rotor_thrust_coefficients.map(curve => curve.reduce((a, c, i) => a + c * Math.pow(state.parameters.dynamics.action_limit.max, i), 0))
            const t2w = thrusts.reduce((a, c) => a+c, 0)/(9.81 * state.parameters.dynamics.mass)
            const torque = Math.abs(state.parameters.dynamics.rotor_positions[0][0]) * Math.sqrt(2) * thrusts[0]
            const t2i = torque / state.parameters.dynamics.J[0][0]
            vehicle.querySelector(".vehicle-t2w-t2i").textContent = `${t2w.toFixed(2)} / ${t2i.toFixed(2)}`
            vehicle.querySelector(".vehicle-avg-error").textContent = `${(state.avgTrackingError * 100).toFixed(1)} cm`
            vehicle.querySelector(".vehicle-steps-since-reset").textContent = `${state.stepsSinceReset}`
            vehicle.title = JSON.stringify(state.parameters.dynamics, null, 2)
        })

    })

    l2f.initialized.then(async () => {
        // Startup-load races: load_model may have run commit_observation before l2f.parameters existed.
        // Re-commit now that the L2F backend is initialized so visual dims land in parameters.visual
        // and the one-shot auto-setup (drones=1, first scene) fires.
        const current_obs = document.getElementById("observations").observation
        if(current_obs) await commit_observation(current_obs)
        await parameter_manager.initialized
        const sim_container_cover = document.getElementById("sim-container-cover")
        sim_container_cover.style.display = "none"
        const pause_button = document.getElementById("pause")
        if(pause_button.innerText === "Resume"){
            pause_button.click()
        }

        const perturbation_id_input = document.getElementById("perturbation-id-input")
        perturbation_id_input.value = "parameters.dynamics.mass"
        perturbation_id_input.dispatchEvent(new Event("input"))
        const event = new Event('keydown')
        event.key = "Enter"
        perturbation_id_input.dispatchEvent(event)
    })

    

    const parameter_manager = new ParameterManager(l2f)
    const sim_controls = new SimControls(l2f, proxy_controller)


    const set_parameters = async (parameters) => {
        const vehicle_container = document.getElementById("vehicle-list")
        const elements = Array.from(vehicle_container.querySelectorAll(":scope .vehicle"))
        let ids = []
        elements.forEach(vehicle => {
            const checkbox = vehicle.querySelector(".vehicle-checkbox")
            if (checkbox && checkbox.checked) {
                const vehicleId = parseInt(vehicle.dataset.vehicleId, 10)
                ids.push(vehicleId)
            }
        })
        // If no checkboxes are checked, update all vehicles
        if (ids.length === 0) {
            ids = elements.map(vehicle => parseInt(vehicle.dataset.vehicleId, 10))
        }
        console.log("setting parameters for vehicles: ", ids)
        console.log("parameters: ", parameters)

        // Preset JSONs lack the visual camera spec stamped in by commit_observation; carry it forward.
        const previous_visual = l2f.parameters?.[ids[0]]?.visual
        if (previous_visual) {
            parameters.visual = { ...previous_visual, ...(parameters.visual || {}) }
        }

        // Reset trajectory steps to all zeros (we don't use the trajectory from the parameters, because we need to feed it dynamically and the user should be able to tweak it from the UI)
        if (parameters.trajectory && parameters.trajectory.steps) {
            parameters.trajectory.steps = parameters.trajectory.steps.map(() => ({
                position: [0, 0, 0],
                yaw: 0,
                linear_velocity: [0, 0, 0],
                yaw_velocity: 0
            }))
        }
        
        await l2f.set_parameters(ids, ids.map(() => parameters))
    }
    document.getElementById("vehicle-load-dynamics-btn-backend").addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = async function (e) {
                console.log(`Loaded dynamics from ${file.name}`)
                const parameters = JSON.parse(e.target.result)
                
                // Generate a name from filename (remove .json extension)
                const name = file.name.replace(/\.json$/i, "")
                
                // Add to history and update dropdown
                const newHistory = paramStore.addToHistory(name, parameters)
                populateParameterDropdown(platform_select, platforms, newHistory)
                
                // Select the newly added custom parameter
                const customValue = `custom:${name}`
                platform_select.value = customValue
                paramStore.setLastSelected(customValue)
                
                set_parameters(parameters)
            };
            reader.readAsText(file);
        }
        event.target.value = "";
    })
    document.getElementById("vehicle-load-dynamics-btn").addEventListener("click", async () => {
        const selectedValue = document.getElementById("vehicle-load-dynamics-selector").value
        
        if (selectedValue === "file") {
            document.getElementById("vehicle-load-dynamics-btn-backend").click();
            return;
        }
        else if (selectedValue.startsWith("custom:")) {
            // Load from history
            const customName = selectedValue.replace("custom:", "")
            const history = paramStore.getHistory()
            const historyEntry = history.find(e => e.name === customName)
            if (historyEntry) {
                const parameters = structuredClone(historyEntry.parameters)
                paramStore.setLastSelected(selectedValue)
                set_parameters(parameters)
            } else {
                console.error(`Custom parameter "${customName}" not found in history`)
            }
        }
        else {
            // Load preset
            const platform = selectedValue
            const parameters = await (await fetch(`./external/blob/registry/${platform}.json`)).json()
            addMeshToParameters(parameters, platform)
            paramStore.setLastSelected(selectedValue)
            set_parameters(parameters)
        }
    })

    // Scene selector
    const scene_select = document.getElementById("scene-selector")
    const scene_off = [document.getElementById("scene-offset-x"), document.getElementById("scene-offset-y"), document.getElementById("scene-offset-z")]
    const scene_rot = [document.getElementById("scene-rot-r"), document.getElementById("scene-rot-p"), document.getElementById("scene-rot-y")]
    const deg2rad = d => d * Math.PI / 180
    const set_scene_defaults = (entry) => {
        const off = entry ? entry.offset : [0, 0, 0]
        const rot = entry ? entry.rotation : [0, 0, 0]
        for(let i = 0; i < 3; i++){ scene_off[i].value = off[i]; scene_rot[i].value = rot[i] }
    }
    Object.entries(SCENE_REGISTRY).forEach(([name, entry]) => {
        const option = document.createElement("option")
        option.value = entry.hash
        option.textContent = name
        scene_select.appendChild(option)
    })
    scene_select.addEventListener("change", () => {
        const entry = Object.values(SCENE_REGISTRY).find(e => e.hash === scene_select.value)
        set_scene_defaults(entry)
    })
    const reload_onboard_from_obs = async (hash) => {
        const visual = l2f?.parameters?.[0]?.visual
        const cam_w = visual?.cam_width ?? 64
        const cam_h = visual?.cam_height ?? 64
        const fov = visual?.fov ?? 63.78166175396324
        if(l2f.ui && l2f.ui.setup_onboard_scene){
            await l2f.ui.setup_onboard_scene(l2f.ui_state, hash, cam_w, cam_h, fov)
            apply_scene_transform()
            l2f.ui_state.show_onboard_preview = document.getElementById("scene-preview-checkbox").checked
        }
    }
    document.getElementById("scene-load-btn").addEventListener("click", async () => {
        await reload_onboard_from_obs(scene_select.value)
    })
    const apply_scene_transform = () => {
        if(!l2f.ui_state) return
        const off = scene_off.map(el => parseFloat(el.value) || 0)
        l2f.ui_state.onboard_scene_translation = off
        if(!l2f.ui_state.onboard_scene) return
        const rpy = scene_rot.map(el => deg2rad(parseFloat(el.value) || 0))
        // Build drone-frame rotation from RPY in FLU, using Three.js axes
        // FLU X=(1,0,0)→Three(1,0,0), FLU Y=(0,1,0)→Three(0,0,-1), FLU Z=(0,0,1)→Three(0,1,0)
        const q_drone = new THREE.Quaternion()
        q_drone.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), rpy[2]))
        q_drone.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, -1), rpy[1]))
        q_drone.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), rpy[0]))
        const q_inv = q_drone.invert()
        // Pivot point in Three.js coords: FLU (x,y,z) → Three (x, z, -y)
        const T = new THREE.Vector3(off[0], off[2], -off[1])
        l2f.ui_state.onboard_scene.children.forEach(child => {
            if(child.isLight) return
            child.quaternion.copy(q_inv)
            child.position.copy(T).sub(new THREE.Vector3().copy(T).applyQuaternion(q_inv))
        })
        if(proxy_controller && proxy_controller.policy && proxy_controller.policy.target_buffers){
            const tb = proxy_controller.policy.target_buffers
            for(let i = 0; i < tb.length; i++) tb[i] = null
        }
    }
    ;[...scene_off, ...scene_rot].forEach(el => el.addEventListener("change", apply_scene_transform))
    document.getElementById("scene-preview-checkbox").addEventListener("change", (e) => {
        if(l2f.ui_state) l2f.ui_state.show_onboard_preview = e.target.checked
    })
    document.getElementById("scene-obs-res-checkbox").addEventListener("change", (e) => {
        if(l2f.ui_state) l2f.ui_state.onboard_obs_resolution = e.target.checked
    })

    const obs_input = document.getElementById("observations")
    obs_input.addEventListener("change", async () => { await commit_observation(obs_input.value) })

    // Frame stack preview
    const frame_stack_container = document.getElementById("frame-stack-preview")
    const frame_stack_canvases = []
    let frame_stack_label = null
    function render_frame_stack_preview() {
        const policy = proxy_controller.policy
        if(!policy.last_stacked_frames || !policy.frame_stack_config){
            frame_stack_container.classList.remove("active")
            return
        }
        const { data, cfg } = policy.last_stacked_frames
        const { w, h, n_frames, stride, with_target } = cfg
        const stacked_c = cfg.channels || (3 * n_frames + (with_target ? 3 : 0))
        const slot_count = n_frames + (with_target ? 1 : 0)
        if(frame_stack_canvases.length !== slot_count){
            frame_stack_container.innerHTML = ""
            frame_stack_label = document.createElement("div")
            frame_stack_label.className = "frame-stack-label"
            frame_stack_container.appendChild(frame_stack_label)
            frame_stack_canvases.length = 0
            for(let f = 0; f < slot_count; f++){
                const canvas = document.createElement("canvas")
                canvas.width = w
                canvas.height = h
                canvas.style.width = Math.max(w, 48) + "px"
                canvas.style.height = Math.max(h, 48) + "px"
                if(with_target && f === slot_count - 1) canvas.title = "target"
                frame_stack_container.appendChild(canvas)
                frame_stack_canvases.push(canvas)
            }
        }
        frame_stack_label.textContent = `Policy input: ${n_frames} frames${with_target ? " + target" : ""} (${w}x${h}, stride ${stride})`
        frame_stack_container.classList.add("active")
        for(let f = 0; f < slot_count; f++){
            const ctx = frame_stack_canvases[f].getContext("2d")
            const img_data = ctx.createImageData(w, h)
            const pixels = img_data.data
            const c_offset = f * 3
            for(let p = 0; p < w * h; p++){
                const r = data[p * stacked_c + c_offset + 0]
                const g = data[p * stacked_c + c_offset + 1]
                const b = data[p * stacked_c + c_offset + 2]
                pixels[p * 4 + 0] = Math.round(r * 255)
                pixels[p * 4 + 1] = Math.round(g * 255)
                pixels[p * 4 + 2] = Math.round(b * 255)
                pixels[p * 4 + 3] = 255
            }
            ctx.putImageData(img_data, 0, 0)
        }
    }
    l2f.state_update_callbacks.push(() => render_frame_stack_preview())

    if(new URLSearchParams(window.location.search).get("DEBUG") === "true"){
        document.getElementById("vehicle-load-dynamics-selector").value = "crazyflie"
        document.getElementById("vehicle-load-dynamics-btn").click()
        const entry = SCENE_REGISTRY["ProcTHOR-Train-1"]
        scene_select.value = entry.hash
        set_scene_defaults(entry)
        l2f.initialized.then(() => {
            document.getElementById("scene-load-btn").click()
        })
    }
}

if (document.readyState === 'loading') {
    document.addEventListener("DOMContentLoaded", main)
} else {
    main()
}

function setupScrollIndicatorHiding() {
    const scrollIndicator = document.getElementById('scroll-indicator');
    const controlsBox = document.getElementById('controls-box');
    if (scrollIndicator && controlsBox) {
        function checkOverflowAndUpdateIndicator() {
            const hasOverflow = controlsBox.scrollHeight > controlsBox.clientHeight;
            if (hasOverflow) {
                const isAtBottom = controlsBox.scrollTop + controlsBox.clientHeight >= controlsBox.scrollHeight - 1;
                if (isAtBottom) {
                    scrollIndicator.style.display = 'none';
                } else {
                    scrollIndicator.style.display = 'flex';
                    scrollIndicator.classList.remove('hidden');
                }
            } else {
                scrollIndicator.style.display = 'none';
            }
        }
        
        checkOverflowAndUpdateIndicator();
        
        controlsBox.addEventListener('scroll', checkOverflowAndUpdateIndicator);
        
        const resizeObserver = new ResizeObserver(checkOverflowAndUpdateIndicator);
        resizeObserver.observe(controlsBox);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener("DOMContentLoaded", setupScrollIndicatorHiding);
} else {
    setupScrollIndicatorHiding();
}

const drag_and_drop_overlay = document.getElementById('drag-and-drop-overlay');
let drag_and_drop_counter = 0;

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
    document.body.addEventListener(event, e => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

document.body.addEventListener('dragenter', () => {
    drag_and_drop_counter++;
    drag_and_drop_overlay.style.display = 'flex';
}, false);

document.body.addEventListener('dragleave', () => {
    drag_and_drop_counter--;
    if (drag_and_drop_counter === 0) {
        drag_and_drop_overlay.style.display = 'none';
    }
}, false);

document.body.addEventListener('drop', e => {
    drag_and_drop_counter = 0;
    drag_and_drop_overlay.style.display = 'none';

    const file = e.dataTransfer.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = async function (e) {
            const array_buffer = e.target.result;
            await load_model(array_buffer)
            console.log("loaded model: ", model.checkpoint_name)
            showStatus(`Loaded model: ${model.checkpoint_name}`);
        };
        reader.readAsArrayBuffer(file);
    }
}, false);
