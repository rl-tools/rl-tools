import {Conta} from "./conta.js";
import {create_touch_stick} from "./touch-stick.js";

const params = new URLSearchParams(window.location.search);
const requested_frames = Number.parseInt(params.get("frames") || "0", 10);
const max_frames = Number.isFinite(requested_frames) && requested_frames > 0 ? requested_frames : 0;
const conta = new Conta();
const rendering_defaults = window.matchMedia("(pointer: coarse)").matches
    ? {id: "medium-mb1-aa1-480x270", fps: 30}
    : {id: "high-mb1-aa1-960x540", fps: 60};

const status_element = document.getElementById("status");
const error_element = document.getElementById("error");
const motor_layout_warning = document.getElementById("motor-layout-warning");
const gpu_element = document.getElementById("gpu");
const fps_element = document.getElementById("fps");
const frame_rate_panel = document.getElementById("frame-rate-tuning");
const target_fps = document.getElementById("target-fps");
target_fps.defaultValue = String(rendering_defaults.fps);
const progress_element = document.getElementById("progress");
const progress_percent = document.getElementById("progress-percent");
const drone_select = document.getElementById("drone");
const scene_select = document.getElementById("scene");
const reset_button = document.getElementById("reset");
const flight_state = document.getElementById("flight-state");
const position_readout = document.getElementById("position-readout");
const orientation_readout = document.getElementById("orientation-readout");
const pose_readout = document.getElementById("pose-readout");
const tuning_panel = document.getElementById("flight-tuning");
const position_offset = document.getElementById("position-offset");
const velocity_target = document.getElementById("velocity-target");
const yaw_rate_limit = document.getElementById("yaw-rate-limit");
const yaw_error_limit = document.getElementById("yaw-error-limit");
const mouse_sensitivity = document.getElementById("mouse-sensitivity");
const release_button = document.getElementById("release-control");
const tuning_button = document.getElementById("touch-tuning");
const rendering_button = document.getElementById("rendering");
const rendering_panel = document.getElementById("rendering-settings");
const onboard_fov = document.getElementById("onboard-fov");
const external_fov = document.getElementById("external-fov");
const render_fidelity = document.getElementById("render-fidelity");
const render_motion_blur = document.getElementById("render-motion-blur");
const render_antialiasing = document.getElementById("render-antialiasing");
const render_resolution = document.getElementById("render-resolution");
const render_agents = document.getElementById("render-agents");
const layout_seed = document.getElementById("layout-seed");
const render_apply = document.getElementById("render-apply");
const render_availability = document.getElementById("render-availability");
const camera_canvases = ["camera0", "camera1"].map(id => document.getElementById(id));
const segmentation_canvas = document.getElementById("segmentation");
const segmentation_tooltip = document.getElementById("segmentation-tooltip");
const segmentation_center = document.getElementById("segmentation-center");
const segmentation_center_label = document.getElementById("segmentation-center-label");
const keys = new Set();
let yaw_delta = 0;
let reset_requested = false;
let ready = false;
let failed = false;
let touch_control = false;
let mouse_control_requested = false;
let segmentation_classes = {};
let segmentation_ids = null;
let segmentation_cursor = null;
let drone_pose = null;
let frame_count = 0;
let fps_window_start = performance.now();
let fps_window_frames = 0;
let render_configurations = [];
let default_render_configuration = null;
const left_stick = create_touch_stick(document.getElementById("left-stick"), () => touch_control && has_control());
const right_stick = create_touch_stick(document.getElementById("right-stick"), () => touch_control && has_control());
document.body.classList.toggle("touch-ui", navigator.maxTouchPoints > 0);

function hide_segmentation_hover(){
    segmentation_cursor = null;
    update_segmentation_readout();
}

function update_segmentation_readout(){
    if(!ready || failed || document.hidden || settings_open() || segmentation_ids === null){
        segmentation_cursor = null;
        segmentation_tooltip.hidden = true;
        segmentation_center.hidden = true;
        return;
    }
    if(segmentation_cursor !== null && !document.pointerLockElement){
        const rect = segmentation_canvas.getBoundingClientRect();
        const x = segmentation_cursor.x - rect.left;
        const y = segmentation_cursor.y - rect.top;
        if(x >= 0 && y >= 0 && x < rect.width && y < rect.height){
            const column = Math.floor(x * segmentation_canvas.width / rect.width);
            const row = Math.floor(y * segmentation_canvas.height / rect.height);
            const id = segmentation_ids[row * segmentation_canvas.width + column];
            segmentation_center.hidden = true;
            segmentation_tooltip.textContent = id === 0xffffffff ? "No surface" : segmentation_classes[id];
            segmentation_tooltip.hidden = false;
            segmentation_canvas.setAttribute("aria-describedby", "segmentation-tooltip");
            segmentation_tooltip.style.left = `${Math.max(0, Math.min(x + 10, rect.width - segmentation_tooltip.offsetWidth))}px`;
            segmentation_tooltip.style.top = `${Math.max(0, Math.min(y + 10, rect.height - segmentation_tooltip.offsetHeight))}px`;
            return;
        }
    }
    segmentation_cursor = null;
    segmentation_tooltip.hidden = true;
    const id = segmentation_ids[Math.floor(segmentation_canvas.height / 2) * segmentation_canvas.width + Math.floor(segmentation_canvas.width / 2)];
    segmentation_center_label.textContent = id === 0xffffffff ? "No surface" : segmentation_classes[id];
    segmentation_center.hidden = false;
    segmentation_canvas.setAttribute("aria-describedby", "segmentation-center-label");
}

segmentation_canvas.addEventListener("pointermove", event => {
    if(event.pointerType === "mouse"){
        segmentation_cursor = {x: event.clientX, y: event.clientY};
        update_segmentation_readout();
    }
    else{
        hide_segmentation_hover();
    }
});
segmentation_canvas.addEventListener("pointerleave", hide_segmentation_hover);
segmentation_canvas.addEventListener("pointercancel", hide_segmentation_hover);
window.addEventListener("scroll", update_segmentation_readout, true);
for(const panel of [tuning_panel, frame_rate_panel, rendering_panel]){
    panel.addEventListener("close", update_segmentation_readout);
}

for(const select of [scene_select, drone_select]){
    select.addEventListener("change", () => {
        release_controls();
        params.set(select.id, select.value);
        window.location.search = params.toString();
    });
}

function populate(select, entries, selected){
    for(const entry of entries){
        const option = document.createElement("option");
        option.value = entry.name;
        option.textContent = entry.label || entry.name;
        option.selected = entry.name === selected.name;
        select.appendChild(option);
    }
    select.disabled = false;
}

function find_configuration(){
    return render_configurations.find(configuration =>
        configuration.fidelity === render_fidelity.value &&
        String(configuration.motion_blur) === render_motion_blur.value &&
        String(configuration.antialiasing) === render_antialiasing.value &&
        `${configuration.width}x${configuration.height}` === render_resolution.value &&
        String(configuration.num_agents) === render_agents.value);
}

function update_rendering(){
    const available = Boolean(find_configuration());
    const valid_seed = layout_seed.checkValidity();
    render_apply.disabled = !available || !valid_seed;
    render_availability.textContent = !valid_seed
        ? "Use a whole-number seed from 0 to 4294967295."
        : available
        ? "Applying scene or rendering settings restarts the flight. Reset flight repeats the same drone layout."
        : "This combination is unavailable. Change a rendering option.";
}

function select_configuration(configuration){
    render_fidelity.value = configuration.fidelity;
    render_motion_blur.value = String(configuration.motion_blur);
    render_antialiasing.value = String(configuration.antialiasing);
    render_resolution.value = `${configuration.width}x${configuration.height}`;
    render_agents.value = String(configuration.num_agents);
    update_rendering();
}

function setup_rendering(configurations){
    if(configurations.length === 0){
        throw new Error("No rendering configurations are available in this simulation bundle.");
    }
    render_configurations = configurations;
    default_render_configuration = configurations.find(configuration => configuration.id === rendering_defaults.id && configuration.num_agents === 8) || configurations[0];
    const requested = params.get("render");
    const agents = params.has("agents") ? Number(params.get("agents")) : default_render_configuration.num_agents;
    const selected = configurations.find(configuration => configuration.id === (requested || default_render_configuration.id) && configuration.num_agents === agents);
    layout_seed.value = params.get("seed") ?? layout_seed.defaultValue;
    const fidelity_labels = {low: "Low", medium: "Medium", high: "High", "very-high": "Very high"};
    const fields = [
        [render_fidelity, configuration => configuration.fidelity, configuration => fidelity_labels[configuration.fidelity]],
        [render_motion_blur, configuration => String(configuration.motion_blur), configuration => configuration.motion_blur ? "On" : "Off"],
        [render_antialiasing, configuration => String(configuration.antialiasing), configuration => configuration.antialiasing ? "On" : "Off"],
        [render_resolution, configuration => `${configuration.width}x${configuration.height}`, configuration => `${configuration.width} × ${configuration.height}`],
        [render_agents, configuration => String(configuration.num_agents), configuration => configuration.num_agents === 1 ? "None" : "7 hovering drones"],
    ];
    for(const [select, value, label] of fields){
        const entries = [...new Map(configurations.map(configuration => [value(configuration), {
            name: value(configuration), label: label(configuration),
        }])).values()];
        populate(select, entries, {name: value(selected || default_render_configuration)});
        select.disabled = entries.length === 1;
    }
    rendering_button.disabled = false;
    update_rendering();
    if(!selected){
        throw new Error(`Requested rendering and drone configuration is unavailable. Open Rendering to choose available settings.`);
    }
    if(!layout_seed.checkValidity()){
        throw new Error("The layout seed must be a whole number from 0 to 4294967295. Open Rendering to change it.");
    }
    return selected;
}

for(const select of [render_fidelity, render_motion_blur, render_antialiasing, render_resolution, render_agents]){
    select.addEventListener("change", update_rendering);
}
layout_seed.addEventListener("input", update_rendering);
document.getElementById("render-defaults").addEventListener("click", () => {
    layout_seed.value = layout_seed.defaultValue;
    for(const control of [onboard_fov, external_fov]){
        control.value = control.defaultValue;
        update_fov(control);
    }
    if(default_render_configuration){
        select_configuration(default_render_configuration);
    }
});
render_apply.addEventListener("click", () => {
    const configuration = find_configuration();
    if(configuration && layout_seed.reportValidity()){
        release_controls();
        params.set("render", configuration.id);
        params.set("agents", configuration.num_agents);
        params.set("seed", layout_seed.value);
        window.location.search = params.toString();
    }
});

function update_fov(control){
    document.getElementById(`${control.id}-value`).textContent = `${control.value}°`;
    if(control.value === control.defaultValue){
        params.delete(control.id);
    }
    else{
        params.set(control.id, control.value);
    }
}
for(const control of [onboard_fov, external_fov]){
    const requested = Number(params.get(control.id));
    if(params.has(control.id) && Number.isFinite(requested) && requested >= Number(control.min) && requested <= Number(control.max)){
        control.value = String(requested);
    }
    control.addEventListener("input", () => update_fov(control));
    update_fov(control);
}

function set_status(text){
    if(!failed){
        status_element.textContent = text;
    }
}

function round_values(values, digits){
    return values.map(value => Number(value.toFixed(digits)));
}

function format_pose(values, digits){
    return `(${values.map(value => value.toFixed(digits)).join(", ")})`;
}

function settings_open(){
    return tuning_panel.open || frame_rate_panel.open || rendering_panel.open;
}

function has_control(){
    return ready && !document.hidden && !settings_open() && (touch_control || mouse_has_control());
}

function mouse_has_control(){
    return camera_canvases.includes(document.pointerLockElement);
}

function clear_input(){
    keys.clear();
    yaw_delta = 0;
    left_stick.clear();
    right_stick.clear();
}

function update_flight_state(){
    const controlling = has_control();
    document.body.classList.toggle("ready", ready);
    document.body.classList.toggle("controlling", controlling);
    document.body.classList.toggle("touch-controlling", controlling && touch_control);
    document.body.classList.toggle("mouse-controlling", controlling && mouse_has_control());
    release_button.hidden = !controlling || !touch_control;
    tuning_button.disabled = !ready;
    pose_readout.disabled = !ready;
    for(const id of ["left-stick", "right-stick"]){
        document.getElementById(id).setAttribute("aria-disabled", String(!controlling || !touch_control));
    }
    flight_state.textContent = failed ? "Unavailable" : !ready ? "Loading" : document.hidden ? "Paused" : controlling ? "User control" : "Position hold";
    if(ready && max_frames === 0){
        set_status(document.hidden ? "Simulation paused while this tab is hidden" : controlling ? (touch_control ? "Control active · Lift thumbs to hold position" : "Control active · Click or Esc to release") : "Simulation running · RAPTOR position hold");
    }
}

function poll_input(){
    const controlling = has_control();
    const input = [
        controlling ? (touch_control ? -right_stick.y : Number(keys.has("KeyW")) - Number(keys.has("KeyS"))) : 0,
        controlling ? (touch_control ? -right_stick.x : Number(keys.has("KeyA")) - Number(keys.has("KeyD"))) : 0,
        controlling ? (touch_control ? -left_stick.y * 0.7 : Number(keys.has("KeyE")) - Number(keys.has("KeyQ"))) : 0,
        controlling && !touch_control ? yaw_delta : 0,
        Number(reset_requested),
        Number(!document.hidden),
        Number(yaw_rate_limit.value) * Math.PI / 180,
        Number(yaw_error_limit.value) * Math.PI / 180,
        Number(position_offset.value),
        Number(velocity_target.value),
        Number(target_fps.value),
        controlling && touch_control ? -left_stick.x * 0.5 : 0,
        Number(onboard_fov.value),
        Number(external_fov.value),
    ];
    yaw_delta = 0;
    reset_requested = false;
    return input;
}

function take_touch_control(){
    if(ready && !settings_open()){
        release_controls();
        document.body.classList.add("touch-ui");
        touch_control = true;
        update_flight_state();
    }
}

async function toggle_pointer_lock(canvas){
    if(!ready || settings_open()){
        return;
    }
    if(!canvas.requestPointerLock){
        take_touch_control();
        return;
    }
    if(mouse_has_control()){
        release_controls();
    }
    else{
        release_controls();
        try{
            mouse_control_requested = true;
            await canvas.requestPointerLock();
        }
        catch(error){
            set_status("Mouse capture was unavailable. Click a camera to try again.");
            console.warn(error);
        }
    }
}

for(const canvas of camera_canvases){
    let pointer_type = "";
    canvas.addEventListener("pointerdown", event => {
        pointer_type = event.pointerType;
    });
    canvas.addEventListener("click", event => {
        const type = event.pointerType || pointer_type;
        pointer_type = "";
        if(type === "touch" || type === "pen"){
            take_touch_control();
        }
        else{
            toggle_pointer_lock(canvas);
        }
    });
    canvas.addEventListener("keydown", event => {
        if(event.code === "Enter" || event.code === "Space"){
            event.preventDefault();
            toggle_pointer_lock(canvas);
        }
    });
}
document.addEventListener("pointerlockchange", () => {
    hide_segmentation_hover();
    if(mouse_has_control() && !mouse_control_requested){
        document.exitPointerLock();
        return;
    }
    clear_input();
    if(mouse_has_control()){
        touch_control = false;
        document.body.classList.remove("touch-ui");
    }
    else{
        mouse_control_requested = false;
    }
    update_flight_state();
});
document.addEventListener("pointerlockerror", () => set_status("Mouse capture was unavailable. Click a camera to try again."));
document.addEventListener("keydown", event => {
    if(event.code === "Escape" && touch_control){
        release_controls();
    }
    if(event.code === "KeyT" && event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey){
        event.preventDefault();
        if(!event.repeat){
            if(tuning_panel.open){
                tuning_panel.close();
            }
            else{
                open_tuning();
            }
        }
        return;
    }
    if(has_control() && mouse_has_control() && ["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"].includes(event.code)){
        event.preventDefault();
        keys.add(event.code);
    }
});
document.addEventListener("keyup", event => keys.delete(event.code));
document.addEventListener("mousemove", event => {
    if(has_control() && mouse_has_control()){
        yaw_delta -= event.movementX * 0.002 * Number(mouse_sensitivity.value);
    }
});
function release_controls(){
    hide_segmentation_hover();
    touch_control = false;
    mouse_control_requested = false;
    clear_input();
    if(mouse_has_control()){
        document.exitPointerLock();
    }
    update_flight_state();
}
window.addEventListener("blur", release_controls);
window.addEventListener("pagehide", release_controls);
window.addEventListener("resize", () => {
    if(touch_control){
        release_controls();
    }
    update_segmentation_readout();
});
release_button.addEventListener("click", release_controls);
function open_tuning(){
    frame_rate_panel.close();
    rendering_panel.close();
    tuning_panel.showModal();
    release_controls();
}
tuning_button.addEventListener("click", open_tuning);
pose_readout.addEventListener("click", async () => {
    try{
        await navigator.clipboard.writeText(JSON.stringify(drone_pose));
        pose_readout.classList.add("copied");
        setTimeout(() => pose_readout.classList.remove("copied"), 1200);
    }
    catch(error){
        set_status("The pose could not be copied to the clipboard.");
        console.warn(error);
        setTimeout(update_flight_state, 2000);
    }
});
rendering_button.addEventListener("click", () => {
    tuning_panel.close();
    frame_rate_panel.close();
    rendering_panel.showModal();
    release_controls();
});
document.addEventListener("visibilitychange", () => {
    if(document.hidden){
        release_controls();
    }
    update_segmentation_readout();
    fps_window_start = performance.now();
    fps_window_frames = 0;
    fps_element.textContent = "— fps";
    update_flight_state();
});
reset_button.addEventListener("click", () => {
    reset_requested = true;
    release_controls();
});

fps_element.addEventListener("click", () => {
    tuning_panel.close();
    rendering_panel.close();
    frame_rate_panel.showModal();
    release_controls();
});
function update_frame_rate(){
    document.getElementById("target-fps-value").textContent = `${target_fps.value} FPS`;
    fps_element.title = `Target: ${target_fps.value} FPS · Click to adjust`;
}
target_fps.addEventListener("input", update_frame_rate);
document.getElementById("frame-rate-defaults").addEventListener("click", () => {
    target_fps.value = target_fps.defaultValue;
    update_frame_rate();
});
update_frame_rate();

function update_tuning_values(){
    document.getElementById("position-offset-value").textContent = `${Number(position_offset.value).toFixed(2)} m`;
    document.getElementById("velocity-target-value").textContent = `${Number(velocity_target.value).toFixed(2)} m/s`;
    document.getElementById("yaw-rate-value").textContent = `${yaw_rate_limit.value}°/s`;
    document.getElementById("yaw-error-value").textContent = `${yaw_error_limit.value}°`;
    document.getElementById("mouse-sensitivity-value").textContent = `${Number(mouse_sensitivity.value).toFixed(2)}×`;
}
tuning_panel.addEventListener("input", update_tuning_values);
document.getElementById("tuning-defaults").addEventListener("click", () => {
    for(const control of [position_offset, velocity_target, yaw_rate_limit, yaw_error_limit, mouse_sensitivity]){
        control.value = control.defaultValue;
    }
    update_tuning_values();
});
update_tuning_values();

function show_error(error, message){
    console.error(error);
    if(failed){
        return;
    }
    set_status("The simulation could not start.");
    failed = true;
    ready = false;
    release_controls();
    reset_button.disabled = true;
    fps_element.hidden = true;
    progress_element.hidden = true;
    progress_percent.hidden = true;
    error_element.textContent = message ?? (error instanceof Error ? error.message : String(error));
    const platform = navigator.userAgentData?.platform || navigator.platform || navigator.userAgent;
    const linux = /Linux/i.test(platform) && !/Android|CrOS/i.test(navigator.userAgent);
    if(linux && (error_element.textContent.includes("WebGPU: no suitable adapter available") || error_element.textContent.startsWith("WebGPU is unavailable"))){
        error_element.textContent += " (WebGPU support on Linux depends on your browser and GPU. Try Chrome or Firefox on Windows/macOS, or Safari on macOS/iOS.)";
    }
    error_element.hidden = false;
}

async function fetch_asset(asset){
    const description = asset.label || asset.name || asset.description;
    progress_element.hidden = false;
    progress_percent.hidden = false;
    const bytes = await conta.load(asset.reference, {
        sha256: asset.sha256,
        on_progress: ({phase, received, total}) => {
            if(phase !== "download"){
                const action = phase === "cache" ? "Loading cached" : phase === "verify" ? "Verifying" : "Locating";
                set_status(`${action} ${description}…`);
                progress_element.removeAttribute("value");
                progress_percent.textContent = "";
                return;
            }
            const downloaded = (received / 1024 / 1024).toFixed(1);
            set_status(`Downloading ${description}: ${downloaded}${total ? ` / ${(total / 1024 / 1024).toFixed(1)}` : ""} MB`);
            if(total){
                const percent = Math.min(100, Math.floor(received / total * 100));
                progress_element.value = percent;
                progress_percent.textContent = `${percent}%`;
            }
            else{
                progress_element.removeAttribute("value");
                progress_percent.textContent = "";
            }
        },
    });
    progress_element.hidden = true;
    progress_percent.hidden = true;
    return bytes;
}

async function read_json(path){
    const response = await fetch(path, {cache: "no-cache"});
    if(!response.ok){
        throw new Error(`Could not load ${path}: HTTP ${response.status}`);
    }
    return response.json();
}

async function main(){
    if(!window.isSecureContext){
        throw new Error("Open this page over HTTPS or localhost to enable WebGPU.");
    }
    if(!navigator.gpu){
        throw new Error("WebGPU is unavailable in this browser. Open the page in a browser with WebGPU support enabled.");
    }
    set_status("Loading the simulation…");
    const {default: createModule} = await import("./blob/build/hyperdrone_web.js");
    const module = await createModule({
        print: text => console.log(text),
        printErr: text => console.error(text),
        onAbort: reason => {
            const scene = scene_select.value || params.get("scene") || "this scene";
            const message = reason === "annotations::free_space: no valid positions found"
                ? `Could not find a suitable starting position in ${scene}. Choose another scene.`
                : `Renderer stopped: ${reason || "an unexpected error occurred"}`;
            show_error(reason, message);
        },
        rlt_status: set_status,
        rlt_gpu_info: info => {
            const name = info.description || [info.vendor, info.architecture, info.device].filter(Boolean).join(" · ");
            gpu_element.textContent = name ? `GPU · ${name}` : "";
            gpu_element.title = Object.entries(info).filter(([, value]) => value).map(([key, value]) => `${key}: ${value}`).join("\n");
            gpu_element.hidden = !name;
        },
        rlt_motor_layout: (deviation, wheelbase, threshold) => {
            motor_layout_warning.hidden = deviation <= threshold;
            motor_layout_warning.textContent = deviation > threshold
                ? `Motor layout warning: visual motor positions differ from the dynamics by up to ${(100 * deviation).toFixed(1)}% of wheelbase (${(1000 * deviation * wheelbase).toFixed(1)} mm horizontally after scaling; threshold ${(100 * threshold).toFixed(0)}%).`
                : "";
        },
        rlt_input: poll_input,
        rlt_segmentation_classes: classes => { segmentation_classes = {...classes, 0xfffffffd: "Drone"}; },
        rlt_present: (segmentation_pointer, width, height, x, y, z, qw, qx, qy, qz) => {
            if(segmentation_ids === null || segmentation_ids.length !== width * height){
                segmentation_ids = new Uint32Array(width * height);
            }
            const heap = module.HEAPU8;
            segmentation_ids.set(new Uint32Array(heap.buffer, heap.byteOffset + segmentation_pointer, width * height));
            drone_pose = {position: round_values([x, y, z], 2), orientation: round_values([qw, qx, qy, qz], 3)};
            position_readout.textContent = format_pose(drone_pose.position, 2);
            orientation_readout.textContent = format_pose(drone_pose.orientation, 3);
            if(!ready){
                ready = true;
                reset_button.disabled = false;
                fps_element.hidden = false;
                update_flight_state();
                fps_window_start = performance.now();
                fps_window_frames = 0;
            }
            update_segmentation_readout();
            frame_count++;
            fps_window_frames++;
            const now = performance.now();
            if(now - fps_window_start >= 1000){
                fps_element.textContent = `${(fps_window_frames * 1000 / (now - fps_window_start)).toFixed(1)} fps`;
                fps_window_start = now;
                fps_window_frames = 0;
            }
            if(max_frames > 0 && frame_count >= max_frames){
                set_status(`Rendered ${frame_count} frames`);
            }
        },
    });
    if(module._demo_api_version?.() !== 10){
        throw new Error("The page and simulation bundle are incompatible. Deploy the matching blob/build files and update the pinned revision in download_blob.sh.");
    }
    const configuration = setup_rendering(JSON.parse(module.ccall("demo_configurations", "string", [], [])));
    const seed = Number(layout_seed.value);
    const [registry, pose] = await Promise.all([read_json("registry.json"), read_json("pose.json")]);
    const scenes = JSON.parse(module.ccall("demo_scenes", "string", [], []));
    const scene = scenes.find(scene => scene.name === (params.get("scene") || pose.scene));
    const drone = registry.drones.find(drone => drone.name === (params.get("drone") || registry.drones[0].name));
    if(!scene || !drone){
        throw new Error("Unknown scene or airframe in the page URL.");
    }
    const default_scene = scenes.find(scene => scene.name === pose.scene);
    if(default_scene?.reference !== pose.reference || pose.position?.length !== 3 || !pose.position.every(Number.isFinite) || !Number.isFinite(pose.yaw_degrees)){
        throw new Error("pose.json must pin a Train200 scene and a finite FLU position and yaw.");
    }
    if(!Array.isArray(pose.other_positions) || pose.other_positions.length !== 7
        || !pose.other_positions.every(position => Array.isArray(position) && position.length === 3 && position.every(Number.isFinite))){
        throw new Error("pose.json must supply seven finite FLU positions for the other drones.");
    }
    const drones = configuration.num_agents === 1 ? [drone] : [drone, ...registry.drones.filter(entry => entry !== drone)];
    for(const entry of drones){
        if(!Array.isArray(entry.fpv_position) || entry.fpv_position.length !== 3 || !entry.fpv_position.every(Number.isFinite)
            || !Number.isInteger(entry.dynamics_index) || entry.dynamics_index < 0 || entry.dynamics_index > 5){
            throw new Error("The airframe registry must supply a dynamics model and a finite FLU camera position in metres.");
        }
    }
    populate(scene_select, scenes, scene);
    populate(drone_select, registry.drones, drone);
    module.FS.writeFile("/scene.glb", await fetch_asset(scene));
    const airframes = [];
    for(const [index, entry] of drones.entries()){
        const path = `/drone-${index}.glb`;
        module.FS.writeFile(path, await fetch_asset(entry));
        airframes.push({path, dynamics_index: entry.dynamics_index, fpv_position: entry.fpv_position});
    }
    module.FS.writeFile("/policy.tar", await fetch_asset(registry.policy));
    const session = JSON.stringify({num_agents: configuration.num_agents, seed, drones: airframes,
        ...(scene.reference === pose.reference ? {
            position: pose.position, yaw: pose.yaw_degrees * Math.PI / 180,
            ...(configuration.num_agents > 1 ? {other_positions: pose.other_positions} : {}),
        } : {})});
    set_status("Preparing the scene and cameras…");
    const result = await module.ccall("demo_run", "number", ["string", "string", "number", "string", "string"], ["/scene.glb", "/policy.tar", max_frames, session, configuration.id], {async: true});
    if(result === 3){
        throw new Error("Could not place seven separated drones in this scene. Choose another scene or seed, or set Other drones to None in Rendering.");
    }
    if(result !== 0){
        throw new Error(`The renderer exited with status ${result}.`);
    }
}

main().catch(show_error);
