import {Conta} from "./conta.js";
import {create_touch_stick} from "./touch-stick.js";

const params = new URLSearchParams(window.location.search);
const requested_frames = Number.parseInt(params.get("frames") || "0", 10);
const max_frames = Number.isFinite(requested_frames) && requested_frames > 0 ? requested_frames : 0;
const conta = new Conta();

const status_element = document.getElementById("status");
const error_element = document.getElementById("error");
const fps_element = document.getElementById("fps");
const frame_rate_panel = document.getElementById("frame-rate-tuning");
const target_fps = document.getElementById("target-fps");
const progress_element = document.getElementById("progress");
const progress_percent = document.getElementById("progress-percent");
const drone_select = document.getElementById("drone");
const scene_select = document.getElementById("scene");
const reset_button = document.getElementById("reset");
const flight_state = document.getElementById("flight-state");
const position_readout = document.getElementById("position-readout");
const orientation_readout = document.getElementById("orientation-readout");
const tuning_panel = document.getElementById("flight-tuning");
const position_offset = document.getElementById("position-offset");
const velocity_target = document.getElementById("velocity-target");
const yaw_rate_limit = document.getElementById("yaw-rate-limit");
const yaw_error_limit = document.getElementById("yaw-error-limit");
const mouse_sensitivity = document.getElementById("mouse-sensitivity");
const release_button = document.getElementById("release-control");
const tuning_button = document.getElementById("touch-tuning");
const canvases = ["camera0", "camera1", "depth", "segmentation", "normals", "flow"].map(id => document.getElementById(id));
const contexts = canvases.map(canvas => canvas.getContext("2d"));
const keys = new Set();
let yaw_delta = 0;
let reset_requested = false;
let ready = false;
let touch_control = false;
let mouse_control_requested = false;
let images = null;
let frame_count = 0;
let fps_window_start = performance.now();
let fps_window_frames = 0;
const left_stick = create_touch_stick(document.getElementById("left-stick"), () => touch_control && has_control());
const right_stick = create_touch_stick(document.getElementById("right-stick"), () => touch_control && has_control());
document.body.classList.toggle("touch-ui", navigator.maxTouchPoints > 0);

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

function set_status(text){
    status_element.textContent = text;
}

function format_pose(values, digits){
    return `(${values.map(value => Number(value.toFixed(digits)).toFixed(digits)).join(", ")})`;
}

function has_control(){
    return ready && !document.hidden && !tuning_panel.open && !frame_rate_panel.open && (touch_control || mouse_has_control());
}

function mouse_has_control(){
    return canvases.slice(0, 2).includes(document.pointerLockElement);
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
    for(const id of ["left-stick", "right-stick"]){
        document.getElementById(id).setAttribute("aria-disabled", String(!controlling || !touch_control));
    }
    flight_state.textContent = !ready ? "Loading" : document.hidden ? "Paused" : controlling ? "User control" : "Position hold";
    if(ready && max_frames === 0){
        set_status(document.hidden ? "Simulation paused while this tab is hidden" : controlling ? (touch_control ? "Control active · Lift thumbs to hold position" : "Control active · Click or Esc to release") : "Simulation running · RAPTOR position hold");
    }
}

function poll_input(){
    const controlling = has_control();
    const input = [
        controlling ? (touch_control ? -right_stick.y : Number(keys.has("KeyW")) - Number(keys.has("KeyS"))) : 0,
        controlling ? (touch_control ? -right_stick.x : Number(keys.has("KeyA")) - Number(keys.has("KeyD"))) : 0,
        controlling ? (touch_control ? -left_stick.y : Number(keys.has("KeyE")) - Number(keys.has("KeyQ"))) : 0,
        controlling && !touch_control ? yaw_delta : 0,
        Number(reset_requested),
        Number(!document.hidden),
        Number(yaw_rate_limit.value) * Math.PI / 180,
        Number(yaw_error_limit.value) * Math.PI / 180,
        Number(position_offset.value),
        Number(velocity_target.value),
        Number(target_fps.value),
        controlling && touch_control ? -left_stick.x : 0,
    ];
    yaw_delta = 0;
    reset_requested = false;
    return input;
}

function take_touch_control(){
    if(ready && !tuning_panel.open && !frame_rate_panel.open){
        release_controls();
        document.body.classList.add("touch-ui");
        touch_control = true;
        update_flight_state();
    }
}

async function toggle_pointer_lock(canvas){
    if(!ready){
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

for(const canvas of canvases.slice(0, 2)){
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
});
release_button.addEventListener("click", release_controls);
function open_tuning(){
    frame_rate_panel.close();
    tuning_panel.showModal();
    release_controls();
}
tuning_button.addEventListener("click", open_tuning);
document.addEventListener("visibilitychange", () => {
    if(document.hidden){
        release_controls();
    }
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

function show_error(error){
    ready = false;
    release_controls();
    reset_button.disabled = true;
    fps_element.hidden = true;
    flight_state.textContent = "Unavailable";
    progress_element.hidden = true;
    progress_percent.hidden = true;
    error_element.textContent = error instanceof Error ? error.message : String(error);
    error_element.hidden = false;
    set_status("The simulation could not start.");
    console.error(error);
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
    if(contexts.some(context => context === null)){
        throw new Error("Canvas drawing is unavailable in this browser.");
    }
    set_status("Loading the simulation…");
    const {default: createModule} = await import("./blob/build/hyperdrone_web.js");
    const module = await createModule({
        print: text => console.log(text),
        printErr: text => console.error(text),
        onAbort: reason => show_error(`Renderer stopped: ${reason}`),
        rlt_status: set_status,
        rlt_input: poll_input,
        rlt_present: (pointer, width, height, num_views, x, y, z, qw, qx, qy, qz) => {
            if(num_views !== canvases.length){
                throw new Error(`Expected ${canvases.length} camera outputs; received ${num_views}. Rebuild the HyperDrone browser module.`);
            }
            if(images === null || images[0].width !== width || images[0].height !== height){
                images = contexts.map((context, index) => {
                    canvases[index].width = width;
                    canvases[index].height = height;
                    return context.createImageData(width, height);
                });
            }
            const heap = module.HEAPU8;
            const image_bytes = width * height * 4;
            for(let view = 0; view < num_views; view++){
                const begin = pointer + view * image_bytes;
                images[view].data.set(heap.subarray(begin, begin + image_bytes));
                contexts[view].putImageData(images[view], 0, 0);
            }
            position_readout.textContent = format_pose([x, y, z], 2);
            orientation_readout.textContent = format_pose([qw, qx, qy, qz], 4);
            if(!ready){
                ready = true;
                reset_button.disabled = false;
                fps_element.hidden = false;
                update_flight_state();
                fps_window_start = performance.now();
                fps_window_frames = 0;
            }
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
    if(module._demo_api_version?.() !== 1){
        throw new Error("The page and simulation bundle are incompatible. Deploy the matching blob/build files and update the pinned revision in download_blob.sh.");
    }
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
    populate(scene_select, scenes, scene);
    populate(drone_select, registry.drones, drone);
    module.FS.writeFile("/scene.glb", await fetch_asset(scene));
    module.FS.writeFile("/drone.glb", await fetch_asset(drone));
    module.FS.writeFile("/policy.tar", await fetch_asset(registry.policy));
    const spawn = scene.reference === pose.reference ? JSON.stringify({position: pose.position, yaw: pose.yaw_degrees * Math.PI / 180}) : "";
    set_status("Preparing the scene and cameras…");
    const result = await module.ccall("demo_run", "number", ["string", "string", "string", "number", "number", "string"], ["/scene.glb", "/drone.glb", "/policy.tar", drone.dynamics_index, max_frames, spawn], {async: true});
    if(result !== 0){
        throw new Error(`The renderer exited with status ${result}.`);
    }
}

main().catch(show_error);
