const SCENE = {description: "ProcTHOR-Train-1.glb", sha1: "7f1c9129532798e0b63bc41edb6b4c09251cf8a0", size_mb: 32};
const POLICY = {description: "Flight controller", sha1: "b6bd082c60fcb27f42a2b5f2f635587d2b60d170", size_mb: 1};
const DRONES = [
    {name: "crazyflie", label: "Crazyflie", sha1: "87d2b8b1d518445872d34906320a76656731902f", size_mb: 5},
    {name: "crazyflie_brushless", label: "Crazyflie Brushless", sha1: "eedb897237c1d77ed8ef0681554d92d06a7c27ca", size_mb: 19},
    {name: "savagebee_pusher", label: "SavageBee Pusher", sha1: "fd4eddc7c4a3f74e51823a2fa392bb96e095e53e", size_mb: 13},
    {name: "arpl", label: "ARPL", sha1: "8398a239fd1cd80df8adab67497baf7be3d8cda4", size_mb: 46},
    {name: "x500", label: "X500", sha1: "f6681e7a8b7fa7ef023bedcd795981becb731d0e", size_mb: 74},
    {name: "soft", label: "Soft", sha1: "fcdf53a791f313b9bb07f775b10b9e4c2bfa0ae9", size_mb: 113},
];
const DEFAULT_ASSET_BASE = "https://huggingface.co/datasets/rl-tools/conta/resolve/main/data/";
const params = new URLSearchParams(window.location.search);
const asset_base = params.get("assets") === "local" ? "./assets/" : (params.get("assets") || DEFAULT_ASSET_BASE);
const requested_frames = Number.parseInt(params.get("frames") || "0", 10);
const max_frames = Number.isFinite(requested_frames) && requested_frames > 0 ? requested_frames : 0;
const drone_index = Math.max(0, DRONES.findIndex(drone => drone.name === params.get("drone")));

const status_element = document.getElementById("status");
const error_element = document.getElementById("error");
const fps_element = document.getElementById("fps");
const progress_element = document.getElementById("progress");
const drone_select = document.getElementById("drone");
const reset_button = document.getElementById("reset");
const flight_state = document.getElementById("flight-state");
const canvases = ["camera0", "camera1", "depth", "segmentation", "normals", "flow"].map(id => document.getElementById(id));
const contexts = canvases.map(canvas => canvas.getContext("2d"));
const keys = new Set();
let yaw_delta = 0;
let reset_requested = false;
let ready = false;
let images = null;
let frame_count = 0;
let fps_window_start = performance.now();
let fps_window_frames = 0;

for(const [index, drone] of DRONES.entries()){
    const option = document.createElement("option");
    option.value = drone.name;
    option.textContent = `${drone.label} · ${drone.size_mb} MB`;
    option.selected = index === drone_index;
    drone_select.appendChild(option);
}
drone_select.addEventListener("change", () => {
    params.set("drone", drone_select.value);
    window.location.search = params.toString();
});

function set_status(text){
    status_element.textContent = text;
}

function active(){
    return !document.hidden && canvases.slice(0, 2).includes(document.pointerLockElement);
}

function clear_input(){
    keys.clear();
    yaw_delta = 0;
}

function update_flight_state(){
    const flying = active();
    document.body.classList.toggle("flying", flying);
    flight_state.replaceChildren();
    const dot = document.createElement("span");
    dot.className = "dot";
    flight_state.append(dot, !ready ? "Loading" : flying ? "Flying" : "Paused");
    if(ready && max_frames === 0){
        set_status(flying ? "WASD to move · Mouse to turn · Click or Esc to pause" : "Ready · Click either camera to fly");
    }
}

function poll_input(){
    const flying = active();
    const input = [
        flying ? Number(keys.has("KeyW")) - Number(keys.has("KeyS")) : 0,
        flying ? Number(keys.has("KeyA")) - Number(keys.has("KeyD")) : 0,
        flying ? yaw_delta : 0,
        Number(reset_requested),
        Number(flying),
    ];
    yaw_delta = 0;
    reset_requested = false;
    return input;
}

async function toggle_pointer_lock(canvas){
    if(!ready){
        return;
    }
    if(active()){
        document.exitPointerLock();
    }
    else{
        try{
            await canvas.requestPointerLock();
        }
        catch(error){
            set_status("Mouse capture was unavailable. Click a camera to try again.");
            console.warn(error);
        }
    }
}

for(const canvas of canvases.slice(0, 2)){
    canvas.addEventListener("click", () => toggle_pointer_lock(canvas));
    canvas.addEventListener("keydown", event => {
        if(event.code === "Enter" || event.code === "Space"){
            event.preventDefault();
            toggle_pointer_lock(canvas);
        }
    });
}
document.addEventListener("pointerlockchange", () => {
    clear_input();
    fps_window_start = performance.now();
    fps_window_frames = 0;
    fps_element.textContent = "";
    update_flight_state();
});
document.addEventListener("pointerlockerror", () => set_status("Mouse capture was unavailable. Click a camera to try again."));
document.addEventListener("keydown", event => {
    if(active() && ["KeyW", "KeyA", "KeyS", "KeyD"].includes(event.code)){
        event.preventDefault();
        keys.add(event.code);
    }
});
document.addEventListener("keyup", event => keys.delete(event.code));
document.addEventListener("mousemove", event => {
    if(active()){
        yaw_delta -= event.movementX * 0.002;
    }
});
function pause(){
    clear_input();
    if(canvases.slice(0, 2).includes(document.pointerLockElement)){
        document.exitPointerLock();
    }
    update_flight_state();
}
window.addEventListener("blur", pause);
document.addEventListener("visibilitychange", () => {
    if(document.hidden){
        pause();
    }
});
reset_button.addEventListener("click", () => {
    reset_requested = true;
    clear_input();
});

function show_error(error){
    pause();
    ready = false;
    document.body.classList.remove("ready");
    reset_button.disabled = true;
    flight_state.textContent = "Unavailable";
    progress_element.hidden = true;
    error_element.textContent = error instanceof Error ? error.message : String(error);
    error_element.hidden = false;
    set_status("The simulation could not start.");
    document.title = "failed";
    console.error(error);
}

async function fetch_asset(asset){
    const url = `${asset_base.replace(/\/$/, "")}/${asset.sha1}`;
    set_status(`Downloading ${asset.description}…`);
    progress_element.hidden = false;
    progress_element.removeAttribute("value");
    const response = await fetch(url);
    if(!response.ok){
        throw new Error(`Could not download ${asset.description}: HTTP ${response.status}.\nFor offline development, run static/hyperdrone/link_local_assets.sh and open this page with ?assets=local.`);
    }
    const content_length = Number(response.headers.get("Content-Length"));
    const total = content_length || asset.size_mb * 1024 * 1024;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    while(true){
        const {done, value} = await reader.read();
        if(done){
            break;
        }
        chunks.push(value);
        received += value.length;
        const downloaded = (received / 1024 / 1024).toFixed(1);
        set_status(`Downloading ${asset.description}: ${downloaded} / ${content_length ? "" : "~"}${(total / 1024 / 1024).toFixed(1)} MB`);
        if(content_length){
            progress_element.max = total;
            progress_element.value = received;
        }
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    for(const chunk of chunks){
        bytes.set(chunk, offset);
        offset += chunk.length;
    }
    set_status(`Verifying ${asset.description}…`);
    const digest = await window.crypto.subtle.digest("SHA-1", bytes);
    const hex = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
    if(hex !== asset.sha1){
        throw new Error(`Integrity check failed for ${asset.description}: expected ${asset.sha1}, received ${hex}.`);
    }
    progress_element.hidden = true;
    return bytes;
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
    const {default: createModule} = await import("./build/hyperdrone_web.js");
    const module = await createModule({
        print: text => console.log(text),
        printErr: text => console.error(text),
        onAbort: reason => show_error(`Renderer stopped: ${reason}`),
        rlt_status: set_status,
        rlt_input: poll_input,
        rlt_present: (pointer, width, height, num_views) => {
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
            if(!ready){
                ready = true;
                document.body.classList.add("ready");
                reset_button.disabled = false;
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
                document.title = "done";
                set_status(`Rendered ${frame_count} frames`);
            }
        },
    });
    const drone = DRONES[drone_index];
    module.FS.writeFile("/scene.glb", await fetch_asset(SCENE));
    module.FS.writeFile("/drone.glb", await fetch_asset({...drone, description: `${drone.label} airframe`}));
    module.FS.writeFile("/policy.tar", await fetch_asset(POLICY));
    set_status("Preparing the scene and cameras…");
    const result = await module.ccall("demo_run", "number", ["string", "string", "string", "number", "number"], ["/scene.glb", "/drone.glb", "/policy.tar", drone_index, max_frames], {async: true});
    if(result !== 0){
        throw new Error(`The renderer exited with status ${result}.`);
    }
}

main().catch(show_error);
