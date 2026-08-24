import createModule from "./build/drone_web.js";

const SCENE = {description: "ProcTHOR-Train-1.glb", sha1: "7f1c9129532798e0b63bc41edb6b4c09251cf8a0", size_mb: 32};
const DRONES = [
    {name: "crazyflie",           sha1: "87d2b8b1d518445872d34906320a76656731902f", size_mb: 5},
    {name: "crazyflie_brushless", sha1: "eedb897237c1d77ed8ef0681554d92d06a7c27ca", size_mb: 19},
    {name: "savagebee_pusher",    sha1: "fd4eddc7c4a3f74e51823a2fa392bb96e095e53e", size_mb: 13},
    {name: "arpl",                sha1: "8398a239fd1cd80df8adab67497baf7be3d8cda4", size_mb: 46},
    {name: "x500",                sha1: "f6681e7a8b7fa7ef023bedcd795981becb731d0e", size_mb: 74},
    {name: "soft",                sha1: "fcdf53a791f313b9bb07f775b10b9e4c2bfa0ae9", size_mb: 113},
];
const DEFAULT_ASSET_BASE = "https://huggingface.co/datasets/rl-tools/conta/resolve/main/data/";

const params = new URLSearchParams(window.location.search);
const asset_base = params.get("assets") === "local" ? "./assets/" : (params.get("assets") || DEFAULT_ASSET_BASE);
const max_frames = parseInt(params.get("frames") || "0", 10);
const drone_name = params.get("drone") || "crazyflie";

const status_element = document.getElementById("status");
const error_element = document.getElementById("error");
const fps_element = document.getElementById("fps");
const drone_select = document.getElementById("drone");
for(const drone of DRONES){
    const option = document.createElement("option");
    option.value = drone.name;
    option.textContent = `${drone.name} (${drone.size_mb} MB)`;
    option.selected = drone.name === drone_name;
    drone_select.appendChild(option);
}
drone_select.addEventListener("change", () => {
    params.set("drone", drone_select.value);
    window.location.search = params.toString();
});

function set_status(text){ status_element.textContent = text; }
function fail(text){
    error_element.textContent = text;
    set_status("Failed");
    document.title = "failed";
    throw new Error(text);
}

async function fetch_asset(sha1, label, size_mb){
    const url = asset_base + sha1;
    const response = await fetch(url);
    if(!response.ok){
        fail(`Failed to fetch ${label} (${url}): HTTP ${response.status}\n` +
             `For local development run static/raytracing/link_local_assets.sh and open with ?assets=local`);
    }
    const total = parseInt(response.headers.get("Content-Length") || "0", 10) || size_mb * 1024 * 1024;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    while(true){
        const {done, value} = await reader.read();
        if(done){ break; }
        chunks.push(value);
        received += value.length;
        set_status(`Downloading ${label}: ${(received / 1024 / 1024).toFixed(1)} / ~${(total / 1024 / 1024).toFixed(0)} MB`);
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    for(const chunk of chunks){ bytes.set(chunk, offset); offset += chunk.length; }
    if(window.crypto && window.crypto.subtle){
        const digest = await window.crypto.subtle.digest("SHA-1", bytes);
        const hex = Array.from(new Uint8Array(digest)).map(byte => byte.toString(16).padStart(2, "0")).join("");
        if(hex !== sha1){
            fail(`Integrity check failed for ${label}: expected ${sha1}, got ${hex}`);
        }
    }
    return bytes;
}

const contexts = [0, 1].map(camera_i => document.getElementById(`camera${camera_i}`).getContext("2d"));
let images = null;
let frame_count = 0;
let fps_window_start = performance.now();
let fps_window_frames = 0;

async function main(){
    if(!navigator.gpu){
        if(!window.isSecureContext){
            fail("navigator.gpu is only exposed on secure contexts (https or localhost), and this page was loaded from an insecure origin. " +
                 "Open it via localhost (e.g. ssh -L 8000:localhost:8000 <host>), or add this origin under chrome://flags/#unsafely-treat-insecure-origin-as-secure.");
        }
        fail("This browser does not expose WebGPU (navigator.gpu). Use a recent Chromium-based browser; on Linux, chrome://flags/#enable-unsafe-webgpu may be required.");
    }
    try{
        const adapter = await navigator.gpu.requestAdapter({powerPreference: "high-performance"});
        const info = (adapter && adapter.info) || {};
        document.getElementById("adapter").textContent = [info.vendor, info.architecture, info.description].filter(Boolean).join(" ");
    }catch(error){ console.warn(error); }
    const drone = DRONES.find(entry => entry.name === drone_select.value) || DRONES[0];

    set_status("Loading WASM module ...");
    const module = await createModule({
        print: text => console.log(text),
        printErr: text => console.error(text),
        onAbort: reason => fail(`Renderer aborted: ${reason}`),
        rlt_status: text => set_status(text),
        rlt_present: (pointer, width, height, num_cameras) => {
            if(images === null || images[0].width !== width){
                images = contexts.map(context => context.createImageData(width, height));
            }
            const heap = module.HEAPU8;
            for(let camera_i = 0; camera_i < num_cameras; camera_i++){
                const begin = pointer + camera_i * width * height * 4;
                images[camera_i].data.set(heap.subarray(begin, begin + width * height * 4));
                contexts[camera_i].putImageData(images[camera_i], 0, 0);
            }
            frame_count++;
            fps_window_frames++;
            const now = performance.now();
            if(now - fps_window_start > 1000){
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

    const [scene_bytes, drone_bytes] = [
        await fetch_asset(SCENE.sha1, SCENE.description, SCENE.size_mb),
        await fetch_asset(drone.sha1, `${drone.name}.glb`, drone.size_mb),
    ];
    module.FS.writeFile("/scene.glb", scene_bytes);
    module.FS.writeFile("/drone.glb", drone_bytes);

    set_status("Starting renderer ...");
    const result = await module.ccall("demo_run", "number", ["string", "string", "number"], ["/scene.glb", "/drone.glb", max_frames], {async: true});
    if(result !== 0){
        fail(`Renderer exited with status ${result}`);
    }
}

main().catch(error => {
    console.error(error);
    if(error_element.textContent === ""){
        error_element.textContent = String(error);
    }
});
