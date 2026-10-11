import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {test} from "node:test";
import vm from "node:vm";
import {create_touch_stick} from "../../../static/hyperdrone/touch-stick.js";

const page = readFileSync(new URL("../../../static/hyperdrone/index.html", import.meta.url), "utf8");
const source = readFileSync(new URL("../../../static/hyperdrone/main.js", import.meta.url), "utf8")
    .replace(/^import .*;\n/gm, "")
    .replace('await import("./blob/build/hyperdrone_web.js")', "await import_module()")
    .replace(/main\(\)\.catch\(show_error\);\s*$/, "");

function send(target, type, fields = {}){
    const event = new Event(type, {cancelable: true});
    Object.assign(event, {pointerId: 1, pointerType: "touch", button: 0, clientX: 60, clientY: 60, ...fields});
    target.dispatchEvent(event);
    return event;
}

const default_configuration = {id: "high-mb4-aa1-480x270", fidelity: "high", motion_blur: true, antialiasing: false, width: 480, height: 270};
const configurations = [];
for(const fidelity of ["low", "medium", "high", "very-high"]){
    for(const motion_blur of [false, true]){
        for(const antialiasing of [false, true]){
            for(const [width, height] of [[480, 270], [960, 540]]){
                configurations.push({id: `${fidelity}-mb${motion_blur ? 4 : 1}-aa${antialiasing ? 2 : 1}-${width}x${height}`, fidelity, motion_blur, antialiasing, width, height});
            }
        }
    }
}

function fixture({defer_mouse_lock = false, search = "", catalog = configurations, api_version = 5, fpv_position = [0.08, 0, 0.08]} = {}){
    const document = new EventTarget();
    const window = new EventTarget();
    const classes = new Set();
    document.body = {classList: {
        add: name => classes.add(name),
        remove: name => classes.delete(name),
        toggle(name, on){ on ? classes.add(name) : classes.delete(name); },
    }};
    document.hidden = false;
    document.pointerLockElement = null;
    document.exitPointerLock = () => {
        document.pointerLockElement = null;
        send(document, "pointerlockchange");
    };
    window.location = {search};
    window.isSecureContext = true;
    document.createElement = () => ({});
    const elements = new Map();
    const pending_locks = [];
    for(const match of page.matchAll(/<[^>]*\bid="([^"]+)"[^>]*>/g)){
        const element = new EventTarget();
        element.id = match[1];
        element.value = element.defaultValue = match[0].match(/\bvalue="([^"]+)"/)?.[1] || "";
        element.min = match[0].match(/\bmin="([^"]+)"/)?.[1] || "";
        element.max = match[0].match(/\bmax="([^"]+)"/)?.[1] || "";
        element.disabled = /\bdisabled\b/.test(match[0]);
        element.hidden = /\bhidden\b/.test(match[0]);
        element.open = false;
        element.attributes = new Map();
        element.options = [];
        element.setAttribute = (name, value) => element.attributes.set(name, value);
        element.removeAttribute = name => element.attributes.delete(name);
        element.appendChild = option => {
            element.options.push(option);
            if(option.selected || element.options.length === 1){
                element.value = option.value;
            }
        };
        element.replaceChildren = () => { element.options = []; element.value = ""; };
        element.showModal = () => { element.open = true; };
        element.close = () => { element.open = false; send(element, "close"); };
        const drawing = {
            createImageData: (width, height) => ({width, height, data: new Uint8ClampedArray(width * height * 4)}),
            putImageData: image => { element.image = image; },
        };
        element.getContext = () => drawing;
        element.captures = new Set();
        element.setPointerCapture = id => element.captures.add(id);
        element.hasPointerCapture = id => element.captures.has(id);
        element.releasePointerCapture = id => {
            element.captures.delete(id);
            send(element, "lostpointercapture", {pointerId: id});
        };
        element.getBoundingClientRect = () => ({left: 0, top: 0, width: 120, height: 120});
        element.querySelector = () => ({
            style: {setProperty(){}},
            getBoundingClientRect: () => ({width: 20, height: 20}),
        });
        element.requestPointerLock = () => new Promise(resolve => {
            const grant = () => {
                document.pointerLockElement = element;
                send(document, "pointerlockchange");
                resolve();
            };
            if(defer_mouse_lock){
                pending_locks.push(grant);
            }
            else{
                grant();
            }
        });
        elements.set(element.id, element);
    }
    document.getElementById = id => {
        assert.ok(elements.has(id), `Missing page element: ${id}`);
        return elements.get(id);
    };
    let module_options;
    let abort_reason;
    const errors = [];
    const calls = [];
    const events = [];
    const scenes = [106, 107].map(id => ({name: `ProcTHOR-Train-${id}`, reference: `scene-${id}`}));
    const module = {
        _demo_api_version: () => api_version,
        HEAPU8: new Uint8Array(),
        FS: {writeFile(){}},
        ccall(name, return_type, argument_types, args){
            calls.push({name, return_type, argument_types: Array.from(argument_types), args: Array.from(args)});
            events.push(name);
            if(name === "demo_configurations"){
                return JSON.stringify(catalog);
            }
            if(name === "demo_scenes"){
                return JSON.stringify(scenes);
            }
            assert.equal(name, "demo_run");
            if(abort_reason !== undefined){
                module_options.onAbort(abort_reason);
                throw new WebAssembly.RuntimeError("Aborted(). Build with -sASSERTIONS for more info.");
            }
            return 0;
        },
    };
    const context = vm.createContext({
        document, window, navigator: {maxTouchPoints: 2, gpu: {}}, performance, URLSearchParams, create_touch_stick,
        Conta: class {async load(reference){ events.push(`asset:${reference}`); return new Uint8Array(); }},
        console: {...console, error: error => errors.push(error)},
        import_module: async () => ({default: async options => {
            module_options = options;
            return module;
        }}),
        fetch: async path => {
            events.push(path);
            return {ok: true, json: async () => path === "registry.json"
                ? {drones: [{name: "ARPL", reference: "drone", dynamics_index: 3, fpv_position}], policy: {reference: "policy"}}
                : {scene: scenes[0].name, reference: scenes[0].reference, position: [0, 0, 1], yaw_degrees: 0}};
        },
    });
    vm.runInContext(`${source}\nglobalThis.controls = {poll_input, main: () => main().catch(show_error), start(){ ready = true; update_flight_state(); }};`, context);
    const element = id => document.getElementById(id);
    const click = (id, pointerType = "touch") => send(element(id), "click", {pointerType});
    const poll = () => Array.from(context.controls.poll_input(), value => value || 0);
    const axes = () => { const input = poll(); return [...input.slice(0, 4), input[11]]; };
    const grant_mouse_lock = () => {
        assert.ok(pending_locks.length > 0, "A mouse capture request must be pending");
        pending_locks.shift()();
    };
    return {
        document, window, classes, element, click, poll, axes, grant_mouse_lock, errors, calls, events,
        pending_mouse_locks: () => pending_locks.length,
        start: context.controls.start,
        load(reason){ abort_reason = reason; return context.controls.main(); },
        abort: reason => module_options.onAbort(reason),
        motor_layout: (deviation, wheelbase, threshold) => module_options.rlt_motor_layout(deviation, wheelbase, threshold),
        present(width, height, num_views = 6){
            const pointer = 32;
            const image_bytes = width * height * 4;
            module.HEAPU8 = new Uint8Array(pointer + num_views * image_bytes);
            for(let view = 0; view < num_views; view++){
                module.HEAPU8.fill(view + 1, pointer + view * image_bytes, pointer + (view + 1) * image_bytes);
            }
            module_options.rlt_present(pointer, width, height, num_views, 0, 0, 1, 1, 0, 0, 0);
        },
    };
}

function touch_flight(f){
    f.start();
    f.click("camera0");
    send(f.element("left-stick"), "pointerdown", {pointerId: 1, clientX: 110});
    send(f.element("right-stick"), "pointerdown", {pointerId: 2, clientY: 10});
}

function select_rendering(f, configuration){
    for(const [id, value] of [
        ["render-fidelity", configuration.fidelity],
        ["render-motion-blur", String(configuration.motion_blur)],
        ["render-antialiasing", String(configuration.antialiasing)],
        ["render-resolution", `${configuration.width}x${configuration.height}`],
    ]){
        f.element(id).value = value;
        send(f.element(id), "change");
    }
}

test("the full rendering catalog supplies the existing default and every compiled combination", async () => {
    const f = fixture();
    await f.load();
    assert.deepEqual(f.errors, []);
    assert.deepEqual(f.calls.find(call => call.name === "demo_run"), {
        name: "demo_run", return_type: "number",
        argument_types: ["string", "string", "string", "number", "number", "string", "string"],
        args: ["/scene.glb", "/drone.glb", "/policy.tar", 3, 0, JSON.stringify({fpv_position: [0.08, 0, 0.08], position: [0, 0, 1], yaw: 0}), default_configuration.id],
    });
    assert.ok(f.events.indexOf("demo_configurations") < f.events.findIndex(event => event.startsWith("asset:")));
    assert.deepEqual(f.element("render-fidelity").options.map(option => option.value), ["low", "medium", "high", "very-high"]);
    assert.deepEqual(f.element("render-resolution").options.map(option => option.value), ["480x270", "960x540"]);
    for(const configuration of configurations){
        select_rendering(f, configuration);
        assert.equal(f.element("render-apply").disabled, false, configuration.id);
        f.click("render-apply");
        assert.equal(new URLSearchParams(f.window.location.search).get("render"), configuration.id);
    }
});

test("a single-configuration build selects its only configuration and disables fixed settings", async () => {
    const only = configurations[0];
    const f = fixture({catalog: [only]});
    await f.load();
    assert.deepEqual(f.errors, []);
    assert.equal(f.calls.find(call => call.name === "demo_run").args.at(-1), only.id);
    assert.equal(f.element("rendering").disabled, false);
    for(const id of ["render-fidelity", "render-motion-blur", "render-antialiasing", "render-resolution"]){
        assert.equal(f.element(id).disabled, true, id);
        assert.equal(f.element(id).options.length, 1, id);
    }
    f.click("rendering");
    assert.equal(f.element("rendering-settings").open, true);
    assert.equal(f.element("render-apply").disabled, false);
});

test("a sparse catalog validates complete tuples while leaving disconnected configurations reachable", async () => {
    const other = configurations.find(configuration => configuration.id === "low-mb1-aa2-960x540");
    const f = fixture({catalog: [other, default_configuration]});
    await f.load();
    assert.equal(f.calls.find(call => call.name === "demo_run").args.at(-1), default_configuration.id);
    f.element("render-fidelity").value = other.fidelity;
    send(f.element("render-fidelity"), "change");
    assert.equal(f.element("render-apply").disabled, true);
    assert.ok(f.element("render-availability").textContent.length > 0);
    f.click("render-apply");
    assert.equal(f.window.location.search, "");
    for(const id of ["render-motion-blur", "render-antialiasing", "render-resolution"]){
        assert.equal(f.element(id).disabled, false, id);
        assert.ok(f.element(id).options.every(option => !option.disabled), id);
    }
    select_rendering(f, other);
    assert.equal(f.element("render-apply").disabled, false);
    f.click("render-apply");
    assert.equal(new URLSearchParams(f.window.location.search).get("render"), other.id);
    f.click("render-defaults");
    assert.equal(f.element("render-fidelity").value, default_configuration.fidelity);
    assert.equal(f.element("render-motion-blur").value, String(default_configuration.motion_blur));
    assert.equal(f.element("render-antialiasing").value, String(default_configuration.antialiasing));
    assert.equal(f.element("render-resolution").value, "480x270");
    assert.equal(f.element("render-apply").disabled, false);
});

test("URL rendering choices reach WASM and survive scene and airframe changes", async () => {
    const selected = configurations.at(-1);
    const f = fixture({search: `?scene=ProcTHOR-Train-106&drone=ARPL&frames=2&render=${selected.id}`});
    await f.load();
    const run = f.calls.find(call => call.name === "demo_run");
    assert.equal(run.args.at(-1), selected.id);
    assert.equal(run.args[4], 2);
    assert.equal(f.element("render-resolution").value, "960x540");
    f.element("scene").value = "ProcTHOR-Train-107";
    send(f.element("scene"), "change");
    send(f.element("drone"), "change");
    let params = new URLSearchParams(f.window.location.search);
    assert.equal(params.get("render"), selected.id);
    assert.equal(params.get("scene"), "ProcTHOR-Train-107");
    assert.equal(params.get("drone"), "ARPL");
    select_rendering(f, default_configuration);
    f.click("render-apply");
    params = new URLSearchParams(f.window.location.search);
    assert.equal(params.get("render"), default_configuration.id);
    assert.equal(params.get("scene"), "ProcTHOR-Train-107");
    assert.equal(params.get("drone"), "ARPL");
    assert.equal(params.get("frames"), "2");
});

test("an unavailable URL rendering choice can be corrected before downloading scene assets", async () => {
    for(const requested of ["not-compiled", ""]){
        const f = fixture({search: `?render=${requested}&frames=2`});
        await f.load();
        assert.equal(f.element("error").hidden, false);
        assert.match(f.element("error").textContent, /render|configuration/i);
        assert.equal(f.events.some(event => event.startsWith("asset:")), false);
        assert.equal(f.calls.some(call => call.name === "demo_run"), false);
        assert.equal(f.element("rendering").disabled, false);
        f.click("rendering");
        assert.equal(f.element("rendering-settings").open, true);
        assert.equal(f.element("render-apply").disabled, false);
        f.click("render-apply");
        const params = new URLSearchParams(f.window.location.search);
        assert.equal(params.get("render"), default_configuration.id);
        assert.equal(params.get("frames"), "2");
    }
});

test("empty or incompatible bundles fail before fetching scene assets", async () => {
    for(const options of [{catalog: []}, {api_version: 1}, {api_version: 2}, {api_version: 3}, {api_version: 4}]){
        const f = fixture(options);
        await f.load();
        assert.equal(f.element("error").hidden, false);
        assert.equal(f.element("rendering").disabled, true);
        assert.equal(f.events.some(event => event.startsWith("asset:")), false);
        assert.equal(f.calls.some(call => call.name === "demo_run"), false);
    }
});

test("camera FOV sliders update independently at runtime and survive a flight reset", async () => {
    const f = fixture({catalog: [default_configuration]});
    await f.load();
    f.start();
    f.click("rendering");
    assert.deepEqual(f.poll().slice(12), [80, 50]);
    assert.equal(f.element("onboard-fov-value").textContent, "80°");
    assert.equal(f.element("external-fov-value").textContent, "50°");
    for(const [id, value, expected] of [["onboard-fov", "110", [110, 50]], ["external-fov", "30", [110, 30]]]){
        assert.equal(f.element(id).disabled, false);
        f.element(id).value = value;
        send(f.element(id), "input");
        assert.equal(f.element(`${id}-value`).textContent, `${value}°`);
        const input = f.poll();
        assert.deepEqual(input.slice(12), expected);
        assert.equal(input[4], 0);
        assert.equal(input[5], 1);
        assert.equal(f.window.location.search, "");
        assert.equal(f.calls.filter(call => call.name === "demo_run").length, 1);
    }
    f.click("reset");
    const reset_input = f.poll();
    assert.equal(reset_input[4], 1);
    assert.deepEqual(reset_input.slice(12), [110, 30]);
    f.click("render-defaults");
    assert.deepEqual(f.poll().slice(12), [80, 50]);
    assert.equal(f.element("onboard-fov-value").textContent, "80°");
    assert.equal(f.element("external-fov-value").textContent, "50°");
});

test("FOV choices survive renderer, scene, and airframe changes", async () => {
    const f = fixture({search: "?onboard-fov=95&external-fov=60"});
    await f.load();
    assert.deepEqual(f.poll().slice(12), [95, 60]);
    f.element("onboard-fov").value = "140";
    send(f.element("onboard-fov"), "input");
    for(const id of ["scene", "drone"]){
        send(f.element(id), "change");
        const params = new URLSearchParams(f.window.location.search);
        assert.equal(params.get("onboard-fov"), "140");
        assert.equal(params.get("external-fov"), "60");
    }
    f.click("render-apply");
    const restored = fixture({search: f.window.location.search});
    assert.deepEqual(restored.poll().slice(12), [140, 60]);
    f.click("render-defaults");
    f.click("render-apply");
    const defaults = new URLSearchParams(f.window.location.search);
    assert.equal(defaults.has("onboard-fov"), false);
    assert.equal(defaults.has("external-fov"), false);
});

test("invalid FOV URL values restore safe camera defaults", () => {
    for(const value of ["", "NaN", "Infinity", "-1", "0", "29", "141", "180", "wide"]){
        const f = fixture({search: `?onboard-fov=${value}&external-fov=${value}`});
        assert.deepEqual(f.poll().slice(12), [80, 50], value);
    }
});

test("motor layout warnings use the native threshold and allow rendering to continue", async () => {
    const f = fixture();
    await f.load();
    const warning = f.element("motor-layout-warning");
    assert.equal(warning.hidden, true);
    f.motor_layout(0.068, 0.25, 0.05);
    assert.equal(warning.hidden, false);
    assert.match(warning.textContent, /6\.8% of wheelbase \(17\.0 mm horizontally after scaling; threshold 5%\)/);
    f.present(480, 270);
    assert.equal(warning.hidden, false);
    assert.equal(f.element("reset").disabled, false);
    assert.equal(f.element("error").hidden, true);
    assert.deepEqual(f.errors, []);
    for(const deviation of [0, 0.02755, 0.05]){
        f.motor_layout(deviation, 0.08, 0.05);
        assert.equal(warning.hidden, true);
        assert.equal(warning.textContent, "");
    }
});

test("each airframe mount reaches a sampled scene without overriding its spawn", async () => {
    const fpv_position = [0.016, 0, 0.026];
    const f = fixture({search: "?scene=ProcTHOR-Train-107", fpv_position});
    await f.load();
    assert.deepEqual(f.errors, []);
    assert.deepEqual(JSON.parse(f.calls.find(call => call.name === "demo_run").args[5]), {fpv_position});
});

test("invalid airframe mounts fail before fetching assets", async () => {
    for(const fpv_position of [null, [0, 0], [0, 0, NaN], [0, 0, "0.1"]]){
        const f = fixture({fpv_position});
        await f.load();
        assert.match(f.element("error").textContent, /camera position/);
        assert.equal(f.events.some(event => event.startsWith("asset:")), false);
        assert.equal(f.calls.some(call => call.name === "demo_run"), false);
    }
});

test("all six outputs resize and retain distinct image strides at both rendering resolutions", async () => {
    const f = fixture();
    await f.load();
    const ids = ["camera0", "camera1", "depth", "segmentation", "normals", "flow"];
    for(const [width, height] of [[480, 270], [960, 540], [480, 270]]){
        f.present(width, height);
        for(const [view, id] of ids.entries()){
            const canvas = f.element(id);
            assert.equal(canvas.width, width, id);
            assert.equal(canvas.height, height, id);
            assert.equal(canvas.image.data.length, width * height * 4, id);
            assert.ok(canvas.image.data.every(value => value === view + 1), id);
        }
    }
    assert.throws(() => f.present(480, 270, 5), /Expected 6 camera outputs/);
});

for(const mode of ["touch", "mouse", "pending mouse"]){
    test(`Rendering releases ${mode} control and blocks recapture until dismissed`, async () => {
        const f = fixture({defer_mouse_lock: mode === "pending mouse"});
        await f.load();
        f.start();
        if(mode === "touch"){
            touch_flight(f);
        }
        else{
            f.click("camera0", "mouse");
            send(f.document, "keydown", {code: "KeyW"});
            send(f.document, "mousemove", {movementX: 100});
        }
        f.click("rendering");
        assert.equal(f.element("rendering-settings").open, true);
        assert.equal(f.document.pointerLockElement, null);
        assert.equal(f.classes.has("controlling"), false);
        assert.equal(f.element("left-stick").captures.size, 0);
        assert.equal(f.element("right-stick").captures.size, 0);
        assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
        if(mode === "pending mouse"){
            f.grant_mouse_lock();
            await Promise.resolve();
            assert.equal(f.document.pointerLockElement, null);
        }
        for(const pointerType of ["touch", "mouse"]){
            f.click("camera0", pointerType);
        }
        send(f.element("camera1"), "keydown", {code: "Enter"});
        send(f.element("left-stick"), "pointerdown", {clientX: 110});
        assert.equal(f.pending_mouse_locks(), 0);
        assert.equal(f.document.pointerLockElement, null);
        assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
        f.element("rendering-settings").close();
        assert.equal(f.classes.has("controlling"), false);
        f.click("camera0");
        assert.equal(f.classes.has("touch-controlling"), true);
        assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
    });
}

test("an empty spawn search keeps its scene-specific error after the runtime rejection and input events", async () => {
    const f = fixture();
    const reason = "annotations::free_space: no valid positions found";
    await f.load(reason);
    const message = "Could not find a suitable starting position in ProcTHOR-Train-106. Choose another scene.";
    assert.equal(f.element("error").textContent, message);
    assert.equal(f.element("error").hidden, false);
    assert.equal(f.errors[0], reason);
    assert.match(f.errors[1].message, /Build with -sASSERTIONS/);
    for(const [target, type] of [[f.window, "blur"], [f.document, "visibilitychange"], [f.document, "pointerlockerror"]]){
        send(target, type);
    }
    f.click("camera0", "mouse");
    f.click("camera1");
    send(f.document, "keydown", {code: "KeyW"});
    assert.equal(f.element("error").textContent, message);
    assert.equal(f.element("flight-state").textContent, "Unavailable");
    assert.equal(f.element("status").textContent, "The simulation could not start.");
    assert.equal(f.element("reset").disabled, true);
    assert.equal(f.element("fps").hidden, true);
    assert.equal(f.document.pointerLockElement, null);
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
});

test("scene selection reloads the page after an empty spawn search", async () => {
    const f = fixture();
    await f.load("annotations::free_space: no valid positions found");
    assert.equal(f.element("scene").disabled, false);
    assert.equal(f.element("drone").disabled, false);
    f.element("scene").value = "ProcTHOR-Train-107";
    send(f.element("scene"), "change");
    assert.equal(new URLSearchParams(f.window.location.search).get("scene"), "ProcTHOR-Train-107");
    assert.equal(f.element("flight-state").textContent, "Unavailable");
});

test("fatal errors retain other diagnostics and handle an unavailable scene name", async () => {
    const generic = fixture();
    await generic.load("WebGPU: staging buffer map failed");
    assert.equal(generic.element("error").textContent, "Renderer stopped: WebGPU: staging buffer map failed");

    const unnamed = fixture();
    await unnamed.load();
    unnamed.element("scene").value = "";
    unnamed.abort("annotations::free_space: no valid positions found");
    assert.equal(unnamed.element("error").textContent, "Could not find a suitable starting position in this scene. Choose another scene.");
});

test("loading rejects control and desktop capture polls held keys with one-shot mouse motion", () => {
    const f = fixture();
    f.click("camera0", "mouse");
    f.click("camera0");
    assert.equal(f.document.pointerLockElement, null);
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
    f.start();
    f.click("camera0", "mouse");
    assert.equal(f.document.pointerLockElement, f.element("camera0"));
    for(const code of ["KeyW", "KeyA", "KeyE"]){
        assert.equal(send(f.document, "keydown", {code}).defaultPrevented, true);
    }
    send(f.document, "mousemove", {movementX: 10});
    assert.deepEqual(f.axes(), [1, 1, 1, -0.04, 0]);
    assert.deepEqual(f.axes(), [1, 1, 1, 0, 0]);
    send(f.document, "keyup", {code: "KeyW"});
    assert.deepEqual(f.axes(), [0, 1, 1, 0, 0]);
    f.click("camera1", "mouse");
    assert.equal(f.document.pointerLockElement, null);
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
});

test("touch holds yaw and translation across polls, ignores desktop inputs, and clears fingers independently", () => {
    const f = fixture();
    touch_flight(f);
    send(f.document, "keydown", {code: "KeyS"});
    send(f.document, "mousemove", {movementX: 100});
    assert.deepEqual(f.axes(), [1, 0, 0, 0, -0.5]);
    assert.deepEqual(f.axes(), [1, 0, 0, 0, -0.5]);
    send(f.element("left-stick"), "pointermove", {pointerId: 1, clientY: 10});
    assert.deepEqual(f.axes(), [1, 0, 0.7, 0, 0]);
    send(f.element("right-stick"), "pointercancel", {pointerId: 2});
    assert.deepEqual(f.axes(), [0, 0, 0.7, 0, 0]);
    send(f.element("left-stick"), "pointerup", {pointerId: 1});
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
    assert.equal(f.element("flight-state").textContent, "User control");
    assert.equal(f.element("release-control").hidden, false);
});

test("browsers without pointer lock can activate and drag the on-screen sticks with a mouse", () => {
    const f = fixture();
    f.element("camera0").requestPointerLock = undefined;
    f.start();
    f.click("camera0", "mouse");
    assert.equal(f.classes.has("touch-controlling"), true);
    assert.equal(f.document.pointerLockElement, null);
    send(f.element("right-stick"), "pointerdown", {pointerType: "mouse", clientY: 10});
    assert.deepEqual(f.axes(), [1, 0, 0, 0, 0]);
    send(f.element("right-stick"), "pointerup", {pointerType: "mouse"});
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
    f.click("release-control", "mouse");
    assert.equal(f.classes.has("controlling"), false);
    f.click("touch-tuning");
    send(f.element("camera0"), "keydown", {code: "Enter"});
    assert.equal(f.classes.has("controlling"), false);
});

test("mouse and touch handoffs clear the previous input source", () => {
    const f = fixture();
    touch_flight(f);
    f.click("camera1", "mouse");
    assert.equal(f.classes.has("touch-ui"), false);
    assert.equal(f.element("left-stick").captures.size, 0);
    assert.equal(f.element("right-stick").captures.size, 0);
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
    send(f.document, "keydown", {code: "KeyD"});
    send(f.document, "mousemove", {movementX: 100});
    f.click("camera0");
    assert.equal(f.document.pointerLockElement, null);
    assert.equal(f.classes.has("touch-ui"), true);
    assert.equal(f.classes.has("touch-controlling"), true);
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
});

test("a delayed mouse grant cannot override newer touch activation", async () => {
    const f = fixture({defer_mouse_lock: true});
    f.start();
    f.click("camera0", "mouse");
    assert.equal(f.document.pointerLockElement, null);
    f.click("camera1");
    f.grant_mouse_lock();
    await Promise.resolve();
    assert.equal(f.document.pointerLockElement, null);
    assert.equal(f.classes.has("touch-controlling"), true);
    assert.equal(f.classes.has("touch-ui"), true);
    assert.equal(f.element("release-control").hidden, false);
    send(f.element("left-stick"), "pointerdown", {clientX: 110});
    assert.deepEqual(f.axes(), [0, 0, 0, 0, -0.5]);
    f.click("camera0", "mouse");
    f.grant_mouse_lock();
    await Promise.resolve();
    assert.equal(f.document.pointerLockElement, f.element("camera0"));
    assert.equal(f.classes.has("mouse-controlling"), true);
    assert.equal(f.classes.has("touch-controlling"), false);
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
});

test("a delayed mouse grant cannot reacquire control after release", async () => {
    const f = fixture({defer_mouse_lock: true});
    f.start();
    f.click("camera0", "mouse");
    f.click("camera1");
    f.click("release-control");
    f.grant_mouse_lock();
    await Promise.resolve();
    assert.equal(f.document.pointerLockElement, null);
    assert.equal(f.classes.has("controlling"), false);
    assert.equal(f.element("flight-state").textContent, "Position hold");
    assert.equal(f.element("release-control").hidden, true);
    send(f.document, "keydown", {code: "KeyW"});
    send(f.document, "mousemove", {movementX: 100});
    send(f.element("left-stick"), "pointerdown", {clientX: 110});
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
});

test("losing mouse capture clears held keys and yaw before keyboard recapture", () => {
    const f = fixture();
    f.start();
    send(f.element("camera0"), "keydown", {code: "Enter"});
    send(f.document, "keydown", {code: "KeyW"});
    send(f.document, "mousemove", {movementX: 100});
    f.document.exitPointerLock();
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
    send(f.element("camera1"), "keydown", {code: "Space"});
    assert.equal(f.document.pointerLockElement, f.element("camera1"));
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
});

test("touch and pen clicks can use the preceding pointerdown type", () => {
    for(const pointerType of ["touch", "pen"]){
        const f = fixture();
        f.start();
        send(f.element("camera0"), "pointerdown", {pointerType});
        send(f.element("camera0"), "click", {pointerType: undefined});
        assert.equal(f.document.pointerLockElement, null);
        assert.equal(f.classes.has("touch-controlling"), true);
    }
});

const releases = {
    button: f => f.click("release-control"),
    escape: f => send(f.document, "keydown", {code: "Escape"}),
    reset: f => f.click("reset"),
    blur: f => send(f.window, "blur"),
    pagehide: f => send(f.window, "pagehide"),
    resize: f => send(f.window, "resize"),
    hidden: f => { f.document.hidden = true; send(f.document, "visibilitychange"); },
    scene: f => send(f.element("scene"), "change"),
    drone: f => send(f.element("drone"), "change"),
    tuning: f => f.click("touch-tuning"),
    fps: f => f.click("fps"),
};
for(const [name, release] of Object.entries(releases)){
    test(`${name} releases both sticks without retaining input after return`, () => {
        const f = fixture();
        touch_flight(f);
        release(f);
        const input = f.poll();
        assert.deepEqual([...input.slice(0, 4), input[11]], [0, 0, 0, 0, 0]);
        assert.equal(input[4], Number(name === "reset"));
        assert.equal(input[5], Number(name !== "hidden"));
        assert.equal(f.element("left-stick").captures.size, 0);
        assert.equal(f.element("right-stick").captures.size, 0);
        assert.equal(f.element("release-control").hidden, true);
        f.document.hidden = false;
        f.element("flight-tuning").close();
        f.element("frame-rate-tuning").close();
        send(f.document, "visibilitychange");
        send(f.element("left-stick"), "pointermove", {pointerId: 1, clientX: 110});
        assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
        assert.equal(f.element("flight-state").textContent, "Position hold");
    });
}

test("dialogs block recapture and defaults restore the live input contract", () => {
    const f = fixture();
    touch_flight(f);
    send(f.document, "keydown", {code: "KeyT", shiftKey: true});
    assert.equal(f.element("flight-tuning").open, true);
    f.click("camera0");
    send(f.element("left-stick"), "pointerdown", {clientX: 110});
    assert.deepEqual(f.axes(), [0, 0, 0, 0, 0]);
    for(const id of ["position-offset", "velocity-target", "yaw-rate-limit", "yaw-error-limit", "mouse-sensitivity"]){
        f.element(id).value = "1";
    }
    f.click("tuning-defaults");
    f.element("target-fps").value = "10";
    f.click("frame-rate-defaults");
    assert.deepEqual(f.poll().slice(6, 11), [Math.PI, 70 * Math.PI / 180, 0.3, 0.7, 60]);
    assert.equal(f.element("mouse-sensitivity").value, "2");
    send(f.document, "keydown", {code: "KeyT", shiftKey: true});
    assert.equal(f.element("flight-tuning").open, false);
    f.click("camera0");
    assert.equal(f.classes.has("touch-controlling"), true);
});
