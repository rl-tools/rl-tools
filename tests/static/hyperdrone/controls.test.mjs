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

function fixture(defer_mouse_lock = false){
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
    window.location = {search: ""};
    window.isSecureContext = true;
    document.createElement = () => ({});
    const elements = new Map();
    const pending_locks = [];
    for(const match of page.matchAll(/<[^>]*\bid="([^"]+)"[^>]*>/g)){
        const element = new EventTarget();
        element.id = match[1];
        element.value = element.defaultValue = match[0].match(/\bvalue="([^"]+)"/)?.[1] || "";
        element.disabled = /\bdisabled\b/.test(match[0]);
        element.hidden = /\bhidden\b/.test(match[0]);
        element.open = false;
        element.attributes = new Map();
        element.setAttribute = (name, value) => element.attributes.set(name, value);
        element.removeAttribute = name => element.attributes.delete(name);
        element.appendChild = option => {
            if(option.selected){
                element.value = option.value;
            }
        };
        element.showModal = () => { element.open = true; };
        element.close = () => { element.open = false; send(element, "close"); };
        element.getContext = () => ({});
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
    const scenes = [106, 107].map(id => ({name: `ProcTHOR-Train-${id}`, reference: `scene-${id}`}));
    const module = {
        _demo_api_version: () => 1,
        FS: {writeFile(){}},
        ccall(name){
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
        Conta: class {async load(){ return new Uint8Array(); }},
        console: {...console, error: error => errors.push(error)},
        import_module: async () => ({default: async options => {
            module_options = options;
            return module;
        }}),
        fetch: async path => ({ok: true, json: async () => path === "registry.json"
            ? {drones: [{name: "ARPL", reference: "drone", dynamics_index: 3}], policy: {reference: "policy"}}
            : {scene: scenes[0].name, reference: scenes[0].reference, position: [0, 0, 1], yaw_degrees: 0}}),
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
        document, window, classes, element, click, poll, axes, grant_mouse_lock, errors,
        start: context.controls.start,
        load(reason){ abort_reason = reason; return context.controls.main(); },
        abort: reason => module_options.onAbort(reason),
    };
}

function touch_flight(f){
    f.start();
    f.click("camera0");
    send(f.element("left-stick"), "pointerdown", {pointerId: 1, clientX: 110});
    send(f.element("right-stick"), "pointerdown", {pointerId: 2, clientY: 10});
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
    const f = fixture(true);
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
    const f = fixture(true);
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
