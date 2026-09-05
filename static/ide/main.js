import { parseLine } from "./protocol.js";

const TOOLCHAIN_URL = "./build/toolchain/llvm.wasm";
const SYSROOT_URL = "./build/toolchain/sysroot.tar";
const INCLUDE_URL = "./build/rl_tools_include.tar";
const EXAMPLE_URL = "./build/examples/pendulum_sac.cpp";
const MANIFEST_URL = "./build/manifest.json";
const DEFAULT_ARGUMENTS = "-std=c++17 -O2 -fno-exceptions -Iinclude training.cpp -o training.wasm";
const COMPILE_TIMEOUT_MILLISECONDS = 300000;

const elements = Object.fromEntries(["editor", "arguments", "seed", "compile", "run", "stop", "status", "terminal", "chart", "commit"].map(id => [id, document.getElementById(id)]));

const state = { module: null, sysrootTar: null, includeTar: null, program: null, compileWorker: null, runWorker: null, points: [] };

function setStatus(text){
    elements.status.textContent = text;
}

function appendLine(line, kind = "stdout"){
    const node = document.createElement("div");
    node.className = `line ${kind}`;
    node.textContent = line;
    elements.terminal.appendChild(node);
    elements.terminal.scrollTop = elements.terminal.scrollHeight;
}

function clearTerminal(){
    elements.terminal.replaceChildren();
}

async function fetchBytes(url, onProgress){
    const response = await fetch(url);
    if(!response.ok){
        throw new Error(`${url}: HTTP ${response.status}`);
    }
    const total = Number(response.headers.get("content-length")) || 0;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    for(;;){
        const { value, done } = await reader.read();
        if(done){
            break;
        }
        chunks.push(value);
        received += value.length;
        if(onProgress){
            onProgress(received, total);
        }
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    for(const chunk of chunks){
        bytes.set(chunk, offset);
        offset += chunk.length;
    }
    return bytes;
}

function megabytes(bytes){
    return (bytes / 1e6).toFixed(0);
}

function drawChart(points){
    const canvas = elements.chart;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);
    const style = getComputedStyle(document.documentElement);
    const ink = style.getPropertyValue("--ink").trim();
    const muted = style.getPropertyValue("--muted").trim();
    const rule = style.getPropertyValue("--rule").trim();
    const accent = style.getPropertyValue("--accent").trim();
    const margin = { left: 56, right: 16, top: 16, bottom: 28 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const stepLimit = points.length > 0 ? points[points.length - 1].stepLimit : 1;
    const returns = points.map(point => point.meanReturn);
    const minReturn = points.length > 0 ? Math.min(...returns) : -1;
    const maxReturn = points.length > 0 ? Math.max(...returns, 0) : 0;
    const span = maxReturn - minReturn || 1;
    const x = step => margin.left + (step / stepLimit) * plotWidth;
    const y = value => margin.top + (1 - (value - minReturn) / span) * plotHeight;
    context.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.fillStyle = muted;
    context.strokeStyle = rule;
    context.lineWidth = 1;
    const ticks = 4;
    for(let i = 0; i <= ticks; i++){
        const value = minReturn + (span * i) / ticks;
        const py = y(value);
        context.beginPath();
        context.moveTo(margin.left, py);
        context.lineTo(width - margin.right, py);
        context.stroke();
        context.textAlign = "right";
        context.textBaseline = "middle";
        context.fillText(value.toFixed(0), margin.left - 6, py);
    }
    context.textAlign = "center";
    context.textBaseline = "top";
    for(let i = 0; i <= 4; i++){
        const step = (stepLimit * i) / 4;
        context.fillText(step.toFixed(0), x(step), height - margin.bottom + 6);
    }
    context.textAlign = "left";
    context.textBaseline = "top";
    context.fillText("mean evaluation return over training steps", margin.left, 2);
    if(points.length === 0){
        return;
    }
    context.strokeStyle = accent;
    context.lineWidth = 2;
    context.beginPath();
    points.forEach((point, index) => {
        const px = x(point.step);
        const py = y(point.meanReturn);
        if(index === 0){
            context.moveTo(px, py);
        }
        else{
            context.lineTo(px, py);
        }
    });
    context.stroke();
    const last = points[points.length - 1];
    context.fillStyle = accent;
    context.beginPath();
    context.arc(x(last.step), y(last.meanReturn), 3.5, 0, 2 * Math.PI);
    context.fill();
    context.fillStyle = ink;
    context.textAlign = "right";
    context.textBaseline = "bottom";
    context.fillText(last.meanReturn.toFixed(1), width - margin.right, y(last.meanReturn) - 6);
}

function createWorker(){
    return new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
}

function terminateRun(){
    if(state.runWorker){
        state.runWorker.terminate();
        state.runWorker = null;
    }
}

function terminateCompile(){
    if(state.compileWorker){
        state.compileWorker.terminate();
        state.compileWorker = null;
    }
}

// The compile worker keeps the compiler module, sysroot and headers resident; the archives are copied so a replacement worker can be seeded after a hang
function compileWorker(){
    if(state.compileWorker){
        return Promise.resolve(state.compileWorker);
    }
    const worker = createWorker();
    return new Promise(resolve => {
        worker.addEventListener("message", ({ data }) => {
            if(data.type === "ready"){
                state.compileWorker = worker;
                resolve(worker);
            }
        }, { once: true });
        const sysrootTar = state.sysrootTar.slice().buffer;
        const includeTar = state.includeTar.slice().buffer;
        worker.postMessage({ type: "toolchain", module: state.module, sysrootTar, includeTar }, [sysrootTar, includeTar]);
    });
}

async function compile(){
    elements.compile.disabled = true;
    elements.run.disabled = true;
    state.program = null;
    clearTerminal();
    setStatus("compiling…");
    const worker = await compileWorker();
    const args = elements.arguments.value.trim().split(/\s+/).filter(token => token.length > 0);
    const started = performance.now();
    const timeout = setTimeout(() => {
        appendLine(`compiler did not finish within ${COMPILE_TIMEOUT_MILLISECONDS / 1000} s, terminated`, "stderr");
        terminateCompile();
        setStatus("compile timed out");
        elements.compile.disabled = false;
    }, COMPILE_TIMEOUT_MILLISECONDS);
    worker.onmessage = ({ data }) => {
        if(data.type === "stdout" || data.type === "stderr"){
            appendLine(data.line, data.type);
        }
        else if(data.type === "job"){
            setStatus(`compiling… ${data.tool} (${((performance.now() - started) / 1000).toFixed(1)} s)`);
        }
        else if(data.type === "compiled"){
            clearTimeout(timeout);
            elements.compile.disabled = false;
            if(data.exitCode === 0 && data.output){
                state.program = data.output;
                elements.run.disabled = false;
                setStatus(`compiled in ${data.seconds.toFixed(1)} s (${(data.output.length / 1024).toFixed(0)} KB)`);
                appendLine(`compiled in ${data.seconds.toFixed(1)} s, ${(data.output.length / 1024).toFixed(0)} KB`, "meta");
            }
            else{
                setStatus(`compile failed (exit code ${data.exitCode})`);
            }
        }
    };
    worker.postMessage({ type: "compile", sources: { "training.cpp": elements.editor.value }, args });
}

function run(){
    terminateRun();
    state.points = [];
    drawChart(state.points);
    clearTerminal();
    elements.run.disabled = true;
    elements.stop.disabled = false;
    setStatus("running…");
    const worker = createWorker();
    state.runWorker = worker;
    const started = performance.now();
    worker.onmessage = ({ data }) => {
        if(data.type === "stdout" || data.type === "stderr"){
            appendLine(data.line, data.type);
            const event = parseLine(data.line);
            if(event && event.type === "evaluation"){
                state.points.push(event);
                drawChart(state.points);
                setStatus(`running… step ${event.step}/${event.stepLimit}, mean return ${event.meanReturn.toFixed(1)} (${((performance.now() - started) / 1000).toFixed(0)} s)`);
            }
        }
        else if(data.type === "exited"){
            setStatus(`finished with exit code ${data.exitCode} after ${data.seconds.toFixed(1)} s`);
            appendLine(`exit code ${data.exitCode} after ${data.seconds.toFixed(1)} s`, "meta");
            terminateRun();
            elements.run.disabled = false;
            elements.stop.disabled = true;
        }
    };
    const seed = elements.seed.value.trim();
    const program = state.program.slice().buffer;
    worker.postMessage({ type: "run", program, args: seed.length > 0 ? [seed] : [] }, [program]);
}

function stop(){
    terminateRun();
    setStatus("stopped");
    appendLine("stopped", "meta");
    elements.run.disabled = state.program === null;
    elements.stop.disabled = true;
}

async function load(){
    elements.arguments.value = DEFAULT_ARGUMENTS;
    drawChart([]);
    try{
        const [manifest, example] = await Promise.all([fetch(MANIFEST_URL).then(response => response.ok ? response.json() : null), fetch(EXAMPLE_URL).then(response => response.text())]);
        elements.editor.value = example;
        if(manifest){
            elements.commit.textContent = `rl_tools @ ${String(manifest.rl_tools_commit).slice(0, 9)}`;
        }
        setStatus("loading RLtools headers…");
        state.includeTar = await fetchBytes(INCLUDE_URL);
        setStatus("loading sysroot…");
        state.sysrootTar = await fetchBytes(SYSROOT_URL);
        const toolchainBytes = await fetchBytes(TOOLCHAIN_URL, (received, total) => setStatus(`loading compiler… ${megabytes(received)}${total ? " / " + megabytes(total) : ""} MB`));
        setStatus("preparing compiler…");
        state.module = await WebAssembly.compile(toolchainBytes);
        await compileWorker();
        setStatus("ready: clang 21.1.4 (wasm32-wasip1)");
        elements.compile.disabled = false;
    }
    catch(error){
        setStatus(`failed to load: ${error.message}. Build the toolchain (tools/ide/toolchain) and run tools/ide/bundle.sh, then serve the repository root.`);
    }
}

elements.compile.addEventListener("click", compile);
elements.run.addEventListener("click", run);
elements.stop.addEventListener("click", stop);
elements.editor.addEventListener("keydown", event => {
    if(event.key === "Tab"){
        event.preventDefault();
        const { selectionStart, selectionEnd, value } = elements.editor;
        elements.editor.value = value.slice(0, selectionStart) + "    " + value.slice(selectionEnd);
        elements.editor.selectionStart = elements.editor.selectionEnd = selectionStart + 4;
    }
    if((event.ctrlKey || event.metaKey) && event.key === "Enter" && !elements.compile.disabled){
        compile();
    }
});
window.addEventListener("resize", () => drawChart(state.points));
load();
