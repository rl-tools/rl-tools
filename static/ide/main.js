import { parseLine } from "./protocol.js";
import { parseArguments, formatArguments } from "./arguments.js";
import { exampleArguments } from "./example.js";
import { fetchChecked } from "./assets.js";
import { WorkerClient } from "./worker_client.js";

const BUNDLE_URL = new URL("./build/", import.meta.url);
const elements = Object.fromEntries(["editor", "arguments", "seed", "compile", "run", "stop", "status", "terminal", "chart", "commit"].map(id => [id, document.getElementById(id)]));
const state = { phase: "loading", operation: 0, program: null, compiler: null, runner: null, points: [] };

function setStatus(text){
    elements.status.textContent = text;
}

function setPhase(phase){
    state.phase = phase;
    elements.compile.disabled = phase !== "idle";
    elements.run.disabled = phase !== "idle" || state.program === null;
    elements.stop.disabled = phase !== "compile" && phase !== "run";
}

function appendLine(line, kind = "stdout"){
    const node = document.createElement("div");
    node.className = "line " + kind;
    node.textContent = line;
    elements.terminal.appendChild(node);
    elements.terminal.scrollTop = elements.terminal.scrollHeight;
}

function clearTerminal(){
    elements.terminal.replaceChildren();
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

function showEvent(event, operation){
    if(operation !== state.operation){
        return;
    }
    if(event.type === "stdout" || event.type === "stderr"){
        appendLine(event.line, event.type);
        const parsed = event.type === "stdout" ? parseLine(event.line) : null;
        if(state.phase === "run" && parsed?.type === "evaluation"){
            state.points.push(parsed);
            drawChart(state.points);
            setStatus("running… step " + parsed.step + "/" + parsed.stepLimit + ", mean return " + parsed.meanReturn.toFixed(1));
        }
    }
    else if(event.type === "status"){
        setStatus(event.text);
    }
    else if(event.type === "job"){
        setStatus("compiling… " + event.tool);
    }
}

async function compiler(operation){
    if(!state.compiler || state.compiler.closed){
        state.compiler = new WorkerClient();
        await state.compiler.request("load", { bundleUrl: BUNDLE_URL.href }, { onEvent: event => showEvent(event, operation) });
    }
    return state.compiler;
}

async function compile(){
    if(state.phase !== "idle"){
        return;
    }
    const operation = ++state.operation;
    state.program = null;
    setPhase("compile");
    clearTerminal();
    setStatus("compiling…");
    try{
        const args = parseArguments(elements.arguments.value);
        const client = await compiler(operation);
        if(operation !== state.operation){
            return;
        }
        const result = await client.request("compile", { sources: { [state.sourceName]: elements.editor.value }, args }, { onEvent: event => showEvent(event, operation) });
        if(operation !== state.operation){
            return;
        }
        if(result.exitCode === 0 && result.output){
            state.program = result.output;
            setStatus("compiled in " + result.seconds.toFixed(1) + " s (" + (result.output.length / 1024).toFixed(0) + " KB)");
        }
        else{
            setStatus(result.exitCode === 0 ? "command completed without an output file" : "compile failed (exit code " + result.exitCode + ")");
        }
    }
    catch(error){
        if(operation === state.operation){
            setStatus(error.message);
            appendLine(error.message, "stderr");
        }
    }
    finally{
        if(operation === state.operation){
            setPhase("idle");
        }
    }
}

async function run(){
    if(state.phase !== "idle" || state.program === null){
        return;
    }
    const operation = ++state.operation;
    state.points = [];
    drawChart(state.points);
    clearTerminal();
    setPhase("run");
    setStatus("running…");
    const client = new WorkerClient();
    state.runner = client;
    try{
        const program = state.program.slice().buffer;
        const seed = elements.seed.value.trim();
        const result = await client.request("run", { program, args: seed.length > 0 ? [seed] : [] }, {
            timeout: 900000, transfer: [program], onEvent: event => showEvent(event, operation),
        });
        if(operation === state.operation){
            setStatus("finished with exit code " + result.exitCode + " after " + result.seconds.toFixed(1) + " s");
        }
    }
    catch(error){
        if(operation === state.operation){
            setStatus(error.message);
            appendLine(error.message, "stderr");
        }
    }
    finally{
        client.terminate();
        if(operation === state.operation){
            state.runner = null;
            setPhase("idle");
        }
    }
}

function stop(){
    ++state.operation;
    if(state.phase === "compile"){
        state.compiler?.terminate();
    }
    state.runner?.terminate();
    state.runner = null;
    setPhase("idle");
    setStatus("stopped");
    appendLine("stopped", "meta");
}

async function load(){
    drawChart([]);
    try{
        const [specification, manifest] = await Promise.all([
            fetchChecked(new URL("examples.json", BUNDLE_URL)).then(response => response.json()),
            fetchChecked(new URL("manifest.json", BUNDLE_URL)).then(response => response.json()),
        ]);
        const name = specification.default;
        const example = specification.examples[name];
        state.sourceName = example.file;
        elements.arguments.value = formatArguments(exampleArguments(specification, name));
        elements.seed.value = example.args[0] ?? "";
        elements.editor.value = await (await fetchChecked(new URL("examples/" + name + ".cpp", BUNDLE_URL))).text();
        elements.commit.textContent = "rl_tools @ " + manifest.rl_tools_commit.slice(0, 9);
        await compiler(state.operation);
        setStatus("ready");
        setPhase("idle");
    }
    catch(error){
        setStatus("failed to load: " + error.message);
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
