import { loadToolchain } from "./assets.js";
import { runProgram } from "./runtime.js";

let toolchain = null;
let busy = false;

self.onmessage = async ({ data }) => {
    const { id, type } = data;
    const emit = event => self.postMessage({ id, type: "event", event });
    if(busy){
        self.postMessage({ id, type: "error", message: "worker is busy" });
        return;
    }
    busy = true;
    try{
        const started = performance.now();
        let result;
        let transfer = [];
        const streams = {
            onStdout: line => emit({ type: "stdout", line }),
            onStderr: line => emit({ type: "stderr", line }),
        };
        if(type === "load"){
            emit({ type: "status", text: "loading compiler and headers…" });
            const loaded = await loadToolchain(data.bundleUrl);
            toolchain = loaded.toolchain;
            result = { manifest: loaded.manifest };
        }
        else if(type === "compile"){
            if(toolchain === null){
                throw new Error("load the toolchain before compiling");
            }
            const compiled = await toolchain.compile(new Map(Object.entries(data.sources)), data.args, {
                ...streams,
                onJob: job => emit({ type: "job", tool: job[0] }),
            });
            const output = compiled.output?.slice() ?? null;
            result = { exitCode: compiled.exitCode, output };
            transfer = output ? [output.buffer] : [];
        }
        else if(type === "run"){
            result = await runProgram(new Uint8Array(data.program), { args: data.args, ...streams });
            transfer = [...new Set([...result.files.values()].map(bytes => bytes.buffer))];
        }
        else{
            throw new Error("unknown worker command: " + type);
        }
        result.seconds = (performance.now() - started) / 1000;
        self.postMessage({ id, type: "result", result }, transfer);
    }
    catch(error){
        self.postMessage({ id, type: "error", message: error.message ?? String(error) });
    }
    finally{
        busy = false;
    }
};
