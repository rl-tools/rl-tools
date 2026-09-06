import { WASI } from "./wasi.js";

const decoder = new TextDecoder("utf-8", { fatal: false });

function lineSink(write){
    let pending = "";
    const sink = bytes => {
        pending += decoder.decode(bytes, { stream: true });
        const lines = pending.split("\n");
        pending = lines.pop();
        for(const line of lines){
            write(line);
        }
    };
    const flush = () => {
        if(pending.length > 0){
            write(pending);
            pending = "";
        }
    };
    return { sink, flush };
}

// Runs one WASI command module to completion: argv in, stdout/stderr lines out, a single directory tree preopened at "/"
export async function runProcess(module, argv, { root, env = [], onStdout = () => {}, onStderr = () => {}, trace = null }){
    const stdout = lineSink(onStdout);
    const stderr = lineSink(onStderr);
    const wasi = new WASI({ args: argv, env, root, stdout: stdout.sink, stderr: stderr.sink, trace });
    const instance = await WebAssembly.instantiate(module, wasi.importObject);
    let exitCode;
    try{
        exitCode = wasi.start(instance);
    }
    catch(error){
        onStderr(`${argv[0]}: ${error.message ?? error}`);
        exitCode = 128;
    }
    stdout.flush();
    stderr.flush();
    return exitCode;
}
