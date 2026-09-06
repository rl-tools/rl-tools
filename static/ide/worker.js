import { Toolchain } from "./toolchain.js";
import { runProgram } from "./runtime.js";
import { treeFromEntries } from "./filesystem.js";
import { untar } from "./tar.js";

let toolchain = null;
let includeEntries = new Map();

// The archives arrive as single transferred buffers and are unpacked here: posting a map of archive slices would clone the archive once per entry
self.onmessage = async ({ data }) => {
    if(data.type === "toolchain"){
        toolchain = new Toolchain(data.module, treeFromEntries(untar(new Uint8Array(data.sysrootTar)), { readonly: true }));
        includeEntries = new Map();
        for(const [path, bytes] of untar(new Uint8Array(data.includeTar))){
            includeEntries.set("include/" + path, bytes);
        }
        self.postMessage({ type: "ready", headers: includeEntries.size });
    }
    else if(data.type === "compile"){
        const started = performance.now();
        const files = new Map(includeEntries);
        for(const [name, text] of Object.entries(data.sources)){
            files.set(name, text);
        }
        const result = await toolchain.compile(files, data.args, {
            onStdout: line => self.postMessage({ type: "stdout", line }),
            onStderr: line => self.postMessage({ type: "stderr", line }),
            onJob: job => self.postMessage({ type: "job", tool: job[0] }),
        });
        self.postMessage({ type: "compiled", exitCode: result.exitCode, output: result.output, seconds: (performance.now() - started) / 1000 });
    }
    else if(data.type === "run"){
        const started = performance.now();
        const result = await runProgram(new Uint8Array(data.program), {
            args: data.args,
            onStdout: line => self.postMessage({ type: "stdout", line }),
            onStderr: line => self.postMessage({ type: "stderr", line }),
        });
        self.postMessage({ type: "exited", exitCode: result.exitCode, files: result.files, seconds: (performance.now() - started) / 1000 });
    }
};
