// Checks a built toolchain package against its manifest: module shape (imports, exports, memory ceiling), compiler identity
// (pinned commit, version suffix, targets, resource dir) and the template-recursion depth the compiler survives on the host
// engine's stack. Pass --bisect-stack to measure the maximum depth separately.
// usage: node tests/src/ide/toolchain.mjs [--bundle <dir>] [--bisect-stack]
import { readFileSync } from "node:fs";
import { options } from "./options.mjs";
import { Toolchain } from "../../../static/ide/toolchain.js";
import { runProgram } from "../../../static/ide/runtime.js";
import { SYSCALLS } from "../../../static/ide/wasi.js";

const MEMORY_PAGES_MAXIMUM = 65536;
const STACK_PROBE_MINIMUM_DEPTH = 256;
const STACK_PROBE_MAXIMUM_DEPTH = 4096;

const args = options({ "bisect-stack": { type: "boolean", default: false } });
const toolchainDirectory = args.bundle + "/toolchain";
const read = name => readFileSync(toolchainDirectory + "/" + name);
const seconds = started => ((performance.now() - started) / 1000).toFixed(1);
let failures = 0;
function check(condition, description){
    console.log(`${condition ? "ok  " : "FAIL"} ${description}`);
    if(!condition){
        failures++;
    }
}

function readUnsigned(bytes, offset){
    let value = 0;
    let shift = 0;
    let position = offset;
    for(;;){
        const byte = bytes[position++];
        value |= (byte & 0x7f) << shift;
        if((byte & 0x80) === 0){
            return [value >>> 0, position];
        }
        shift += 7;
    }
}

// Memory limits are not exposed by the WebAssembly API, so the memory section (id 5) is read from the binary
function memoryLimits(bytes){
    let position = 8;
    while(position < bytes.length){
        const id = bytes[position++];
        const [size, start] = readUnsigned(bytes, position);
        if(id === 5){
            const [count, cursor] = readUnsigned(bytes, start);
            if(count === 0){
                return null;
            }
            const flags = bytes[cursor];
            const [minimum, afterMinimum] = readUnsigned(bytes, cursor + 1);
            const maximum = (flags & 1) !== 0 ? readUnsigned(bytes, afterMinimum)[0] : null;
            return { count, minimum, maximum, shared: (flags & 2) !== 0 };
        }
        position = start + size;
    }
    return null;
}

const manifest = JSON.parse(read("toolchain.json").toString());
const moduleBytes = read("llvm.wasm");
check(manifest.llvm_commit !== undefined && manifest.llvm_version !== undefined, `manifest names the LLVM build (${manifest.llvm_version} @ ${manifest.llvm_commit})`);

const module = await WebAssembly.compile(moduleBytes);
const imports = WebAssembly.Module.imports(module);
const foreignModules = imports.filter(entry => entry.module !== "wasi_snapshot_preview1").map(entry => `${entry.module}.${entry.name}`);
const unknown = imports.filter(entry => entry.module === "wasi_snapshot_preview1" && !SYSCALLS.includes(entry.name)).map(entry => entry.name);
check(foreignModules.length === 0, `all ${imports.length} imports come from wasi_snapshot_preview1${foreignModules.length ? ": " + foreignModules.join(" ") : ""}`);
check(unknown.length === 0, `every import is implemented by static/ide/wasi.js${unknown.length ? ", missing: " + unknown.join(" ") : ""}`);
const exportNames = WebAssembly.Module.exports(module).map(entry => entry.name).sort();
check(exportNames.join(",") === "_start,memory", `exports are _start and memory (${exportNames.join(" ")})`);
const limits = memoryLimits(new Uint8Array(moduleBytes));
check(limits !== null && limits.count === 1 && !limits.shared, `one unshared memory (${JSON.stringify(limits)})`);
check(limits !== null && limits.maximum === MEMORY_PAGES_MAXIMUM, `memory maximum is ${MEMORY_PAGES_MAXIMUM} pages (${limits?.maximum})`);

const toolchain = await Toolchain.fromBytes(module, read("sysroot.tar"));
async function query(...args){
    const lines = [];
    const exitCode = await toolchain.run(["clang++", ...args], toolchain.root(new Map()), { onStdout: line => lines.push(line), onStderr: line => lines.push(line) });
    return { exitCode, lines };
}
const version = await query("--version");
const versionMatch = /^clang version (\S+) \((\S+) ([0-9a-f]{40})\)/.exec(version.lines[0] ?? "");
check(version.exitCode === 0 && versionMatch !== null, `--version is parseable: ${version.lines[0]}`);
if(versionMatch){
    check(versionMatch[1] === manifest.llvm_version, `version ${versionMatch[1]} matches the manifest (${manifest.llvm_version})`);
    check(versionMatch[3] === manifest.llvm_commit, `commit ${versionMatch[3]} matches the manifest`);
    check(versionMatch[2] === manifest.llvm_repository.replace(/\.git$/, ""), `repository is the pinned one (${versionMatch[2]})`);
}
check(version.lines.some(line => line === `Target: ${manifest.target.replace("wasm32-", "wasm32-unknown-")}`), `default target is ${manifest.target}`);
const targets = await query("-print-targets");
const registered = targets.lines.map(line => /^\s+(\S+)\s+-/.exec(line)?.[1]).filter(Boolean).sort();
check(registered.join(",") === "wasm32,wasm64", `registered targets are wasm32 and wasm64 (${registered.join(" ")})`);
const resourceDirectory = await query("-print-resource-dir");
check(resourceDirectory.lines[0] === "/usr", `resource dir is /usr (${resourceDirectory.lines[0]})`);

function probeSource(depth){
    return `template<int N> struct Depth{ static constexpr int value = Depth<N - 1>::value + 1; };
template<> struct Depth<0>{ static constexpr int value = 0; };
#include <cstdio>
int main(){ std::printf("depth %d\\n", Depth<${depth}>::value); return Depth<${depth}>::value == ${depth} ? 0 : 1; }
`;
}
async function compiles(depth){
    const compiled = await toolchain.compile(new Map([["probe.cpp", probeSource(depth)]]), ["-std=c++17", "-O2", "-fno-exceptions", "probe.cpp", "-o", "probe.wasm"]);
    return compiled.exitCode === 0 && compiled.output !== null ? compiled.output : null;
}
let started = performance.now();
const minimumProgram = await compiles(STACK_PROBE_MINIMUM_DEPTH);
check(minimumProgram !== null, `template recursion depth ${STACK_PROBE_MINIMUM_DEPTH} compiles and links (${seconds(started)} s)`);
if(minimumProgram !== null){
    started = performance.now();
    const lines = [];
    const run = await runProgram(minimumProgram, { onStdout: line => lines.push(line), onStderr: line => lines.push(line) });
    check(run.exitCode === 0 && lines[0] === `depth ${STACK_PROBE_MINIMUM_DEPTH}`, `probe runs (exit ${run.exitCode}, "${lines[0]}", ${seconds(started)} s)`);
    if(args["bisect-stack"]){
        started = performance.now();
        let low = STACK_PROBE_MINIMUM_DEPTH;
        let high = STACK_PROBE_MAXIMUM_DEPTH;
        while(high - low > 16){
            const middle = Math.floor((low + high) / 2);
            if(await compiles(middle)){
                low = middle;
            }
            else{
                high = middle;
            }
        }
        console.log(`info deepest template recursion that compiles here: ${low} (bisected in ${seconds(started)} s)`);
    }
}

console.log(failures === 0 ? "PASS" : `FAIL: ${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
