// Headless end-to-end test of the browser IDE pipeline: mounts the RLtools headers, compiles the example program with the
// WASI-hosted clang, runs the result, and checks the learning curve. Uses exactly the modules the page uses.
// Prerequisites: build static/ide/build via the tools/ide/toolchain superbuild + tools/ide/bundle.sh. Run with node >= 18 from anywhere.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Toolchain } from "../../../static/ide/toolchain.js";
import { runProgram } from "../../../static/ide/runtime.js";
import { untar } from "../../../static/ide/tar.js";
import { parseLine } from "../../../static/ide/protocol.js";

const ideDirectory = fileURLToPath(new URL("../../../static/ide/", import.meta.url));
const read = relativePath => readFileSync(ideDirectory + relativePath);
const seconds = started => ((performance.now() - started) / 1000).toFixed(1);
const arguments_ = ["-std=c++17", "-O2", "-fno-exceptions", "-Iinclude", "training.cpp", "-o", "training.wasm"];
const MINIMUM_FINAL_MEAN_RETURN = -400;

let started = performance.now();
const toolchain = await Toolchain.fromBytes(read("build/toolchain/llvm.wasm"), read("build/toolchain/sysroot.tar"));
console.log(`toolchain loaded in ${seconds(started)} s`);

const files = new Map();
for(const [path, data] of untar(read("build/rl_tools_include.tar"))){
    files.set("include/" + path, data);
}
files.set("training.cpp", read("build/examples/pendulum_sac.cpp"));
console.log(`${files.size - 1} headers mounted`);

started = performance.now();
const compiled = await toolchain.compile(files, arguments_, { onStderr: line => console.error(line), onJob: job => console.log(`  ${job[0]} ${job.length - 1} arguments`) });
console.log(`compile exit code ${compiled.exitCode} in ${seconds(started)} s`);
if(compiled.exitCode !== 0 || !compiled.output){
    process.exit(1);
}
console.log(`training.wasm ${(compiled.output.length / 1024).toFixed(0)} KB`);

started = performance.now();
const evaluations = [];
const run = await runProgram(compiled.output, {
    args: ["0"],
    onStdout: line => {
        console.log("  | " + line);
        const event = parseLine(line);
        if(event && event.type === "evaluation"){
            evaluations.push(event);
        }
    },
    onStderr: line => console.error("  ! " + line),
});
console.log(`run exit code ${run.exitCode} in ${seconds(started)} s, ${evaluations.length} evaluations`);

const last = evaluations[evaluations.length - 1];
const ok = run.exitCode === 0 && evaluations.length >= 2 && last.step === last.stepLimit && last.meanReturn > MINIMUM_FINAL_MEAN_RETURN;
console.log(ok ? `PASS: final mean return ${last.meanReturn.toFixed(1)} > ${MINIMUM_FINAL_MEAN_RETURN}` : `FAIL: ${JSON.stringify(last)}`);
process.exit(ok ? 0 : 1);
