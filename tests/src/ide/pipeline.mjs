// Headless end-to-end test of the browser IDE pipeline: mounts the RLtools headers, compiles the example program with the
// WASI-hosted clang, runs the result, and checks the learning curve. Uses exactly the modules the page uses.
// Prerequisites: static/ide/build from the tools/ide/toolchain superbuild + tools/ide/bundle.sh. Run with node >= 18.
// usage: node tests/src/ide/pipeline.mjs [--reference <training.wasm>]
//   --reference: the same program cross-compiled by the host clang against the same sysroot (src/rl/environments/pendulum/
//   sac/wasi/build.sh); its output must match the in-browser build's line for line, timing lines aside
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
const referenceIndex = process.argv.indexOf("--reference");
const referencePath = referenceIndex >= 0 ? process.argv[referenceIndex + 1] : null;
const isTimingLine = line => ["steps_per_second", "total_time"].includes(parseLine(line)?.type);

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

async function execute(program, label){
    const started = performance.now();
    const lines = [];
    const run = await runProgram(program, {
        args: ["0"],
        onStdout: line => {
            console.log("  | " + line);
            lines.push(line);
        },
        onStderr: line => console.error("  ! " + line),
    });
    const evaluations = lines.map(parseLine).filter(event => event && event.type === "evaluation");
    console.log(`${label} exit code ${run.exitCode} in ${seconds(started)} s, ${evaluations.length} evaluations`);
    return { exitCode: run.exitCode, lines, evaluations };
}

const run = await execute(compiled.output, "run");
const last = run.evaluations[run.evaluations.length - 1];
let ok = run.exitCode === 0 && run.evaluations.length >= 2 && last.step === last.stepLimit && last.meanReturn > MINIMUM_FINAL_MEAN_RETURN;
console.log(ok ? `PASS: final mean return ${last.meanReturn.toFixed(1)} > ${MINIMUM_FINAL_MEAN_RETURN}` : `FAIL: ${JSON.stringify(last)}`);

if(referencePath !== null){
    const reference = await execute(readFileSync(referencePath), "reference run");
    const expected = reference.lines.filter(line => !isTimingLine(line));
    const actual = run.lines.filter(line => !isTimingLine(line));
    const firstDifference = expected.findIndex((line, index) => line !== actual[index]);
    const parity = reference.exitCode === 0 && expected.length === actual.length && firstDifference < 0;
    if(parity){
        console.log(`PASS: ${actual.length} lines identical to the host-compiled reference ${referencePath}`);
    }
    else{
        console.log(`FAIL: output differs from the reference at line ${firstDifference}: expected ${JSON.stringify(expected[firstDifference])}, got ${JSON.stringify(actual[firstDifference])}`);
    }
    ok = ok && parity;
}
process.exit(ok ? 0 : 1);
