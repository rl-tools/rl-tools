// Runs a WASI command module under node with the IDE's own runtime: node tools/ide/run.mjs <program.wasm> [args...]
import { readFileSync } from "node:fs";
import { runProgram } from "../../static/ide/runtime.js";

const [program, ...args] = process.argv.slice(2);
if(!program){
    console.error("usage: node tools/ide/run.mjs <program.wasm> [args...]");
    process.exit(2);
}
const started = performance.now();
const result = await runProgram(readFileSync(program), { args, onStdout: line => console.log(line), onStderr: line => console.error(line) });
console.error(`exit ${result.exitCode} after ${((performance.now() - started) / 1000).toFixed(1)} s`);
process.exit(result.exitCode);
