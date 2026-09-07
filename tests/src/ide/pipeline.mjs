import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Toolchain } from "../../../static/ide/toolchain.js";
import { runProgram } from "../../../static/ide/runtime.js";
import { parseLine } from "../../../static/ide/protocol.js";
import { exampleArguments, checkExample } from "../../../static/ide/example.js";
import { options } from "./options.mjs";

const args = options({ example: { type: "string", default: "smoke" }, reference: { type: "string" } });
const read = path => readFileSync(join(args.bundle, path));
const specification = JSON.parse(read("examples.json"));
const example = specification.examples[args.example];
const started = performance.now();
const toolchain = await Toolchain.fromBytes(read("toolchain/llvm.wasm"), read("toolchain/sysroot.tar"), read("rl_tools_include.tar"));
const compiled = await toolchain.compile(new Map([[example.file, read("examples/" + args.example + ".cpp")]]),
    exampleArguments(specification, args.example), { onStderr: console.error, onJob: job => console.log("job: " + job[0]) });
assert.equal(compiled.exitCode, 0);
assert.ok(compiled.output, "compiler produced an output file");
console.log("compiled " + compiled.output.length + " bytes in " + ((performance.now() - started) / 1000).toFixed(1) + " s");

async function execute(program){
    const lines = [];
    const result = await runProgram(program, { args: example.args, onStdout: line => { console.log(line); lines.push(line); }, onStderr: console.error });
    const evaluations = checkExample(example, { ...result, lines });
    console.log("PASS: " + args.example + ", " + evaluations.length + " evaluations");
    return lines;
}
const actual = await execute(compiled.output);
if(args.reference){
    const expected = await execute(readFileSync(args.reference));
    const withoutTiming = lines => lines.filter(line => !["steps_per_second", "total_time"].includes(parseLine(line)?.type));
    assert.deepEqual(withoutTiming(actual), withoutTiming(expected), "browser and host-compiled programs agree");
    console.log("PASS: parity with " + args.reference);
}
