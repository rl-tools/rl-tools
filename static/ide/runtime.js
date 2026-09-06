import { runProcess } from "./process.js";
import { treeFromEntries, entriesFromTree } from "./filesystem.js";

// Runs a compiled program as a WASI process. The program sees its files at "/", reads args, writes stdout/stderr lines and files.
export async function runProgram(program, { args = [], files = new Map(), onStdout = () => {}, onStderr = () => {} } = {}){
    const module = program instanceof WebAssembly.Module ? program : await WebAssembly.compile(program);
    const root = treeFromEntries(files);
    const exitCode = await runProcess(module, ["program", ...args], { root, onStdout, onStderr });
    return { exitCode, files: entriesFromTree(root) };
}
