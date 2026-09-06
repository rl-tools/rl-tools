import { Directory } from "./wasi.js";
import { runProcess } from "./process.js";
import { treeFromEntries, mergeTrees, readFile } from "./filesystem.js";
import { untar } from "./tar.js";

const DRIVER_HEADER_PREFIXES = ["clang version", "Target:", "Thread model:", "InstalledDir:", "Build config:"];

// Splits one `clang -###` job line into argv, undoing the driver's quoting
function parseJobLine(line){
    const tokens = Array.from(line.matchAll(/ (?:([^ "]+)|"((?:[^"\\$]|\\["\\$])*)")/g), match => match[1] !== undefined ? match[1] : match[2].replaceAll(/\\(["$\\])/g, (_, character) => character));
    return tokens.filter((token, index) => !(index === 0 && token.length === 0));
}

function parseDriverPlan(text){
    const jobs = [];
    for(const line of text.split("\n")){
        if(line.startsWith(' "')){
            jobs.push(parseJobLine(line));
        }
    }
    return jobs;
}

function outputName(args){
    const index = args.indexOf("-o");
    return index >= 0 && index + 1 < args.length ? args[index + 1] : "a.out";
}

// clang and lld compiled to WASI cannot spawn processes, so the driver is asked for its plan (-###) and each job is run as its own WASI process
export class Toolchain{
    constructor(module, sysrootTree){
        this.module = module;
        this.sysrootTree = sysrootTree;
    }
    static async fromBytes(llvmBytes, sysrootTarBytes){
        const module = await WebAssembly.compile(llvmBytes);
        return new Toolchain(module, treeFromEntries(untar(sysrootTarBytes), { readonly: true }));
    }
    root(workTree){
        return mergeTrees(workTree, new Map([["usr", new Directory(this.sysrootTree, { readonly: true })], ["tmp", new Directory()]]));
    }
    async run(argv, root, { onStdout, onStderr, trace } = {}){
        return runProcess(this.module, ["llvm", ...argv], { root, onStdout, onStderr, trace });
    }
    // files: path -> string | Uint8Array placed at the filesystem root; args: clang++ arguments, e.g. ["-O2", "-Iinclude", "training.cpp", "-o", "training.wasm"]
    async compile(files, args, { onStderr = () => {}, onStdout = () => {}, onJob = () => {} } = {}){
        const workTree = treeFromEntries(files);
        const root = this.root(workTree);
        const planLines = [];
        const planExitCode = await this.run(["clang++", "-###", ...args], root, { onStdout: line => planLines.push(line), onStderr: line => planLines.push(line) });
        const planText = planLines.join("\n");
        if(planExitCode !== 0){
            for(const line of planLines){
                onStderr(line);
            }
            return { exitCode: planExitCode, output: null, jobs: [] };
        }
        for(const line of planLines){
            if(!DRIVER_HEADER_PREFIXES.some(prefix => line.startsWith(prefix)) && !line.startsWith(' "') && line.trim() !== "(in-process)" && line.length > 0){
                onStderr(line);
            }
        }
        const jobs = parseDriverPlan(planText);
        let exitCode = 0;
        for(const job of jobs){
            onJob(job);
            exitCode = await this.run(job, root, { onStdout, onStderr });
            if(exitCode !== 0){
                break;
            }
        }
        const output = exitCode === 0 ? readFile(root, outputName(args)) : null;
        return { exitCode, output, jobs, root };
    }
}
