import { Directory } from "./wasi.js";
import { runProcess } from "./process.js";
import { treeFromEntries, readFile } from "./filesystem.js";
import { parseArguments } from "./arguments.js";
import { untar } from "./tar.js";

const DRIVER_HEADER_PREFIXES = ["clang version", "Target:", "Thread model:", "InstalledDir:", "Build config:"];

export function parseDriverPlan(text){
    const jobs = [];
    for(const line of text.split("\n")){
        if(line.startsWith(' "')){
            const job = parseArguments(line);
            jobs.push(job[0] === "" ? job.slice(1) : job);
        }
    }
    return jobs;
}

function outputName(job){
    const index = job.lastIndexOf("-o");
    return index >= 0 ? job[index + 1] : null;
}

// clang and lld compiled to WASI cannot spawn processes, so the driver is asked for its plan (-###) and each job is run as its own WASI process
export class Toolchain{
    constructor(module, sysrootTree, includeTree = null){
        this.module = module;
        this.sysrootTree = sysrootTree;
        this.includeTree = includeTree;
    }
    static async fromBytes(llvmBytes, sysrootTarBytes, includeTarBytes = null){
        const module = llvmBytes instanceof WebAssembly.Module ? llvmBytes : await WebAssembly.compile(llvmBytes);
        return new Toolchain(module, treeFromEntries(untar(sysrootTarBytes), { readonly: true }),
            includeTarBytes === null ? null : treeFromEntries(untar(includeTarBytes), { readonly: true }));
    }
    root(workTree){
        const root = new Map(workTree);
        const mounts = new Map([["usr", new Directory(this.sysrootTree, { readonly: true })], ["tmp", new Directory()]]);
        if(this.includeTree !== null){
            mounts.set("include", new Directory(this.includeTree, { readonly: true }));
        }
        for(const [path, directory] of mounts){
            if(root.has(path)){
                throw new Error(`${path} is a reserved mount`);
            }
            root.set(path, directory);
        }
        return root;
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
        const outputPath = jobs.length > 0 ? outputName(jobs[jobs.length - 1]) : null;
        const output = exitCode === 0 && outputPath ? readFile(root, outputPath) : null;
        return { exitCode, output, jobs, root };
    }
}
