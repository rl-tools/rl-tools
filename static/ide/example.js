import { parseLine } from "./protocol.js";

export function exampleArguments(specification, name){
    const example = specification.examples[name];
    if(!example){
        throw new Error("unknown example: " + name);
    }
    return [...specification.flags, example.file, "-o", example.output];
}

export function checkExample(example, { exitCode, lines, files }){
    if(exitCode !== 0){
        throw new Error("program exited with code " + exitCode);
    }
    if(example.stdout && JSON.stringify(lines) !== JSON.stringify(example.stdout)){
        throw new Error("unexpected stdout: " + JSON.stringify(lines));
    }
    for(const [path, expected] of Object.entries(example.files ?? {})){
        if(!files.has(path) || new TextDecoder().decode(files.get(path)) !== expected){
            throw new Error("unexpected file contents: " + path);
        }
    }
    const evaluations = lines.map(parseLine).filter(event => event?.type === "evaluation");
    if(example.minimumFinalMeanReturn !== undefined){
        const last = evaluations[evaluations.length - 1];
        if(evaluations.length < 2 || last.step !== last.stepLimit || !(last.meanReturn > example.minimumFinalMeanReturn)){
            throw new Error("learning check failed: " + JSON.stringify(last));
        }
    }
    return evaluations;
}
