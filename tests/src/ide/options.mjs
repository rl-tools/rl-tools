import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

export function options(extra = {}){
    if(Number(process.versions.node.split(".")[0]) < 22){
        throw new Error("Node.js 22+ is required");
    }
    const { values } = parseArgs({ options: { bundle: { type: "string", default: fileURLToPath(new URL("../../../static/ide/build/", import.meta.url)) }, ...extra } });
    values.bundle = resolve(values.bundle);
    return values;
}
