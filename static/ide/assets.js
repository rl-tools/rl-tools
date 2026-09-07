import { Toolchain } from "./toolchain.js";

export async function fetchChecked(url){
    const response = await fetch(url);
    if(!response.ok){
        throw new Error(url + ": HTTP " + response.status);
    }
    return response;
}

export async function loadToolchain(bundleUrl){
    const at = path => new URL(path, bundleUrl);
    const bytes = async path => new Uint8Array(await (await fetchChecked(at(path))).arrayBuffer());
    const [module, sysroot, includes, manifest] = await Promise.all([
        fetchChecked(at("toolchain/llvm.wasm")).then(response => WebAssembly.compileStreaming(response)),
        bytes("toolchain/sysroot.tar"),
        bytes("rl_tools_include.tar"),
        fetchChecked(at("toolchain/toolchain.json")).then(response => response.json()),
    ]);
    return { toolchain: await Toolchain.fromBytes(module, sysroot, includes), manifest };
}
