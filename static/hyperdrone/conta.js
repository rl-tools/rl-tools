const ENTRY_POINTS = [
    "https://conta.rl.tools/version/1/meta.json",
    "https://raw.githubusercontent.com/rl-tools/conta-index/main/version/1/meta.json",
    "https://huggingface.co/datasets/rl-tools/conta-index/resolve/main/version/1/meta.json",
];

async function verify(bytes, algorithm, expected){
    const digest = await crypto.subtle.digest(algorithm, bytes);
    const actual = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
    if(actual !== expected){
        throw new Error(`${algorithm} mismatch: expected ${expected}, received ${actual}`);
    }
}

async function response(url){
    const result = await fetch(url);
    if(!result.ok){
        throw new Error(`HTTP ${result.status}`);
    }
    return result;
}

export class Conta {
    constructor({overlay = new URL("./blob/", import.meta.url), entry_points = ENTRY_POINTS} = {}){
        this.overlay = overlay;
        this.entry_points = entry_points;
        this.indexes = new Map();
    }

    async index(entry_point){
        if(!this.indexes.has(entry_point)){
            this.indexes.set(entry_point, (async () => {
                const meta = await (await response(entry_point)).json();
                if(meta.format !== 1 || !/^[0-9a-f]{64}$/.test(meta.index?.sha256)){
                    throw new Error("Unsupported conta metadata");
                }
                const url = new URL(meta.index.location, entry_point);
                const bytes = await (await response(url)).arrayBuffer();
                await verify(bytes, "SHA-256", meta.index.sha256);
                const index = JSON.parse(new TextDecoder().decode(bytes));
                if(!index.shards || !Array.isArray(index.entries)){
                    throw new Error("Invalid conta index");
                }
                return index;
            })());
        }
        return this.indexes.get(entry_point);
    }

    async load(reference, {sha256, on_progress = () => {}} = {}){
        const sha1 = reference.replace(/^conta:/, "");
        if(!/^[0-9a-f]{40}$/.test(sha1)){
            throw new Error(`Invalid conta reference: ${reference}`);
        }
        const failures = [];
        const attempted = new Set();
        const download = async (url, compressed, expected_sha256) => {
            const key = String(url);
            if(attempted.has(key)){
                return null;
            }
            attempted.add(key);
            try{
                on_progress({phase: "download", received: 0, total: 0});
                const result = await response(url);
                const total = Number(result.headers.get("Content-Length"));
                let received = 0;
                let stream = result.body.pipeThrough(new TransformStream({
                    transform(chunk, controller){
                        received += chunk.byteLength;
                        on_progress({phase: "download", received, total});
                        controller.enqueue(chunk);
                    },
                }));
                if(compressed){
                    stream = stream.pipeThrough(new DecompressionStream("gzip"));
                }
                const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
                on_progress({phase: "verify", received, total});
                await verify(bytes, "SHA-1", sha1);
                if(expected_sha256){
                    await verify(bytes, "SHA-256", expected_sha256);
                }
                return bytes;
            }
            catch(error){
                failures.push(`${key}: ${error.message}`);
                return null;
            }
        };
        const local = await download(new URL(`${sha1}.gz`, this.overlay), true, sha256);
        if(local !== null){
            return local;
        }
        for(const entry_point of this.entry_points){
            try{
                on_progress({phase: "resolve"});
                const index = await this.index(entry_point);
                const entry = index.entries.find(entry => entry.sha1 === sha1);
                if(!entry){
                    throw new Error(`Index does not list ${sha1}`);
                }
                if(!/^[0-9a-f]{64}$/.test(entry.sha256) || (sha256 && sha256 !== entry.sha256)){
                    throw new Error(`Index SHA-256 disagrees with the pinned asset ${sha1}`);
                }
                for(const shard of entry.shards){
                    const template = index.shards[shard.id]?.url;
                    if(!template){
                        continue;
                    }
                    const url = new URL(template.replaceAll("{sha1}", sha1).replaceAll("{sha256}", entry.sha256));
                    if(shard.compressed?.includes("gz")){
                        const compressed_url = new URL(url);
                        compressed_url.pathname += ".gz";
                        const bytes = await download(compressed_url, true, entry.sha256);
                        if(bytes !== null){
                            return bytes;
                        }
                    }
                    const bytes = await download(url, false, entry.sha256);
                    if(bytes !== null){
                        return bytes;
                    }
                }
            }
            catch(error){
                failures.push(`${entry_point}: ${error.message}`);
            }
        }
        throw new Error(`Could not load ${reference}\n${failures.join("\n")}`);
    }
}
