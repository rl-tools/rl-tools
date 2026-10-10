import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {gzipSync, gunzipSync} from "node:zlib";
import {test} from "node:test";
import {Conta} from "../../../static/hyperdrone/conta.js";

const data = new TextEncoder().encode("A small content-addressed scene fixture");
const sha1 = createHash("sha1").update(data).digest("hex");
const sha256 = createHash("sha256").update(data).digest("hex");
const compressed = gzipSync(data);
const overlay = "https://page.test/blob/";
const entry_point = "https://index.test/version/1/meta.json";
const index_url = "https://index.test/version/1/generated/index.json";
const blob_url = `https://shard.test/${sha1}?download=true`;
const gzip_url = `https://shard.test/${sha1}.gz?download=true`;

function index_routes({compressed = ["gz"], digest = sha256} = {}){
    const index = JSON.stringify({
        shards: {fixture: {url: "https://shard.test/{sha1}?download=true"}},
        entries: [{sha1, sha256: digest, shards: [{id: "fixture", compressed}]}],
    });
    return {
        [entry_point]: JSON.stringify({format: 1, index: {
            location: "generated/index.json",
            sha256: createHash("sha256").update(index).digest("hex"),
        }}),
        [index_url]: index,
    };
}

function network(t, routes){
    const requests = [];
    t.mock.method(globalThis, "fetch", async url => {
        const key = String(url);
        requests.push(key);
        if(!(key in routes)){
            return new Response(null, {status: 404});
        }
        const body = routes[key];
        return new Response(body, {headers: {"Content-Length": String(Buffer.byteLength(body))}});
    });
    return requests;
}

function browser_cache(t){
    const entries = new Map();
    const cache = {
        async match(key){
            return entries.get(String(key))?.clone();
        },
        async put(key, response){
            entries.set(String(key), new Response(await response.arrayBuffer(), {headers: response.headers}));
        },
        async delete(key){
            return entries.delete(String(key));
        },
    };
    const storage = {async open(){ return cache; }};
    const original = Object.getOwnPropertyDescriptor(globalThis, "caches");
    Object.defineProperty(globalThis, "caches", {value: storage, configurable: true});
    t.after(() => {
        if(original){
            Object.defineProperty(globalThis, "caches", original);
        }
        else{
            delete globalThis.caches;
        }
    });
    return {cache, storage};
}

test("local gzip overlay loads offline and reports compressed transfer bytes", async t => {
    const requests = network(t, {[`${overlay}${sha1}.gz`]: compressed});
    const progress = [];
    const conta = new Conta({overlay, entry_points: [entry_point]});
    const result = await conta.load(`conta:${sha1}`, {sha256, on_progress: value => progress.push(value)});
    assert.deepEqual(result, data);
    assert.deepEqual(requests, [`${overlay}${sha1}.gz`]);
    assert.equal(progress.at(-1).received, compressed.byteLength);
    assert.equal(progress.at(-1).total, compressed.byteLength);
});

test("missing local blob resolves the index and prefers remote gzip", async t => {
    const requests = network(t, {...index_routes(), [gzip_url]: compressed});
    const conta = new Conta({overlay, entry_points: [entry_point]});
    assert.deepEqual(await conta.load(sha1), data);
    assert.deepEqual(requests, [`${overlay}${sha1}.gz`, entry_point, index_url, gzip_url]);
});

test("corrupt local gzip falls through to the indexed remote provider", async t => {
    network(t, {...index_routes(), [`${overlay}${sha1}.gz`]: gzipSync("bad bytes"), [gzip_url]: compressed});
    const conta = new Conta({overlay, entry_points: [entry_point]});
    assert.deepEqual(await conta.load(sha1), data);
});

test("missing remote gzip falls back to the same provider's raw asset", async t => {
    const requests = network(t, {...index_routes(), [blob_url]: data});
    const conta = new Conta({overlay, entry_points: [entry_point]});
    assert.deepEqual(await conta.load(sha1), data);
    assert.deepEqual(requests.slice(-2), [gzip_url, blob_url]);
});

test("uncompressed providers are supported without guessing gzip availability", async t => {
    const requests = network(t, {...index_routes({compressed: []}), [blob_url]: data});
    const conta = new Conta({overlay, entry_points: [entry_point]});
    assert.deepEqual(await conta.load(sha1), data);
    assert.ok(!requests.includes(gzip_url));
});

test("index integrity failures try the next entry point", async t => {
    const bad_point = "https://stale.test/meta.json";
    network(t, {
        ...index_routes(), [gzip_url]: compressed,
        [bad_point]: JSON.stringify({format: 1, index: {location: index_url, sha256: "0".repeat(64)}}),
    });
    const conta = new Conta({overlay, entry_points: [bad_point, entry_point]});
    assert.deepEqual(await conta.load(sha1), data);
});

test("remote decoded content must match both indexed hashes", async t => {
    network(t, {...index_routes({digest: "0".repeat(64)}), [gzip_url]: compressed, [blob_url]: data});
    const conta = new Conta({overlay, entry_points: [entry_point]});
    await assert.rejects(conta.load(sha1), /SHA-256 mismatch/);
});

test("the registry SHA-256 cannot be overridden by an index", async t => {
    const requests = network(t, {...index_routes(), [gzip_url]: compressed});
    const conta = new Conta({overlay, entry_points: [entry_point]});
    await assert.rejects(conta.load(sha1, {sha256: "0".repeat(64)}), /disagrees with the pinned asset/);
    assert.ok(!requests.includes(gzip_url));
});

test("malformed references do not issue requests", async t => {
    const requests = network(t, {});
    await assert.rejects(new Conta({overlay}).load("../not-a-hash"), /Invalid conta reference/);
    assert.equal(requests.length, 0);
});

test("local gzip persists across resolver instances with no repeat requests", async t => {
    const {cache} = browser_cache(t);
    const requests = network(t, {[`${overlay}${sha1}.gz`]: compressed});
    assert.deepEqual(await new Conta({overlay}).load(sha1, {sha256}), data);
    const stored = await cache.match(`${overlay}${sha1}.gz`);
    assert.equal(stored.headers.get("Content-Type"), "application/gzip");
    assert.deepEqual(Buffer.from(await stored.arrayBuffer()), compressed);
    requests.length = 0;
    const progress = [];
    assert.deepEqual(await new Conta({overlay}).load(sha1, {sha256, on_progress: event => progress.push(event)}), data);
    assert.deepEqual(requests, []);
    assert.equal(progress[0].phase, "cache");
    assert.equal(progress.at(-1).received, compressed.byteLength);
    assert.ok(!progress.some(event => event.phase === "download"));
});

test("remote gzip is reusable offline without resolving an index", async t => {
    const {cache} = browser_cache(t);
    const routes = {...index_routes(), [gzip_url]: compressed};
    const requests = network(t, routes);
    assert.deepEqual(await new Conta({overlay, entry_points: [entry_point]}).load(sha1), data);
    assert.equal((await cache.match(`${overlay}${sha1}.gz`)).headers.get("X-Conta-SHA256"), sha256);
    for(const key of Object.keys(routes)){
        delete routes[key];
    }
    requests.length = 0;
    assert.deepEqual(await new Conta({overlay, entry_points: [entry_point]}).load(sha1), data);
    assert.deepEqual(requests, []);
});

test("raw-only assets are cached as gzip and reused without requests", async t => {
    const {cache} = browser_cache(t);
    const requests = network(t, {...index_routes({compressed: []}), [blob_url]: data});
    assert.deepEqual(await new Conta({overlay, entry_points: [entry_point]}).load(sha1), data);
    const stored = Buffer.from(await (await cache.match(`${overlay}${sha1}.gz`)).arrayBuffer());
    assert.deepEqual(new Uint8Array(gunzipSync(stored)), data);
    requests.length = 0;
    assert.deepEqual(await new Conta({overlay}).load(sha1), data);
    assert.deepEqual(requests, []);
});

for(const [description, body, digest] of [
    ["invalid gzip", "not gzip", sha256],
    ["wrong decoded SHA-1", gzipSync("corrupt scene"), sha256],
    ["wrong recorded SHA-256", compressed, "0".repeat(64)],
]){
    test(`cache with ${description} is discarded and repaired from the overlay`, async t => {
        const {cache} = browser_cache(t);
        await cache.put(`${overlay}${sha1}.gz`, new Response(body, {headers: {"X-Conta-SHA256": digest}}));
        const requests = network(t, {[`${overlay}${sha1}.gz`]: compressed});
        assert.deepEqual(await new Conta({overlay}).load(sha1), data);
        assert.deepEqual(requests, [`${overlay}${sha1}.gz`]);
        assert.deepEqual(Buffer.from(await (await cache.match(`${overlay}${sha1}.gz`)).arrayBuffer()), compressed);
    });
}

test("cached bytes cannot bypass a caller's pinned SHA-256", async t => {
    const {cache} = browser_cache(t);
    await cache.put(`${overlay}${sha1}.gz`, new Response(compressed, {headers: {"X-Conta-SHA256": sha256}}));
    network(t, {});
    await assert.rejects(new Conta({overlay, entry_points: []}).load(sha1, {sha256: "0".repeat(64)}), /SHA-256 mismatch/);
    assert.equal(await cache.match(`${overlay}${sha1}.gz`), undefined);
});

test("unverified downloads never enter the cache", async t => {
    const {cache} = browser_cache(t);
    network(t, {...index_routes({digest: "0".repeat(64)}), [gzip_url]: compressed, [blob_url]: data});
    await assert.rejects(new Conta({overlay, entry_points: [entry_point]}).load(sha1), /SHA-256 mismatch/);
    assert.equal(await cache.match(`${overlay}${sha1}.gz`), undefined);
});

for(const method of ["open", "match", "put"]){
    test(`cache ${method} failures do not prevent loading`, async t => {
        const {cache, storage} = browser_cache(t);
        t.mock.method(method === "open" ? storage : cache, method, async () => {
            throw new Error("Storage unavailable or quota exceeded");
        });
        t.mock.method(console, "warn", () => {});
        network(t, {[`${overlay}${sha1}.gz`]: compressed});
        assert.deepEqual(await new Conta({overlay}).load(sha1), data);
    });
}
