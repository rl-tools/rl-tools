import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {gzipSync} from "node:zlib";
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
