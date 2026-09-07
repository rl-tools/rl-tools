import { test } from "node:test";
import assert from "node:assert/strict";
import { parseArguments, formatArguments } from "../../../static/ide/arguments.js";
import { parseDriverPlan, Toolchain } from "../../../static/ide/toolchain.js";
import { treeFromEntries, readFile } from "../../../static/ide/filesystem.js";
import { File } from "../../../static/ide/wasi.js";
import { lineSink } from "../../../static/ide/process.js";
import { WorkerClient } from "../../../static/ide/worker_client.js";
import { exampleArguments, checkExample } from "../../../static/ide/example.js";

test("argument quoting preserves empty values, whitespace, quotes, dollars and backslashes", () => {
    const args = ["-std=c++17", "", "-DNAME=a b", "apostrophe's", 'a"b', "a\\b", "$HOME", "line\nbreak"];
    assert.deepEqual(parseArguments(formatArguments(args)), args);
    assert.deepEqual(parseArguments("-DNAME='a b' out\\ file.cpp -o'program file.wasm'"), ["-DNAME=a b", "out file.cpp", "-oprogram file.wasm"]);
    assert.throws(() => parseArguments('"unfinished'), /unclosed/);
    assert.throws(() => parseArguments("unfinished\\"), /escape/);
});

test("driver plans share the argv quoting rules", () => {
    assert.deepEqual(parseDriverPlan(' "" "clang++" "-cc1" ""'), [["clang++", "-cc1", ""]]);
    assert.deepEqual(parseDriverPlan('clang version 22\n "/usr/bin/clang++" "-cc1" "-DNAME=a b"\n (in-process)\n "wasm-ld" "-o" "my output.wasm"'), [
        ["/usr/bin/clang++", "-cc1", "-DNAME=a b"], ["wasm-ld", "-o", "my output.wasm"],
    ]);
});

test("the final driver job selects the output, with normalized paths", async () => {
    const toolchain = new Toolchain(null, new Map());
    toolchain.run = async (argv, root, { onStderr } = {}) => {
        if(argv.includes("-###")){
            onStderr(' "wasm-ld" "-o" "sub/../actual output.wasm"');
        }
        else{
            root.set("actual output.wasm", new File(new Uint8Array([1, 2, 3])));
        }
        return 0;
    };
    const result = await toolchain.compile(new Map(), ["-owrong.wasm"]);
    assert.equal(result.exitCode, 0);
    assert.deepEqual([...result.output], [1, 2, 3]);
});

test("compiler roots share read-only headers and isolate project files and scratch", () => {
    const includes = treeFromEntries(new Map([["rl_tools/a.h", "header"]]), { readonly: true });
    const toolchain = new Toolchain(null, new Map(), includes);
    const first = toolchain.root(treeFromEntries(new Map([["source.cpp", "first"]])));
    const second = toolchain.root(treeFromEntries(new Map([["source.cpp", "second"]])));
    assert.equal(first.get("include").contents, second.get("include").contents);
    assert.equal(first.get("include").readonly, true);
    assert.equal(includes.get("rl_tools").contents.get("a.h").readonly, true);
    assert.ok(first.get("tmp") !== second.get("tmp"));
    first.get("source.cpp").write(0, new TextEncoder().encode("changed"));
    assert.equal(new TextDecoder().decode(readFile(second, "x/../source.cpp")), "second");
    assert.throws(() => toolchain.root(treeFromEntries(new Map([["include/a.h", "override"]]))), /reserved/);
});

test("filesystem paths have one normalization and reject escaping or conflicting entries", () => {
    const tree = treeFromEntries(new Map([["./x/../a", "text"]]));
    assert.equal(new TextDecoder().decode(readFile(tree, "/a")), "text");
    assert.throws(() => treeFromEntries(new Map([["../a", "text"]])), /root/);
    assert.throws(() => treeFromEntries(new Map([["a\0b", "text"]])), /NUL/);
    assert.throws(() => treeFromEntries(new Map([["a", "text"], ["a/b", "text"]])), /parent/);
});

test("stdout and stderr decode split UTF-8 independently and flush EOF", () => {
    const out = [], err = [];
    const stdout = lineSink(line => out.push(line));
    const stderr = lineSink(line => err.push(line));
    stdout.sink(new Uint8Array([0xe2]));
    stderr.sink(new TextEncoder().encode("diagnostic\n"));
    stdout.sink(new Uint8Array([0x82, 0xac, 10, 0xf0]));
    stdout.flush();
    stderr.flush();
    assert.deepEqual(out, ["€", "�"]);
    assert.deepEqual(err, ["diagnostic"]);
    stdout.sink(new TextEncoder().encode("next"));
    stdout.flush();
    assert.deepEqual(out, ["€", "�", "next"]);
});

class FakeWorker extends EventTarget{
    postMessage(message, transfer){ this.message = message; this.transfer = transfer; }
    terminate(){ this.closed = true; }
    emit(type, data){
        const event = new Event(type);
        Object.assign(event, data);
        this.dispatchEvent(event);
    }
    reply(data){ this.emit("message", { data: { id: this.message.id, ...data } }); }
}

test("worker requests route events and results, reject overlap, and ignore stale replies", async () => {
    const worker = new FakeWorker();
    const client = new WorkerClient(worker);
    const events = [];
    const first = client.request("load", {}, { onEvent: event => events.push(event) });
    await assert.rejects(client.request("compile"), /busy/);
    worker.reply({ type: "event", event: { type: "status", text: "loading" } });
    worker.reply({ type: "result", result: "ready" });
    assert.equal(await first, "ready");
    assert.equal(events.length, 1);
    const second = client.request("compile");
    worker.emit("message", { data: { id: 1, type: "result", result: "stale" } });
    worker.reply({ type: "result", result: "compiled" });
    assert.equal(await second, "compiled");
    client.terminate();
});

test("initialization timeout terminates the worker and rejects the pending request", async () => {
    const worker = new FakeWorker();
    const client = new WorkerClient(worker);
    await assert.rejects(client.request("load", {}, { timeout: 5 }), /load timed out/);
    assert.equal(worker.closed, true);
    assert.equal(client.pending, null);
    await assert.rejects(client.request("load"), /closed/);
});

test("worker errors, decode failures, remote exceptions and cancellation settle requests", async () => {
    for(const failure of ["error", "messageerror", "remote", "cancel"]){
        const worker = new FakeWorker();
        const client = new WorkerClient(worker);
        const pending = client.request("load");
        const rejected = assert.rejects(pending);
        if(failure === "remote") worker.reply({ type: "error", message: "load failed" });
        else if(failure === "cancel") client.terminate();
        else worker.emit(failure, { message: "worker failed" });
        await rejected;
        assert.equal(client.closed, true);
        assert.equal(client.pending, null);
    }
});

test("example validation shares compile arguments, output checks and learning requirements", () => {
    const example = { file: "a.cpp", output: "a.wasm", stdout: ["hello"], files: { "a.txt": "content" } };
    const files = new Map([["a.txt", new TextEncoder().encode("content")]]);
    assert.deepEqual(exampleArguments({ flags: ["-O2"], examples: { smoke: example } }, "smoke"), ["-O2", "a.cpp", "-o", "a.wasm"]);
    assert.deepEqual(checkExample(example, { exitCode: 0, lines: ["hello"], files }), []);
    assert.throws(() => checkExample(example, { exitCode: 1, lines: ["hello"], files }), /exit/);
    assert.throws(() => checkExample({ files: { "missing.txt": "" } }, { exitCode: 0, lines: [], files }), /file contents/);
    assert.throws(() => checkExample({ minimumFinalMeanReturn: -400 }, { exitCode: 0, lines: [], files }), /learning/);
});
