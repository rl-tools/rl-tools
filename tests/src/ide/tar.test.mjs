// Unit tests of the tar reader against archives written by the host's tar with the recipe the packaging uses
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { untar } from "../../../static/ide/tar.js";

const decoder = new TextDecoder();

function hasGnuTar(){
    try{
        return execFileSync("tar", ["--version"]).toString().includes("GNU tar");
    }
    catch{
        return false;
    }
}

function withTree(callback){
    const directory = mkdtempSync(join(tmpdir(), "rl-tools-ide-tar-"));
    try{
        const longDirectory = join("d".repeat(50), "e".repeat(50), "f".repeat(40));
        mkdirSync(join(directory, "tree", "include", "sub"), { recursive: true });
        mkdirSync(join(directory, "tree", longDirectory), { recursive: true });
        mkdirSync(join(directory, "tree", "empty"), { recursive: true });
        writeFileSync(join(directory, "tree", "include", "a.h"), "#pragma once\n");
        writeFileSync(join(directory, "tree", "include", "sub", "b.h"), "int b;\n");
        writeFileSync(join(directory, "tree", "blob.bin"), new Uint8Array(Array.from({ length: 700 }, (_, index) => index % 251)));
        writeFileSync(join(directory, "tree", "zero"), "");
        writeFileSync(join(directory, "tree", longDirectory, "deep.txt"), "deep\n");
        return callback(directory);
    }
    finally{
        rmSync(directory, { recursive: true, force: true });
    }
}

test("ustar archive from the packaging recipe", { skip: !hasGnuTar() && "GNU tar not found" }, () => {
    withTree(directory => {
        execFileSync("tar", ["--format=ustar", "--sort=name", "--owner=0", "--group=0", "--numeric-owner", "--mtime=@1700000000", "-C", join(directory, "tree"), "-cf", join(directory, "tree.tar"), "include", "blob.bin", "zero", "empty", "d".repeat(50)]);
        const entries = untar(new Uint8Array(readFileSync(join(directory, "tree.tar"))));
        assert.deepEqual([...entries.keys()].sort(), ["blob.bin", "d".repeat(50) + "/" + "e".repeat(50) + "/" + "f".repeat(40) + "/deep.txt", "include/a.h", "include/sub/b.h", "zero"]);
        assert.equal(decoder.decode(entries.get("include/sub/b.h")), "int b;\n");
        assert.equal(entries.get("zero").length, 0);
        const blob = entries.get("blob.bin");
        assert.equal(blob.length, 700);
        assert.equal(blob[699], 699 % 251);
        assert.equal(decoder.decode(entries.get("d".repeat(50) + "/" + "e".repeat(50) + "/" + "f".repeat(40) + "/deep.txt")), "deep\n");
    });
});

test("unsupported GNU long-name entries are rejected", { skip: !hasGnuTar() && "GNU tar not found" }, () => {
    withTree(directory => {
        const longName = "n".repeat(150) + ".h";
        writeFileSync(join(directory, "tree", longName), "long\n");
        execFileSync("tar", ["--format=gnu", "-C", join(directory, "tree"), "-cf", join(directory, "tree.tar"), longName, "include"]);
        assert.throws(() => untar(new Uint8Array(readFileSync(join(directory, "tree.tar")))), /ustar/);
    });
});

test("truncated and empty input", () => {
    assert.equal(untar(new Uint8Array(0)).size, 0);
    assert.equal(untar(new Uint8Array(1024)).size, 0);
    assert.throws(() => untar(new Uint8Array(513)), /truncated/);
});
