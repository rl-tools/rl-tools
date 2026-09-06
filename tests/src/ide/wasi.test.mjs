// Unit tests of the WASI preview1 layer: syscalls are called directly on a WebAssembly.Memory, plus one hand-assembled
// module for the instantiate/start path. Run with `node --test tests/src/ide/`.
import { test } from "node:test";
import assert from "node:assert/strict";
import { WASI, File, Directory, ERRNO, OFLAGS, FDFLAGS, WHENCE, CLOCKID, EVENTTYPE, RIGHTS, FILETYPE, SYSCALLS } from "../../../static/ide/wasi.js";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function harness(root, options = {}){
    const wasi = new WASI({ root, ...options });
    wasi.memory = new WebAssembly.Memory({ initial: 2 });
    const bytes = () => new Uint8Array(wasi.memory.buffer);
    const view = () => new DataView(wasi.memory.buffer);
    let cursor = 1024;
    const alloc = size => {
        const pointer = cursor;
        cursor += (size + 7) & ~7;
        return pointer;
    };
    const put = text => {
        const encoded = encoder.encode(text);
        const pointer = alloc(encoded.length);
        bytes().set(encoded, pointer);
        return [pointer, encoded.length];
    };
    const iov = (pointer, length) => {
        const iovs = alloc(8);
        view().setUint32(iovs, pointer, true);
        view().setUint32(iovs + 4, length, true);
        return iovs;
    };
    const syscalls = wasi.imports;
    const open = (fd, path, oflags = 0, rights = RIGHTS.FD_READ, fdflags = 0) => {
        const result = alloc(4);
        const error = syscalls.path_open(fd, 0, ...put(path), oflags, rights, RIGHTS.ALL, fdflags, result);
        return [error, view().getUint32(result, true)];
    };
    const write = (fd, text) => {
        const [pointer, length] = put(text);
        const result = alloc(4);
        const error = syscalls.fd_write(fd, iov(pointer, length), 1, result);
        return [error, view().getUint32(result, true)];
    };
    const read = (fd, length) => {
        const buffer = alloc(length);
        const result = alloc(4);
        const error = syscalls.fd_read(fd, iov(buffer, length), 1, result);
        const count = view().getUint32(result, true);
        return [error, decoder.decode(bytes().slice(buffer, buffer + count))];
    };
    return { wasi, syscalls, bytes, view, alloc, put, iov, open, write, read };
}

function sampleRoot(){
    const archive = encoder.encode("archived header contents");
    return new Map([
        ["usr", new Directory(new Map([
            ["include", new Directory(new Map([["stddef.h", new File(archive, { readonly: true })]]), { readonly: true })],
            ["lib", new Directory(new Map([["libc.a", new File(encoder.encode("!<arch>"), { readonly: true })]]), { readonly: true })],
        ]), { readonly: true })],
        ["tmp", new Directory()],
        ["training.cpp", new File(encoder.encode("int main(){}"))],
    ]);
}

test("preopen: fd 3 is the root, the scan ends with EBADF", () => {
    const { syscalls, view, bytes, alloc } = harness(sampleRoot());
    const prestat = alloc(8);
    assert.equal(syscalls.fd_prestat_get(3, prestat), ERRNO.SUCCESS);
    assert.equal(view().getUint8(prestat), 0);
    assert.equal(view().getUint32(prestat + 4, true), 1);
    const name = alloc(8);
    assert.equal(syscalls.fd_prestat_dir_name(3, name, 1), ERRNO.SUCCESS);
    assert.equal(decoder.decode(bytes().slice(name, name + 1)), "/");
    assert.equal(syscalls.fd_prestat_dir_name(3, name, 0), ERRNO.NAMETOOLONG);
    assert.equal(syscalls.fd_prestat_get(4, prestat), ERRNO.BADF);
    assert.equal(syscalls.fd_prestat_get(0, prestat), ERRNO.BADF);
});

test("stdio descriptors: character devices with full rights, stdin at EOF", () => {
    const out = [];
    const { syscalls, view, alloc, write, read } = harness(new Map(), { stdout: chunk => out.push(decoder.decode(chunk)) });
    const fdstat = alloc(24);
    assert.equal(syscalls.fd_fdstat_get(1, fdstat), ERRNO.SUCCESS);
    assert.equal(view().getUint8(fdstat), FILETYPE.CHARACTER_DEVICE);
    assert.equal(view().getBigUint64(fdstat + 8, true), RIGHTS.ALL);
    assert.equal(view().getBigUint64(fdstat + 16, true), RIGHTS.ALL);
    assert.deepEqual(write(1, "hello "), [ERRNO.SUCCESS, 6]);
    assert.deepEqual(write(1, "world\n"), [ERRNO.SUCCESS, 6]);
    assert.equal(out.join(""), "hello world\n");
    assert.deepEqual(read(0, 16), [ERRNO.SUCCESS, ""]);
    assert.equal(write(0, "x")[0], ERRNO.BADF);
    assert.equal(read(1, 16)[0], ERRNO.BADF);
    const offset = alloc(8);
    assert.equal(syscalls.fd_seek(1, 0n, WHENCE.CUR, offset), ERRNO.SPIPE);
});

test("files: create, write, seek, read, stat, truncate, close", () => {
    const { syscalls, view, alloc, open, write, read, put } = harness(sampleRoot());
    const [openError, fd] = open(3, "tmp/out.o", OFLAGS.CREAT | OFLAGS.TRUNC, RIGHTS.FD_READ | RIGHTS.FD_WRITE);
    assert.equal(openError, ERRNO.SUCCESS);
    assert.ok(fd >= 4);
    assert.deepEqual(write(fd, "abcdef"), [ERRNO.SUCCESS, 6]);
    const offset = alloc(8);
    assert.equal(syscalls.fd_seek(fd, 2n, WHENCE.SET, offset), ERRNO.SUCCESS);
    assert.equal(view().getBigUint64(offset, true), 2n);
    assert.deepEqual(read(fd, 16), [ERRNO.SUCCESS, "cdef"]);
    assert.equal(syscalls.fd_seek(fd, -1n, WHENCE.END, offset), ERRNO.SUCCESS);
    assert.deepEqual(read(fd, 16), [ERRNO.SUCCESS, "f"]);
    assert.equal(syscalls.fd_seek(fd, -10n, WHENCE.CUR, offset), ERRNO.INVAL);
    const filestat = alloc(64);
    assert.equal(syscalls.fd_filestat_get(fd, filestat), ERRNO.SUCCESS);
    assert.equal(view().getUint8(filestat + 16), FILETYPE.REGULAR_FILE);
    assert.equal(view().getBigUint64(filestat + 32, true), 6n);
    assert.equal(syscalls.fd_filestat_set_size(fd, 3n), ERRNO.SUCCESS);
    assert.equal(syscalls.path_filestat_get(3, 0, ...put("tmp/out.o"), filestat), ERRNO.SUCCESS);
    assert.equal(view().getBigUint64(filestat + 32, true), 3n);
    assert.equal(syscalls.fd_filestat_set_size(fd, 8n), ERRNO.SUCCESS);
    assert.equal(syscalls.fd_seek(fd, 0n, WHENCE.SET, offset), ERRNO.SUCCESS);
    assert.deepEqual(read(fd, 16), [ERRNO.SUCCESS, "abc\0\0\0\0\0"]);
    assert.equal(syscalls.fd_close(fd), ERRNO.SUCCESS);
    assert.equal(syscalls.fd_close(fd), ERRNO.BADF);
    const [again, fd2] = open(3, "tmp/out.o", OFLAGS.CREAT | OFLAGS.EXCL);
    assert.equal(again, ERRNO.EXIST);
    assert.equal(open(3, "tmp/missing.o")[0], ERRNO.NOENT);
    assert.equal(open(3, "training.cpp/x")[0], ERRNO.NOTDIR);
});

test("pread and pwrite leave the position alone; append opens seek to the end", () => {
    const { syscalls, view, alloc, open, write, read, put, iov } = harness(sampleRoot());
    const [, fd] = open(3, "tmp/log", OFLAGS.CREAT, RIGHTS.FD_READ | RIGHTS.FD_WRITE);
    write(fd, "0123456789");
    const [pointer, length] = put("XY");
    const result = alloc(4);
    assert.equal(syscalls.fd_pwrite(fd, iov(pointer, length), 1, 4n, result), ERRNO.SUCCESS);
    const offset = alloc(8);
    assert.equal(syscalls.fd_tell(fd, offset), ERRNO.SUCCESS);
    assert.equal(view().getBigUint64(offset, true), 10n);
    const buffer = alloc(4);
    assert.equal(syscalls.fd_pread(fd, iov(buffer, 4), 1, 3n, result), ERRNO.SUCCESS);
    assert.equal(view().getUint32(result, true), 4);
    assert.equal(decoder.decode(new Uint8Array(view().buffer, buffer, 4)), "3XY6");
    const [, appendFd] = open(3, "tmp/log", 0, RIGHTS.FD_WRITE, FDFLAGS.APPEND);
    write(appendFd, "!");
    syscalls.fd_seek(appendFd, 0n, WHENCE.SET, offset);
    assert.deepEqual(read(appendFd, 32), [ERRNO.SUCCESS, "0123XY6789!"]);
});

test("read-only trees: reads succeed, any write path reports EROFS, archive bytes stay untouched", () => {
    const root = sampleRoot();
    const archive = root.get("usr").contents.get("include").contents.get("stddef.h").bytes;
    const { syscalls, open, read, put, write } = harness(root);
    const [error, fd] = open(3, "usr/include/stddef.h");
    assert.equal(error, ERRNO.SUCCESS);
    assert.deepEqual(read(fd, 64), [ERRNO.SUCCESS, "archived header contents"]);
    assert.equal(write(fd, "x")[0], ERRNO.ROFS);
    assert.equal(open(3, "usr/include/stddef.h", 0, RIGHTS.FD_WRITE)[0], ERRNO.ROFS);
    assert.equal(open(3, "usr/include/stddef.h", OFLAGS.TRUNC)[0], ERRNO.ROFS);
    assert.equal(open(3, "usr/include/new.h", OFLAGS.CREAT, RIGHTS.FD_WRITE)[0], ERRNO.ROFS);
    assert.equal(syscalls.path_create_directory(3, ...put("usr/include/sub")), ERRNO.ROFS);
    assert.equal(syscalls.path_unlink_file(3, ...put("usr/include/stddef.h")), ERRNO.ROFS);
    assert.equal(syscalls.path_rename(3, ...put("usr/lib/libc.a"), 3, ...put("tmp/libc.a")), ERRNO.ROFS);
    assert.equal(decoder.decode(archive), "archived header contents");
});

test("writing a file backed by shared archive bytes copies first", () => {
    const shared = encoder.encode("shared");
    const file = new File(shared.subarray(0, 6));
    file.write(0, encoder.encode("S"));
    assert.equal(decoder.decode(shared), "shared");
    assert.equal(decoder.decode(file.data), "Shared");
    file.write(10, encoder.encode("!"));
    assert.equal(file.size, 11);
    assert.equal(file.data[8], 0);
    file.resize(2);
    assert.equal(decoder.decode(file.data), "Sh");
    const grown = new File(shared);
    grown.resize(3);
    grown.write(3, encoder.encode("XYZ"));
    assert.equal(decoder.decode(grown.data), "shaXYZ");
    assert.equal(decoder.decode(shared), "shared");
});

test("path resolution: dot, dot-dot, trailing slash, sandbox escape", () => {
    const { syscalls, open, put, alloc } = harness(sampleRoot());
    assert.equal(open(3, "./usr/include/../lib/libc.a")[0], ERRNO.SUCCESS);
    assert.equal(open(3, "usr//lib/./libc.a")[0], ERRNO.SUCCESS);
    assert.equal(open(3, "usr/lib/")[0], ERRNO.SUCCESS);
    assert.equal(open(3, "usr/lib/libc.a/")[0], ERRNO.NOTDIR);
    assert.equal(open(3, "../etc/passwd")[0], ERRNO.NOTCAPABLE);
    assert.equal(open(3, "usr/../../x")[0], ERRNO.NOTCAPABLE);
    assert.equal(open(3, "/usr")[0], ERRNO.NOTCAPABLE);
    const [directoryError, directoryFd] = open(3, "usr", OFLAGS.DIRECTORY);
    assert.equal(directoryError, ERRNO.SUCCESS);
    assert.equal(open(directoryFd, "lib/libc.a")[0], ERRNO.SUCCESS);
    assert.equal(open(directoryFd, "../tmp")[0], ERRNO.NOTCAPABLE);
    assert.equal(open(3, "training.cpp", OFLAGS.DIRECTORY)[0], ERRNO.NOTDIR);
    const filestat = alloc(64);
    assert.equal(syscalls.path_filestat_get(3, 0, ...put("."), filestat), ERRNO.SUCCESS);
    assert.equal(syscalls.path_filestat_get(3, 0, ...put(""), filestat), ERRNO.SUCCESS);
    assert.equal(syscalls.path_filestat_get(3, 0, ...put("nope"), filestat), ERRNO.NOENT);
    assert.equal(syscalls.path_filestat_get(0, 0, ...put("x"), filestat), ERRNO.NOTDIR);
    assert.equal(syscalls.path_filestat_get(9, 0, ...put("x"), filestat), ERRNO.BADF);
});

test("readdir: dot entries first, truncation reports a full buffer, cookies resume", () => {
    const { syscalls, view, bytes, alloc, open } = harness(sampleRoot());
    const [, fd] = open(3, "usr", OFLAGS.DIRECTORY);
    const result = alloc(4);
    const small = alloc(30);
    assert.equal(syscalls.fd_readdir(fd, small, 30, 0n, result), ERRNO.SUCCESS);
    assert.equal(view().getUint32(result, true), 30);
    const large = alloc(512);
    assert.equal(syscalls.fd_readdir(fd, large, 512, 0n, result), ERRNO.SUCCESS);
    const used = view().getUint32(result, true);
    const names = [];
    let offset = 0;
    let next = 0n;
    while(offset < used){
        next = view().getBigUint64(large + offset, true);
        const nameLength = view().getUint32(large + offset + 16, true);
        const type = view().getUint8(large + offset + 20);
        names.push([decoder.decode(bytes().slice(large + offset + 24, large + offset + 24 + nameLength)), type]);
        offset += 24 + nameLength;
    }
    assert.deepEqual(names, [[".", FILETYPE.DIRECTORY], ["..", FILETYPE.DIRECTORY], ["include", FILETYPE.DIRECTORY], ["lib", FILETYPE.DIRECTORY]]);
    assert.equal(next, 4n);
    assert.equal(syscalls.fd_readdir(fd, large, 512, next, result), ERRNO.SUCCESS);
    assert.equal(view().getUint32(result, true), 0);
    assert.equal(syscalls.fd_readdir(fd, large, 512, 2n, result), ERRNO.SUCCESS);
    assert.equal(view().getUint32(result, true), used - (24 + 1) - (24 + 2));
    assert.equal(syscalls.fd_readdir(0, large, 512, 0n, result), ERRNO.NOTDIR);
});

test("directories: create, rename, unlink, remove", () => {
    const { syscalls, put, open, write } = harness(sampleRoot());
    assert.equal(syscalls.path_create_directory(3, ...put("tmp/work")), ERRNO.SUCCESS);
    assert.equal(syscalls.path_create_directory(3, ...put("tmp/work")), ERRNO.EXIST);
    assert.equal(syscalls.path_create_directory(3, ...put("nope/work")), ERRNO.NOENT);
    const [, fd] = open(3, "tmp/work/a.o", OFLAGS.CREAT, RIGHTS.FD_WRITE);
    write(fd, "object");
    assert.equal(syscalls.path_remove_directory(3, ...put("tmp/work")), ERRNO.NOTEMPTY);
    assert.equal(syscalls.path_remove_directory(3, ...put("tmp/work/a.o")), ERRNO.NOTDIR);
    assert.equal(syscalls.path_unlink_file(3, ...put("tmp/work")), ERRNO.ISDIR);
    assert.equal(syscalls.path_rename(3, ...put("tmp/work/a.o"), 3, ...put("tmp/b.o")), ERRNO.SUCCESS);
    assert.equal(open(3, "tmp/work/a.o")[0], ERRNO.NOENT);
    assert.equal(open(3, "tmp/b.o")[0], ERRNO.SUCCESS);
    assert.equal(syscalls.path_rename(3, ...put("tmp/b.o"), 3, ...put("tmp/work")), ERRNO.ISDIR);
    assert.equal(syscalls.path_rename(3, ...put("tmp/work"), 3, ...put("tmp/b.o")), ERRNO.NOTDIR);
    assert.equal(syscalls.path_rename(3, ...put("tmp/b.o"), 3, ...put("tmp/b.o")), ERRNO.SUCCESS);
    assert.equal(syscalls.path_rename(3, ...put("tmp/missing"), 3, ...put("tmp/x")), ERRNO.NOENT);
    assert.equal(syscalls.path_remove_directory(3, ...put("tmp/work")), ERRNO.SUCCESS);
    assert.equal(syscalls.path_remove_directory(3, ...put("tmp/work")), ERRNO.NOENT);
    assert.equal(syscalls.path_unlink_file(3, ...put("tmp/b.o")), ERRNO.SUCCESS);
    assert.equal(syscalls.path_unlink_file(3, ...put("tmp/b.o")), ERRNO.NOENT);
    assert.equal(syscalls.path_remove_directory(3, ...put(".")), ERRNO.INVAL);
});

test("links: readlink says not-a-symlink, creating links is unsupported", () => {
    const { syscalls, put, alloc } = harness(sampleRoot());
    const buffer = alloc(64);
    const result = alloc(4);
    assert.equal(syscalls.path_readlink(3, ...put("training.cpp"), buffer, 64, result), ERRNO.INVAL);
    assert.equal(syscalls.path_readlink(3, ...put("usr"), buffer, 64, result), ERRNO.INVAL);
    assert.equal(syscalls.path_readlink(3, ...put("missing"), buffer, 64, result), ERRNO.NOENT);
    assert.equal(syscalls.path_symlink(...put("training.cpp"), 3, ...put("tmp/link")), ERRNO.NOTSUP);
    assert.equal(syscalls.path_link(3, 0, ...put("training.cpp"), 3, ...put("tmp/link")), ERRNO.NOTSUP);
});

test("arguments and environment", () => {
    const { syscalls, view, bytes, alloc } = harness(new Map(), { args: ["llvm", "clang++", "--version"], env: ["HOME=/", "LANG=C"] });
    const count = alloc(4);
    const size = alloc(4);
    assert.equal(syscalls.args_sizes_get(count, size), ERRNO.SUCCESS);
    assert.equal(view().getUint32(count, true), 3);
    assert.equal(view().getUint32(size, true), 5 + 8 + 10);
    const pointers = alloc(12);
    const buffer = alloc(32);
    assert.equal(syscalls.args_get(pointers, buffer), ERRNO.SUCCESS);
    const argument = index => {
        const pointer = view().getUint32(pointers + 4 * index, true);
        let end = pointer;
        while(bytes()[end] !== 0){
            end++;
        }
        return decoder.decode(bytes().slice(pointer, end));
    };
    assert.deepEqual([argument(0), argument(1), argument(2)], ["llvm", "clang++", "--version"]);
    assert.equal(syscalls.environ_sizes_get(count, size), ERRNO.SUCCESS);
    assert.equal(view().getUint32(count, true), 2);
    assert.equal(view().getUint32(size, true), 7 + 7);
    assert.equal(syscalls.environ_get(pointers, buffer), ERRNO.SUCCESS);
    assert.equal(argument(1), "LANG=C");
});

test("clocks, randomness, polling", () => {
    const { syscalls, view, bytes, alloc } = harness(new Map());
    const time = alloc(8);
    assert.equal(syscalls.clock_time_get(CLOCKID.MONOTONIC, 1n, time), ERRNO.SUCCESS);
    const first = view().getBigUint64(time, true);
    assert.equal(syscalls.clock_time_get(CLOCKID.MONOTONIC, 1n, time), ERRNO.SUCCESS);
    assert.ok(view().getBigUint64(time, true) >= first);
    assert.equal(syscalls.clock_time_get(CLOCKID.REALTIME, 1n, time), ERRNO.SUCCESS);
    assert.ok(view().getBigUint64(time, true) > 1600000000n * 1000000000n);
    assert.equal(syscalls.clock_time_get(7, 1n, time), ERRNO.INVAL);
    assert.equal(syscalls.clock_res_get(CLOCKID.REALTIME, time), ERRNO.SUCCESS);
    const random = alloc(70000);
    assert.equal(syscalls.random_get(random, 70000), ERRNO.SUCCESS);
    assert.ok(new Set(bytes().slice(random, random + 70000)).size > 200);
    const subscriptions = alloc(96);
    const events = alloc(64);
    const count = alloc(4);
    view().setBigUint64(subscriptions, 42n, true);
    view().setUint8(subscriptions + 8, EVENTTYPE.CLOCK);
    view().setUint32(subscriptions + 16, CLOCKID.MONOTONIC, true);
    view().setBigUint64(subscriptions + 24, 3000000n, true);
    view().setBigUint64(subscriptions + 48, 43n, true);
    view().setUint8(subscriptions + 56, EVENTTYPE.FD_READ);
    view().setUint32(subscriptions + 64, 0, true);
    const started = performance.now();
    assert.equal(syscalls.poll_oneoff(subscriptions, events, 2, count), ERRNO.SUCCESS);
    assert.ok(performance.now() - started >= 2.5);
    assert.equal(view().getUint32(count, true), 2);
    assert.equal(view().getBigUint64(events, true), 42n);
    assert.equal(view().getUint16(events + 8, true), ERRNO.SUCCESS);
    assert.equal(view().getUint8(events + 10), EVENTTYPE.CLOCK);
    assert.equal(view().getBigUint64(events + 32, true), 43n);
    assert.equal(view().getUint8(events + 42), EVENTTYPE.FD_READ);
    assert.equal(syscalls.sched_yield(), ERRNO.SUCCESS);
    assert.equal(syscalls.proc_raise(6), ERRNO.NOTSUP);
    assert.equal(syscalls.sock_accept(3, 0, count), ERRNO.NOTSUP);
});

test("trace hook sees every call", () => {
    const calls = [];
    const { syscalls, alloc } = harness(new Map(), { trace: (name, syscallArguments, result) => calls.push([name, result]) });
    syscalls.fd_close(9);
    syscalls.clock_res_get(0, alloc(8));
    assert.deepEqual(calls, [["fd_close", ERRNO.BADF], ["clock_res_get", ERRNO.SUCCESS]]);
    assert.throws(() => syscalls.proc_exit(3));
    assert.equal(calls[2][0], "proc_exit");
    assert.equal(calls[2][1].code, 3);
});

function unsignedLeb(value){
    const out = [];
    do{
        let byte = value & 0x7f;
        value >>>= 7;
        if(value !== 0){
            byte |= 0x80;
        }
        out.push(byte);
    }
    while(value !== 0);
    return out;
}
const vector = items => [...unsignedLeb(items.length), ...items.flat()];
const string = text => vector([...encoder.encode(text)].map(byte => [byte]));
const section = (id, body) => [id, ...unsignedLeb(body.length), ...body];

// (module (import "wasi_snapshot_preview1" "fd_write") (import ... "proc_exit") (memory 1) (data "hello\n" iov)
//   (func $_start (call $fd_write 1 iov 1 nwritten) drop (call $proc_exit 7)))
function helloModule(){
    const types = vector([
        [0x60, ...vector([[0x7f], [0x7f], [0x7f], [0x7f]]), ...vector([[0x7f]])],
        [0x60, ...vector([[0x7f]]), ...vector([])],
        [0x60, ...vector([]), ...vector([])],
    ]);
    const imports = vector([
        [...string("wasi_snapshot_preview1"), ...string("fd_write"), 0x00, 0],
        [...string("wasi_snapshot_preview1"), ...string("proc_exit"), 0x00, 1],
    ]);
    const functions = vector([[2]]);
    const memory = vector([[0x00, 1]]);
    const exports = vector([[...string("memory"), 0x02, 0], [...string("_start"), 0x00, 2]]);
    const body = [0, 0x41, 1, 0x41, 8, 0x41, 1, 0x41, 24, 0x10, 0, 0x1a, 0x41, 7, 0x10, 1, 0x0b];
    const code = vector([[...unsignedLeb(body.length), ...body]]);
    const payload = [...encoder.encode("hello\n"), 0, 0, 0, 0, 0, 0, 6, 0, 0, 0];
    const data = vector([[0x00, 0x41, 0, 0x0b, ...vector(payload.map(byte => [byte]))]]);
    return new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0, ...section(1, types), ...section(2, imports), ...section(3, functions), ...section(5, memory), ...section(7, exports), ...section(10, code), ...section(11, data)]);
}

test("instantiate and start a module: stdout bytes and the exit code arrive", () => {
    const out = [];
    const wasi = new WASI({ args: ["hello"], root: new Map(), stdout: chunk => out.push(decoder.decode(chunk)) });
    const module = new WebAssembly.Module(helloModule());
    assert.deepEqual(WebAssembly.Module.imports(module).map(entry => entry.name), ["fd_write", "proc_exit"]);
    const instance = new WebAssembly.Instance(module, wasi.importObject);
    assert.equal(wasi.start(instance), 7);
    assert.equal(out.join(""), "hello\n");
});

test("the import object covers the whole preview1 surface", () => {
    const wasi = new WASI({ root: new Map() });
    assert.deepEqual(Object.keys(wasi.importObject.wasi_snapshot_preview1).sort(), [...SYSCALLS].sort());
    assert.equal(SYSCALLS.length, 46);
});
