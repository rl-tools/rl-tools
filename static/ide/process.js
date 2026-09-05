import { WASI, File, OpenFile, ConsoleStdout, PreopenDirectory } from "./dependencies/browser_wasi_shim/index.js";

const decoder = new TextDecoder("utf-8", { fatal: false });
const ALL_RIGHTS = 0xFFFFFFFFFFFFFFFFn;

// wasi-libc answers access() from the rights that fd_fdstat_get reports for the directory, and clang checks access(W_OK) before it
// overwrites an output file. The shim reports no rights at all, which turns into "Operation not permitted", so the fdstat struct
// (filetype u8, flags u16, rights_base u64 at +8, rights_inheriting u64 at +16) is rewritten with full rights after the shim filled it.
function withFullRights(wasi){
    const imports = wasi.wasiImport;
    const original = imports.fd_fdstat_get;
    imports.fd_fdstat_get = (fd, fdstat_ptr) => {
        const ret = original(fd, fdstat_ptr);
        if(ret === 0){
            const view = new DataView(wasi.inst.exports.memory.buffer);
            view.setBigUint64(fdstat_ptr + 8, ALL_RIGHTS, true);
            view.setBigUint64(fdstat_ptr + 16, ALL_RIGHTS, true);
        }
        return ret;
    };
    return wasi;
}

function lineSink(write){
    let pending = "";
    const fd = new ConsoleStdout(bytes => {
        pending += decoder.decode(bytes, { stream: true });
        const lines = pending.split("\n");
        pending = lines.pop();
        for(const line of lines){
            write(line);
        }
    });
    const flush = () => {
        if(pending.length > 0){
            write(pending);
            pending = "";
        }
    };
    return { fd, flush };
}

// Runs one WASI command module to completion: argv in, stdout/stderr lines out, a single directory tree preopened at "/"
export async function runProcess(module, argv, { root, env = [], onStdout = () => {}, onStderr = () => {} }){
    const stdout = lineSink(onStdout);
    const stderr = lineSink(onStderr);
    const fds = [new OpenFile(new File(new Uint8Array())), stdout.fd, stderr.fd, new PreopenDirectory("/", root)];
    const wasi = withFullRights(new WASI(argv, env, fds, { debug: false }));
    const instance = await WebAssembly.instantiate(module, { wasi_snapshot_preview1: wasi.wasiImport });
    let exitCode;
    try{
        exitCode = wasi.start(instance);
    }
    catch(error){
        onStderr(`${argv[0]}: ${error.message ?? error}`);
        exitCode = 128;
    }
    stdout.flush();
    stderr.flush();
    return exitCode;
}
