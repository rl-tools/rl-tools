// WASI preview1 host for the browser IDE: one in-memory directory tree preopened at "/", stdio as byte sinks, no threads,
// no sockets. Struct layouts and errno values follow the wasi_snapshot_preview1 specification.

const encoder = new TextEncoder();
const decoder = new TextDecoder("utf-8");

export const ERRNO = Object.freeze({
    SUCCESS: 0, TOOBIG: 1, ACCES: 2, BADF: 8, BUSY: 10, EXIST: 20, INVAL: 28, IO: 29, ISDIR: 31, NAMETOOLONG: 37, NOENT: 44,
    NOMEM: 48, NOSYS: 52, NOTDIR: 54, NOTEMPTY: 55, NOTSUP: 58, PERM: 63, ROFS: 69, SPIPE: 70, NOTCAPABLE: 76,
});
export const FILETYPE = Object.freeze({ UNKNOWN: 0, CHARACTER_DEVICE: 2, DIRECTORY: 3, REGULAR_FILE: 4 });
export const OFLAGS = Object.freeze({ CREAT: 1, DIRECTORY: 2, EXCL: 4, TRUNC: 8 });
export const FDFLAGS = Object.freeze({ APPEND: 1 });
export const WHENCE = Object.freeze({ SET: 0, CUR: 1, END: 2 });
export const CLOCKID = Object.freeze({ REALTIME: 0, MONOTONIC: 1, PROCESS_CPUTIME_ID: 2, THREAD_CPUTIME_ID: 3 });
export const EVENTTYPE = Object.freeze({ CLOCK: 0, FD_READ: 1, FD_WRITE: 2 });
export const RIGHTS = Object.freeze({ FD_READ: 1n << 1n, FD_WRITE: 1n << 6n, ALL: (1n << 29n) - 1n });
const SUBCLOCKFLAGS_ABSTIME = 1;
const PREOPENTYPE_DIR = 0;
const FILESTAT_SIZE = 64;
const DIRENT_SIZE = 24;
const SUBSCRIPTION_SIZE = 48;
const EVENT_SIZE = 32;

export const SYSCALLS = Object.freeze([
    "args_get", "args_sizes_get", "environ_get", "environ_sizes_get", "clock_res_get", "clock_time_get",
    "fd_advise", "fd_allocate", "fd_close", "fd_datasync", "fd_fdstat_get", "fd_fdstat_set_flags", "fd_fdstat_set_rights",
    "fd_filestat_get", "fd_filestat_set_size", "fd_filestat_set_times", "fd_pread", "fd_prestat_get", "fd_prestat_dir_name",
    "fd_pwrite", "fd_read", "fd_readdir", "fd_renumber", "fd_seek", "fd_sync", "fd_tell", "fd_write",
    "path_create_directory", "path_filestat_get", "path_filestat_set_times", "path_link", "path_open", "path_readlink",
    "path_remove_directory", "path_rename", "path_symlink", "path_unlink_file",
    "poll_oneoff", "proc_exit", "proc_raise", "sched_yield", "random_get",
    "sock_accept", "sock_recv", "sock_send", "sock_shutdown",
]);

let inodeCounter = 0;

// A regular file. The initial bytes may be a view into a shared archive; they are copied on the first write.
export class File{
    constructor(data = new Uint8Array(0), { readonly = false } = {}){
        this.bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
        this.size = this.bytes.length;
        this.owned = false;
        this.readonly = readonly;
        this.ino = ++inodeCounter;
    }
    get data(){
        return this.bytes.subarray(0, this.size);
    }
    reserve(capacity){
        if(this.owned && this.bytes.length >= capacity){
            return;
        }
        const bytes = new Uint8Array(Math.max(capacity, 2 * this.bytes.length, 4096));
        bytes.set(this.bytes.subarray(0, this.size));
        this.bytes = bytes;
        this.owned = true;
    }
    write(offset, chunk){
        const end = offset + chunk.length;
        this.reserve(Math.max(end, this.size));
        if(offset > this.size){
            this.bytes.fill(0, this.size, offset);
        }
        this.bytes.set(chunk, offset);
        if(end > this.size){
            this.size = end;
        }
    }
    resize(size){
        if(size > this.size){
            this.reserve(size);
            this.bytes.fill(0, this.size, size);
        }
        else if(!this.owned){
            this.bytes = this.bytes.subarray(0, size);
        }
        this.size = size;
    }
}

export class Directory{
    constructor(contents = new Map(), { readonly = false } = {}){
        this.contents = contents instanceof Map ? contents : new Map(contents);
        this.readonly = readonly;
        this.ino = ++inodeCounter;
    }
}

export class ProcessExit extends Error{
    constructor(code){
        super(`process exited with code ${code}`);
        this.code = code;
    }
}

function filetypeOf(inode){
    if(inode instanceof File){
        return FILETYPE.REGULAR_FILE;
    }
    if(inode instanceof Directory){
        return FILETYPE.DIRECTORY;
    }
    return FILETYPE.CHARACTER_DEVICE;
}

function now(clock){
    if(clock === CLOCKID.REALTIME){
        return BigInt(Date.now()) * 1000000n;
    }
    return BigInt(Math.round(performance.now() * 1e6));
}

// Atomics.wait is the only blocking sleep available to a worker; it is unavailable on a main thread or without a
// SharedArrayBuffer, where a busy wait keeps the clock semantics
function sleepUntil(deadline, clock){
    const remaining = deadline - now(clock);
    if(remaining <= 0n){
        return;
    }
    if(typeof SharedArrayBuffer === "function" && typeof Atomics === "object"){
        try{
            Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, Number(remaining / 1000000n));
        }
        catch{
        }
    }
    while(now(clock) < deadline){
    }
}

// Resolves a path relative to a directory: empty and "." components are skipped, ".." pops, and a path that would leave
// the starting directory is refused since the tree is the sandbox
function resolve(directory, path){
    const parts = [];
    for(const component of path.split("/")){
        if(component === "" || component === "."){
            continue;
        }
        if(component === ".."){
            if(parts.length === 0){
                return { error: ERRNO.NOTCAPABLE };
            }
            parts.pop();
            continue;
        }
        parts.push(component);
    }
    let node = directory;
    for(const part of parts.slice(0, -1)){
        const child = node.contents.get(part);
        if(child === undefined){
            return { error: ERRNO.NOENT };
        }
        if(!(child instanceof Directory)){
            return { error: ERRNO.NOTDIR };
        }
        node = child;
    }
    if(parts.length === 0){
        return { error: 0, parent: null, name: null, inode: directory };
    }
    const name = parts[parts.length - 1];
    return { error: 0, parent: node, name, inode: node.contents.get(name) ?? null };
}

export class WASI{
    constructor({ args = [], env = [], root = new Map(), stdout = () => {}, stderr = () => {}, trace = null } = {}){
        this.args = args.map(argument => encoder.encode(argument));
        this.env = env.map(variable => encoder.encode(variable));
        const rootDirectory = root instanceof Directory ? root : new Directory(root);
        this.fds = [
            { type: "stream", write: null, ino: ++inodeCounter },
            { type: "stream", write: stdout, ino: ++inodeCounter },
            { type: "stream", write: stderr, ino: ++inodeCounter },
            { type: "directory", directory: rootDirectory, preopen: "/" },
        ];
        this.memory = null;
        this.imports = {};
        for(const name of SYSCALLS){
            const method = this[name].bind(this);
            this.imports[name] = trace === null ? method : (...syscallArguments) => {
                let result;
                try{
                    result = method(...syscallArguments);
                }
                catch(error){
                    trace(name, syscallArguments, error);
                    throw error;
                }
                trace(name, syscallArguments, result);
                return result;
            };
        }
    }
    get importObject(){
        return { wasi_snapshot_preview1: this.imports };
    }
    start(instance){
        this.memory = instance.exports.memory;
        try{
            instance.exports._start();
            return 0;
        }
        catch(error){
            if(error instanceof ProcessExit){
                return error.code;
            }
            throw error;
        }
    }

    view(){
        return new DataView(this.memory.buffer);
    }
    bytes(){
        return new Uint8Array(this.memory.buffer);
    }
    string(pointer, length){
        return decoder.decode(this.bytes().subarray(pointer, pointer + length));
    }
    allocate(entry, resultPointer){
        let fd = 3;
        while(this.fds[fd] !== undefined){
            fd++;
        }
        this.fds[fd] = entry;
        this.view().setUint32(resultPointer, fd, true);
        return ERRNO.SUCCESS;
    }
    lookup(fd, pathPointer, pathLength){
        const entry = this.fds[fd];
        if(entry === undefined){
            return { error: ERRNO.BADF };
        }
        if(entry.type !== "directory"){
            return { error: ERRNO.NOTDIR };
        }
        const path = this.string(pathPointer, pathLength);
        if(path.startsWith("/")){
            return { error: ERRNO.NOTCAPABLE };
        }
        if(path.includes("\0")){
            return { error: ERRNO.INVAL };
        }
        const resolved = resolve(entry.directory, path);
        if(resolved.error !== 0){
            return resolved;
        }
        resolved.trailingSlash = path.endsWith("/");
        if(resolved.trailingSlash && resolved.inode !== null && !(resolved.inode instanceof Directory)){
            return { error: ERRNO.NOTDIR };
        }
        return resolved;
    }
    writeFilestat(pointer, inode){
        const view = this.view();
        this.bytes().fill(0, pointer, pointer + FILESTAT_SIZE);
        view.setBigUint64(pointer + 8, BigInt(inode.ino), true);
        view.setUint8(pointer + 16, filetypeOf(inode));
        view.setBigUint64(pointer + 24, 1n, true);
        view.setBigUint64(pointer + 32, BigInt(inode instanceof File ? inode.size : 0), true);
    }
    writeSizes(list, countPointer, sizePointer){
        const view = this.view();
        view.setUint32(countPointer, list.length, true);
        view.setUint32(sizePointer, list.reduce((sum, item) => sum + item.length + 1, 0), true);
        return ERRNO.SUCCESS;
    }
    writeStrings(list, pointersPointer, bufferPointer){
        const view = this.view();
        const bytes = this.bytes();
        for(const item of list){
            view.setUint32(pointersPointer, bufferPointer, true);
            pointersPointer += 4;
            bytes.set(item, bufferPointer);
            bytes[bufferPointer + item.length] = 0;
            bufferPointer += item.length + 1;
        }
        return ERRNO.SUCCESS;
    }
    readInto(fd, iovsPointer, iovsLength, offset, resultPointer){
        const entry = this.fds[fd];
        if(entry === undefined){
            return ERRNO.BADF;
        }
        if(entry.type === "directory"){
            return ERRNO.ISDIR;
        }
        const view = this.view();
        if(entry.type === "stream"){
            if(entry.write !== null){
                return ERRNO.BADF;
            }
            view.setUint32(resultPointer, 0, true);
            return ERRNO.SUCCESS;
        }
        const bytes = this.bytes();
        const file = entry.file;
        let position = offset ?? entry.position;
        let total = 0;
        for(let i = 0; i < iovsLength; i++){
            const bufferPointer = view.getUint32(iovsPointer + 8 * i, true);
            const bufferLength = view.getUint32(iovsPointer + 8 * i + 4, true);
            const count = Math.min(bufferLength, Math.max(file.size - position, 0));
            bytes.set(file.bytes.subarray(position, position + count), bufferPointer);
            position += count;
            total += count;
            if(count < bufferLength){
                break;
            }
        }
        if(offset === null){
            entry.position = position;
        }
        view.setUint32(resultPointer, total, true);
        return ERRNO.SUCCESS;
    }
    writeFrom(fd, iovsPointer, iovsLength, offset, resultPointer){
        const entry = this.fds[fd];
        if(entry === undefined){
            return ERRNO.BADF;
        }
        if(entry.type === "directory"){
            return ERRNO.BADF;
        }
        const view = this.view();
        const bytes = this.bytes();
        let total = 0;
        if(entry.type === "stream"){
            if(entry.write === null){
                return ERRNO.BADF;
            }
            for(let i = 0; i < iovsLength; i++){
                const bufferPointer = view.getUint32(iovsPointer + 8 * i, true);
                const bufferLength = view.getUint32(iovsPointer + 8 * i + 4, true);
                entry.write(bytes.slice(bufferPointer, bufferPointer + bufferLength));
                total += bufferLength;
            }
        }
        else{
            const file = entry.file;
            if(file.readonly){
                return ERRNO.ROFS;
            }
            let position = offset ?? (entry.append ? file.size : entry.position);
            for(let i = 0; i < iovsLength; i++){
                const bufferPointer = view.getUint32(iovsPointer + 8 * i, true);
                const bufferLength = view.getUint32(iovsPointer + 8 * i + 4, true);
                file.write(position, bytes.subarray(bufferPointer, bufferPointer + bufferLength));
                position += bufferLength;
                total += bufferLength;
            }
            if(offset === null){
                entry.position = position;
            }
        }
        view.setUint32(resultPointer, total, true);
        return ERRNO.SUCCESS;
    }

    args_get(pointersPointer, bufferPointer){
        return this.writeStrings(this.args, pointersPointer, bufferPointer);
    }
    args_sizes_get(countPointer, sizePointer){
        return this.writeSizes(this.args, countPointer, sizePointer);
    }
    environ_get(pointersPointer, bufferPointer){
        return this.writeStrings(this.env, pointersPointer, bufferPointer);
    }
    environ_sizes_get(countPointer, sizePointer){
        return this.writeSizes(this.env, countPointer, sizePointer);
    }
    clock_res_get(clock, resultPointer){
        if(clock > CLOCKID.THREAD_CPUTIME_ID){
            return ERRNO.INVAL;
        }
        this.view().setBigUint64(resultPointer, clock === CLOCKID.REALTIME ? 1000000n : 1000n, true);
        return ERRNO.SUCCESS;
    }
    clock_time_get(clock, precision, resultPointer){
        if(clock > CLOCKID.THREAD_CPUTIME_ID){
            return ERRNO.INVAL;
        }
        this.view().setBigUint64(resultPointer, now(clock), true);
        return ERRNO.SUCCESS;
    }
    fd_advise(fd, offset, length, advice){
        return this.fds[fd] === undefined ? ERRNO.BADF : ERRNO.SUCCESS;
    }
    fd_allocate(fd, offset, length){
        const entry = this.fds[fd];
        if(entry === undefined || entry.type !== "file"){
            return ERRNO.BADF;
        }
        if(entry.file.readonly){
            return ERRNO.ROFS;
        }
        entry.file.resize(Math.max(entry.file.size, Number(offset + length)));
        return ERRNO.SUCCESS;
    }
    fd_close(fd){
        if(this.fds[fd] === undefined){
            return ERRNO.BADF;
        }
        this.fds[fd] = undefined;
        return ERRNO.SUCCESS;
    }
    fd_datasync(fd){
        return this.fds[fd] === undefined ? ERRNO.BADF : ERRNO.SUCCESS;
    }
    fd_fdstat_get(fd, resultPointer){
        const entry = this.fds[fd];
        if(entry === undefined){
            return ERRNO.BADF;
        }
        const view = this.view();
        this.bytes().fill(0, resultPointer, resultPointer + 24);
        view.setUint8(resultPointer, entry.type === "file" ? FILETYPE.REGULAR_FILE : entry.type === "directory" ? FILETYPE.DIRECTORY : FILETYPE.CHARACTER_DEVICE);
        view.setUint16(resultPointer + 2, entry.type === "file" && entry.append ? FDFLAGS.APPEND : 0, true);
        view.setBigUint64(resultPointer + 8, RIGHTS.ALL, true);
        view.setBigUint64(resultPointer + 16, RIGHTS.ALL, true);
        return ERRNO.SUCCESS;
    }
    fd_fdstat_set_flags(fd, flags){
        const entry = this.fds[fd];
        if(entry === undefined){
            return ERRNO.BADF;
        }
        if(entry.type === "file"){
            entry.append = (flags & FDFLAGS.APPEND) !== 0;
        }
        return ERRNO.SUCCESS;
    }
    fd_fdstat_set_rights(fd, rightsBase, rightsInheriting){
        return this.fds[fd] === undefined ? ERRNO.BADF : ERRNO.SUCCESS;
    }
    fd_filestat_get(fd, resultPointer){
        const entry = this.fds[fd];
        if(entry === undefined){
            return ERRNO.BADF;
        }
        this.writeFilestat(resultPointer, entry.type === "file" ? entry.file : entry.type === "directory" ? entry.directory : entry);
        return ERRNO.SUCCESS;
    }
    fd_filestat_set_size(fd, size){
        const entry = this.fds[fd];
        if(entry === undefined || entry.type !== "file"){
            return ERRNO.BADF;
        }
        if(entry.file.readonly){
            return ERRNO.ROFS;
        }
        entry.file.resize(Number(size));
        return ERRNO.SUCCESS;
    }
    fd_filestat_set_times(fd, atim, mtim, flags){
        return this.fds[fd] === undefined ? ERRNO.BADF : ERRNO.SUCCESS;
    }
    fd_pread(fd, iovsPointer, iovsLength, offset, resultPointer){
        return this.readInto(fd, iovsPointer, iovsLength, Number(offset), resultPointer);
    }
    fd_prestat_get(fd, resultPointer){
        const entry = this.fds[fd];
        if(entry === undefined || entry.type !== "directory" || entry.preopen === null){
            return ERRNO.BADF;
        }
        const view = this.view();
        view.setUint8(resultPointer, PREOPENTYPE_DIR);
        view.setUint32(resultPointer + 4, encoder.encode(entry.preopen).length, true);
        return ERRNO.SUCCESS;
    }
    fd_prestat_dir_name(fd, pathPointer, pathLength){
        const entry = this.fds[fd];
        if(entry === undefined || entry.type !== "directory" || entry.preopen === null){
            return ERRNO.BADF;
        }
        const name = encoder.encode(entry.preopen);
        if(name.length > pathLength){
            return ERRNO.NAMETOOLONG;
        }
        this.bytes().set(name, pathPointer);
        return ERRNO.SUCCESS;
    }
    fd_pwrite(fd, iovsPointer, iovsLength, offset, resultPointer){
        return this.writeFrom(fd, iovsPointer, iovsLength, Number(offset), resultPointer);
    }
    fd_read(fd, iovsPointer, iovsLength, resultPointer){
        return this.readInto(fd, iovsPointer, iovsLength, null, resultPointer);
    }
    // Entries are written until one no longer fits; that one is truncated and the buffer reported full, so the caller
    // retries with a larger buffer at the same cookie
    fd_readdir(fd, bufferPointer, bufferLength, cookie, resultPointer){
        const entry = this.fds[fd];
        if(entry === undefined){
            return ERRNO.BADF;
        }
        if(entry.type !== "directory"){
            return ERRNO.NOTDIR;
        }
        const directory = entry.directory;
        const entries = [[".", directory], ["..", directory], ...directory.contents];
        const view = this.view();
        const bytes = this.bytes();
        let used = 0;
        for(let index = Number(cookie); index < entries.length; index++){
            const [name, inode] = entries[index];
            const nameBytes = encoder.encode(name);
            const record = new Uint8Array(DIRENT_SIZE + nameBytes.length);
            const recordView = new DataView(record.buffer);
            recordView.setBigUint64(0, BigInt(index + 1), true);
            recordView.setBigUint64(8, BigInt(inode.ino), true);
            recordView.setUint32(16, nameBytes.length, true);
            recordView.setUint8(20, filetypeOf(inode));
            record.set(nameBytes, DIRENT_SIZE);
            const count = Math.min(record.length, bufferLength - used);
            bytes.set(record.subarray(0, count), bufferPointer + used);
            used += count;
            if(count < record.length){
                break;
            }
        }
        view.setUint32(resultPointer, used, true);
        return ERRNO.SUCCESS;
    }
    fd_renumber(fd, to){
        if(this.fds[fd] === undefined){
            return ERRNO.BADF;
        }
        this.fds[to] = this.fds[fd];
        this.fds[fd] = undefined;
        return ERRNO.SUCCESS;
    }
    fd_seek(fd, offset, whence, resultPointer){
        const entry = this.fds[fd];
        if(entry === undefined){
            return ERRNO.BADF;
        }
        if(entry.type !== "file"){
            return entry.type === "directory" ? ERRNO.BADF : ERRNO.SPIPE;
        }
        const delta = Number(offset);
        let position;
        switch(whence){
            case WHENCE.SET: position = delta; break;
            case WHENCE.CUR: position = entry.position + delta; break;
            case WHENCE.END: position = entry.file.size + delta; break;
            default: return ERRNO.INVAL;
        }
        if(position < 0){
            return ERRNO.INVAL;
        }
        entry.position = position;
        this.view().setBigUint64(resultPointer, BigInt(position), true);
        return ERRNO.SUCCESS;
    }
    fd_sync(fd){
        return this.fds[fd] === undefined ? ERRNO.BADF : ERRNO.SUCCESS;
    }
    fd_tell(fd, resultPointer){
        const entry = this.fds[fd];
        if(entry === undefined){
            return ERRNO.BADF;
        }
        if(entry.type !== "file"){
            return entry.type === "directory" ? ERRNO.BADF : ERRNO.SPIPE;
        }
        this.view().setBigUint64(resultPointer, BigInt(entry.position), true);
        return ERRNO.SUCCESS;
    }
    fd_write(fd, iovsPointer, iovsLength, resultPointer){
        return this.writeFrom(fd, iovsPointer, iovsLength, null, resultPointer);
    }
    path_create_directory(fd, pathPointer, pathLength){
        const resolved = this.lookup(fd, pathPointer, pathLength);
        if(resolved.error !== 0){
            return resolved.error;
        }
        if(resolved.inode !== null){
            return ERRNO.EXIST;
        }
        if(resolved.parent.readonly){
            return ERRNO.ROFS;
        }
        resolved.parent.contents.set(resolved.name, new Directory());
        return ERRNO.SUCCESS;
    }
    path_filestat_get(fd, flags, pathPointer, pathLength, resultPointer){
        const resolved = this.lookup(fd, pathPointer, pathLength);
        if(resolved.error !== 0){
            return resolved.error;
        }
        if(resolved.inode === null){
            return ERRNO.NOENT;
        }
        this.writeFilestat(resultPointer, resolved.inode);
        return ERRNO.SUCCESS;
    }
    path_filestat_set_times(fd, flags, pathPointer, pathLength, atim, mtim, fstFlags){
        const resolved = this.lookup(fd, pathPointer, pathLength);
        if(resolved.error !== 0){
            return resolved.error;
        }
        return resolved.inode === null ? ERRNO.NOENT : ERRNO.SUCCESS;
    }
    path_link(oldFd, oldFlags, oldPathPointer, oldPathLength, newFd, newPathPointer, newPathLength){
        return ERRNO.NOTSUP;
    }
    path_open(fd, dirflags, pathPointer, pathLength, oflags, rightsBase, rightsInheriting, fdflags, resultPointer){
        const resolved = this.lookup(fd, pathPointer, pathLength);
        if(resolved.error !== 0){
            return resolved.error;
        }
        let inode = resolved.inode;
        if(inode === null){
            if((oflags & OFLAGS.CREAT) === 0 || (oflags & OFLAGS.DIRECTORY) !== 0){
                return ERRNO.NOENT;
            }
            if(resolved.parent.readonly){
                return ERRNO.ROFS;
            }
            inode = new File();
            resolved.parent.contents.set(resolved.name, inode);
        }
        else if((oflags & OFLAGS.CREAT) !== 0 && (oflags & OFLAGS.EXCL) !== 0){
            return ERRNO.EXIST;
        }
        if(inode instanceof Directory){
            return this.allocate({ type: "directory", directory: inode, preopen: null }, resultPointer);
        }
        if((oflags & OFLAGS.DIRECTORY) !== 0){
            return ERRNO.NOTDIR;
        }
        const writing = (rightsBase & RIGHTS.FD_WRITE) !== 0n || (oflags & OFLAGS.TRUNC) !== 0;
        if(writing && inode.readonly){
            return ERRNO.ROFS;
        }
        if((oflags & OFLAGS.TRUNC) !== 0){
            inode.resize(0);
        }
        const append = (fdflags & FDFLAGS.APPEND) !== 0;
        return this.allocate({ type: "file", file: inode, position: append ? inode.size : 0, append }, resultPointer);
    }
    path_readlink(fd, pathPointer, pathLength, bufferPointer, bufferLength, resultPointer){
        const resolved = this.lookup(fd, pathPointer, pathLength);
        if(resolved.error !== 0){
            return resolved.error;
        }
        return resolved.inode === null ? ERRNO.NOENT : ERRNO.INVAL;
    }
    path_remove_directory(fd, pathPointer, pathLength){
        const resolved = this.lookup(fd, pathPointer, pathLength);
        if(resolved.error !== 0){
            return resolved.error;
        }
        if(resolved.inode === null){
            return ERRNO.NOENT;
        }
        if(!(resolved.inode instanceof Directory)){
            return ERRNO.NOTDIR;
        }
        if(resolved.name === null){
            return ERRNO.INVAL;
        }
        if(resolved.inode.contents.size > 0){
            return ERRNO.NOTEMPTY;
        }
        if(resolved.parent.readonly){
            return ERRNO.ROFS;
        }
        resolved.parent.contents.delete(resolved.name);
        return ERRNO.SUCCESS;
    }
    path_rename(fd, oldPathPointer, oldPathLength, newFd, newPathPointer, newPathLength){
        const source = this.lookup(fd, oldPathPointer, oldPathLength);
        if(source.error !== 0){
            return source.error;
        }
        const target = this.lookup(newFd, newPathPointer, newPathLength);
        if(target.error !== 0){
            return target.error;
        }
        if(source.inode === null){
            return ERRNO.NOENT;
        }
        if(source.name === null || target.name === null){
            return ERRNO.INVAL;
        }
        if(source.parent.readonly || target.parent.readonly){
            return ERRNO.ROFS;
        }
        if(target.inode !== null && target.inode !== source.inode){
            const sourceIsDirectory = source.inode instanceof Directory;
            const targetIsDirectory = target.inode instanceof Directory;
            if(sourceIsDirectory && !targetIsDirectory){
                return ERRNO.NOTDIR;
            }
            if(!sourceIsDirectory && targetIsDirectory){
                return ERRNO.ISDIR;
            }
            if(targetIsDirectory && target.inode.contents.size > 0){
                return ERRNO.NOTEMPTY;
            }
        }
        source.parent.contents.delete(source.name);
        target.parent.contents.set(target.name, source.inode);
        return ERRNO.SUCCESS;
    }
    path_symlink(oldPathPointer, oldPathLength, fd, newPathPointer, newPathLength){
        return ERRNO.NOTSUP;
    }
    path_unlink_file(fd, pathPointer, pathLength){
        const resolved = this.lookup(fd, pathPointer, pathLength);
        if(resolved.error !== 0){
            return resolved.error;
        }
        if(resolved.inode === null){
            return ERRNO.NOENT;
        }
        if(resolved.inode instanceof Directory){
            return ERRNO.ISDIR;
        }
        if(resolved.parent.readonly){
            return ERRNO.ROFS;
        }
        resolved.parent.contents.delete(resolved.name);
        return ERRNO.SUCCESS;
    }
    // Nothing can be waited on in a single-threaded process: clock subscriptions sleep, descriptors are always ready
    poll_oneoff(subscriptionsPointer, eventsPointer, count, resultPointer){
        const view = this.view();
        const bytes = this.bytes();
        for(let i = 0; i < count; i++){
            const subscription = subscriptionsPointer + SUBSCRIPTION_SIZE * i;
            const userdata = view.getBigUint64(subscription, true);
            const type = view.getUint8(subscription + 8);
            let error = ERRNO.SUCCESS;
            let available = 0n;
            if(type === EVENTTYPE.CLOCK){
                const clock = view.getUint32(subscription + 16, true);
                const timeout = view.getBigUint64(subscription + 24, true);
                const flags = view.getUint16(subscription + 40, true);
                if(clock > CLOCKID.THREAD_CPUTIME_ID){
                    error = ERRNO.INVAL;
                }
                else{
                    sleepUntil((flags & SUBCLOCKFLAGS_ABSTIME) !== 0 ? timeout : now(clock) + timeout, clock);
                }
            }
            else if(type === EVENTTYPE.FD_READ || type === EVENTTYPE.FD_WRITE){
                const entry = this.fds[view.getUint32(subscription + 16, true)];
                if(entry === undefined){
                    error = ERRNO.BADF;
                }
                else if(entry.type === "file"){
                    available = BigInt(entry.file.size);
                }
            }
            else{
                error = ERRNO.INVAL;
            }
            const event = eventsPointer + EVENT_SIZE * i;
            bytes.fill(0, event, event + EVENT_SIZE);
            view.setBigUint64(event, userdata, true);
            view.setUint16(event + 8, error, true);
            view.setUint8(event + 10, type);
            view.setBigUint64(event + 16, available, true);
        }
        view.setUint32(resultPointer, count, true);
        return ERRNO.SUCCESS;
    }
    proc_exit(code){
        throw new ProcessExit(code);
    }
    proc_raise(signal){
        return ERRNO.NOTSUP;
    }
    sched_yield(){
        return ERRNO.SUCCESS;
    }
    random_get(bufferPointer, length){
        const bytes = this.bytes().subarray(bufferPointer, bufferPointer + length);
        for(let offset = 0; offset < length; offset += 65536){
            crypto.getRandomValues(bytes.subarray(offset, offset + 65536));
        }
        return ERRNO.SUCCESS;
    }
    sock_accept(fd, flags, resultPointer){
        return ERRNO.NOTSUP;
    }
    sock_recv(fd, iovsPointer, iovsLength, flags, resultPointer, flagsPointer){
        return ERRNO.NOTSUP;
    }
    sock_send(fd, iovsPointer, iovsLength, flags, resultPointer){
        return ERRNO.NOTSUP;
    }
    sock_shutdown(fd, how){
        return ERRNO.NOTSUP;
    }
}
