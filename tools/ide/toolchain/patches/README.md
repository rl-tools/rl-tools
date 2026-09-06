# WASI host patches for LLVM

The series makes clang, lld and llvm-ar build for `wasm32-wasip1` from upstream `llvm-project` at the pin in `../CMakeLists.txt` (`RL_TOOLS_IDE_LLVM_TAG`). The superbuild applies the patches in name order onto a detached worktree of the pristine checkout (`cmake/prepare_wasi_source.cmake`); nothing in the series touches code generation. compiler-rt and wasi-libc build from the pristine tree; libc++abi needs the one-line patch 0008, so the runtimes build from the worktree.

## Rules the series follows

1. **Use wasi-libc before patching.** wasi-libc emulates `mmap`, `getpid` and `signal()`/`raise()` behind `_WASI_EMULATED_MMAN`, `_WASI_EMULATED_GETPID` and `_WASI_EMULATED_SIGNAL` (`cmake/Toolchain-WASI.cmake` enables them). Every site those cover compiles unchanged, which is why `<signal.h>` includes, `getpid()` and `mprotect()` need no hunks. Process clocks are deliberately not enabled: their header would also declare the rlimit functions wasi-libc does not implement.
2. **Feature checks for what depends on how wasi-libc was built.** `setjmp` exists only in an exception-handling build of wasi-libc and `sys/resource.h` only as an emulation, so both are configure-time checks (`HAVE_SETJMP`, `HAVE_SYS_RESOURCE_H`) and the code tests those, never the platform.
3. **Platform guards for what WASI lacks by design.** Signals, subprocesses, the password database, `umask`, `madvise`, file ownership: `defined(__wasi__)` is added to the platform list the file already keeps for the same gap (`__EMSCRIPTEN__`, `__HAIKU__`, `__Fuchsia__`, `__MVS__`, `_AIX`), or a small `#if defined(__wasi__)` alternative is placed next to the real implementation.
4. **No behaviour change anywhere else.** Every hunk is conditional; on every other platform the preprocessed source is identical.
5. **Scope is what the module needs.** The series covers the libraries linked into `llvm-driver` with clang, lld and llvm-ar, plus the one libc++abi function that link needs. It is not a full port of every tool (the interpreter, the module build daemon and the like are out of scope) and every claim is checked by building the module.

## The patches

| Patch | Files | What is missing on WASI | Mechanism |
|---|---|---|---|
| 0001 Fix the spelling of `__wasm__` | `llvm/.../Support/Compiler.h`, `clang/.../Support/Compiler.h` | (a typo: `__WASM__` is never defined) | plain fix, upstream candidate |
| 0002 Recognize WASI hosts, test for setjmp and sys/resource.h | `HandleLLVMOptions.cmake`, `config-ix.cmake`, `config.h.cmake` | version scripts; setjmp and rlimits are optional | `elseif(WASI)` sets `LLVM_ON_UNIX`; `check_symbol_exists(setjmp)`, `check_include_file(sys/resource.h)` |
| 0003 Test for sys/resource.h before using resource limits | `ProgramStack.cpp`, `Unix/Process.inc`, `Unix/Program.inc` | `getrlimit`, `setrlimit` | `HAVE_SYS_RESOURCE_H` |
| 0004 Build without signals | `Signals.cpp`, `CrashRecoveryContext.cpp`, `Unix/Watchdog.inc`, `Unix/Process.inc`, `LockFileManager.cpp` | `sigaction`, `sigprocmask`, `sigaltstack`, `alarm`, `kill`, `setjmp` | no-op signal entry points for `__wasi__`; `HAVE_SETJMP` |
| 0005 Build without subprocesses | `Unix/Unix.h`, `Unix/Program.inc` | `fork`, `exec`, `wait4`, `sys/wait.h` | `Execute`/`Wait` report the error for `__wasi__` |
| 0006 Build the Unix path and process implementations | `ADT/bit.h`, `Unix/Path.inc`, `Unix/Process.inc` | `pwd.h`, `umask`, `madvise`, `fchown`, `TIOCGWINSZ`; has `endian.h`, argv[0] only, no remote mounts | `__wasi__` in the existing platform lists; `defined(TIOCGWINSZ)` |
| 0007 Build without Unix domain sockets | `raw_socket_stream.cpp` | `socket`, `bind`, `listen`, `connect`, `sockaddr_un::sun_path` | the file is compiled out for `__wasi__`, like `zOSLibFunctions.cpp` is elsewhere |
| 0008 Provide `__cxa_thread_atexit` (libc++abi) | `libcxxabi/src/cxa_thread_atexit.cpp` | the function was defined for Linux and Fuchsia only; `thread_local` objects with destructors need it | `__wasi__` joins the platform list (LLVM 23's libc++abi already defines it for WASI) |

The commit message of each patch carries the rationale; `git am` keeps it. `toolchain.json` records the SHA-256 of every patch and `tests/src/ide/provenance.sh` compares them with the checked-in files.

## Rebasing onto another LLVM version

The series is maintained as commits, not as edited diff text:

```
tools/ide/toolchain/rebase_patches.sh llvmorg-23.1.0        # fetches the tag at depth 1, applies the series as commits, rebases, re-exports
```

The helper creates a worktree (`$RL_TOOLS_IDE_TOOLCHAIN_DIR/patchwork`, or `RL_TOOLS_IDE_PATCHWORK_DIR`; a local disk is much faster than NFS for the 150,000-file checkout), runs `git am --3way` with the current series on the current pin, then `git rebase --onto` the new pin. A conflict stops there for manual resolution (`git rebase --continue` in the worktree), after which `rebase_patches.sh --export llvmorg-23.1.0` writes the series back with `git format-patch --zero-commit --no-signature`. Then change `RL_TOOLS_IDE_LLVM_TAG`, rebuild, and let the verify stage confirm the module.

When a hunk no longer applies, first check whether upstream removed the need: a call site that gained a feature check, or wasi-libc gaining the function. Drop hunks before adapting them.

## Origin

The set of sites comes from building the module and from YoWASP's port of LLVM 21/22 to WASI by whitequark ("Conditionalize use of POSIX features missing on WASI/WebAssembly"), which guarded 23 files with `__wasi__`. This series replaces it: its `getpid`, `dup2`, `mprotect`, `<signal.h>` and interpreter hunks are unnecessary once wasi-libc's emulation libraries are linked, its libc++ hunk is unnecessary with `LIBCXX_HAS_MUSL_LIBC=OFF` (libc++ 21 and later know WASI), its jobserver, lld and lock-file hunks turned out not to be needed at all, its resource-limit guards became the `HAVE_SYS_RESOURCE_H` check, and the rest is restated as small per-subsystem patches with commit messages. The series was verified by compiling Support, lld and the clang driver libraries against llvmorg-22.1.8 with zero errors and no warnings of its own, and then by building and verifying the module.
