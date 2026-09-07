# Browser IDE toolchain (from source)

Builds the compiler the browser IDE (`static/ide`) runs: clang, lld and llvm-ar for `wasm32-wasip1` as one WASI module (`llvm.wasm`), and the sysroot both the module and the programs it compiles link against (`sysroot.tar`). Nothing prebuilt enters the pipeline: the sources are RLtools' fork of llvm-project at the branch that carries the WASI commit, and stock wasi-libc, both fetched by CMake at depth 1; the cross compiler is the distro's clang and lld.

```
sudo apt install clang-22 lld-22 ninja-build nodejs         # Ubuntu 26.04; ninja and node are optional
cmake -S tools/ide/toolchain -B /vm/data/rl-tools/ide-toolchain/build -G Ninja   # fetches the two checkouts into .dependencies
cmake --build /vm/data/rl-tools/ide-toolchain/build            # about 40 min at 5 jobs after the fetch; --target sysroot stops after stage 2
```

## Sources

The pins are the literals at the top of `CMakeLists.txt`: `RL_TOOLS_IDE_LLVM_REPOSITORY` and `RL_TOOLS_IDE_LLVM_BRANCH` (the fork and its WASI branch), `RL_TOOLS_IDE_LLVM_UPSTREAM` (the tag the branch sits on), `RL_TOOLS_IDE_WASI_LIBC_REPOSITORY` and `RL_TOOLS_IDE_WASI_LIBC_TAG`, and `RL_TOOLS_IDE_HOST_LLVM` (the distro clang's major version). `tests/src/ide/provenance.sh` checks the served package against them; the package's `toolchain.json` and the compiler's `--version` name the exact commit.

The sources are FetchContent checkouts, handled like every other RLtools dependency: they land in `<repository>/.dependencies/<build directory name>/` (`llvm_project-src`, `wasi_libc-src`) and survive a deleted build directory. `cmake/fetch_source.cmake` fetches exactly one ref at depth 1 and no tags (CMake's own `GIT_SHALLOW` would clone every branch tip), so the llvm-project transfer is about 280 MB. llvm-project follows its branch: a reconfigure with network access moves the checkout to the branch tip and the stages rebuild what changed; without network the checkout is kept with a warning; `-DRL_TOOLS_OFFLINE_BUILD=ON` never asks. `-DFETCHCONTENT_SOURCE_DIR_LLVM_PROJECT=<dir>` (and `..._WASI_LIBC`) points at an existing checkout instead, for example a local mirror; `-DFETCHCONTENT_BASE_DIR=<dir>` moves the tree (a local disk is much faster than NFS for the 150,000 files of llvm-project).

The llvm-project checkout is an ordinary git repository, so a change to the fork is made right there: commit on top, `git push origin HEAD:rltools-wasi-22.1.8`, reconfigure. The checkout holds only the branch tip (depth 1); `git fetch --deepen=<n> origin` adds history when a rebase needs it.

## The LLVM fork

Branch `rltools-wasi-22.1.8` of `rl-tools/llvm-project` is `llvmorg-22.1.8` plus one commit that makes clang, lld and llvm-ar build for `wasm32-wasip1` (17 files, +190/−21); `git diff llvmorg-22.1.8..rltools-wasi-22.1.8` is the whole review surface (the fork carries only the branch, so fetch the tag from upstream: `git fetch https://github.com/llvm/llvm-project.git tag llvmorg-22.1.8`). Nothing in it touches code generation. Its parts, one per subsystem:

| Change | Files | What is missing on WASI | Mechanism |
|---|---|---|---|
| Fix the spelling of `__wasm__` | `llvm/.../Support/Compiler.h`, `clang/.../Support/Compiler.h` | (a typo: `__WASM__` is never defined) | plain fix, upstream candidate |
| Recognize WASI hosts, test for setjmp and sys/resource.h | `HandleLLVMOptions.cmake`, `config-ix.cmake`, `config.h.cmake` | version scripts; setjmp and rlimits are optional | `elseif(WASI)` sets `LLVM_ON_UNIX`; `check_symbol_exists(setjmp)`, `check_include_file(sys/resource.h)` |
| Test for sys/resource.h before using resource limits | `ProgramStack.cpp`, `Unix/Process.inc`, `Unix/Program.inc` | `getrlimit`, `setrlimit` | `HAVE_SYS_RESOURCE_H` |
| Build without signals | `Signals.cpp`, `CrashRecoveryContext.cpp`, `Unix/Watchdog.inc`, `Unix/Process.inc`, `LockFileManager.cpp` | `sigaction`, `sigprocmask`, `sigaltstack`, `alarm`, `kill`, `setjmp` | a WASI implementation of the signal entry points that keeps the file-removal and handler bookkeeping; `HAVE_SETJMP` |
| Build without subprocesses | `Unix/Unix.h`, `Unix/Program.inc` | `fork`, `exec`, `wait4`, `sys/wait.h` | `Execute`/`Wait` report the error for `__wasi__` |
| Build the Unix path and process implementations | `ADT/bit.h`, `Unix/Path.inc`, `Unix/Process.inc` | `pwd.h`, `umask`, `madvise`, `fchown`, `TIOCGWINSZ`; has `endian.h`, argv[0] only, no remote mounts | `__wasi__` in the existing platform lists; `defined(TIOCGWINSZ)` |
| Build without Unix domain sockets | `raw_socket_stream.cpp` | `socket`, `bind`, `listen`, `connect` | the file is compiled out for `__wasi__`, like `zOSLibFunctions.cpp` elsewhere |
| Provide `__cxa_thread_atexit` (libc++abi) | `libcxxabi/src/cxa_thread_atexit.cpp` | defined for Linux and Fuchsia only; `thread_local` objects with destructors need it | `__wasi__` joins the list (LLVM 23's libc++abi already has it) |

The rules the change follows, for the next rebase:

1. **Use wasi-libc before patching.** wasi-libc emulates `mmap`, `getpid` and `signal()`/`raise()` behind `_WASI_EMULATED_MMAN`, `_WASI_EMULATED_GETPID` and `_WASI_EMULATED_SIGNAL` (`cmake/Toolchain-WASI.cmake` enables them); every site those cover compiles unchanged. Process clocks are deliberately not enabled: their header would also declare the rlimit functions wasi-libc does not implement.
2. **Feature checks for what depends on how wasi-libc was built** (`HAVE_SETJMP`, `HAVE_SYS_RESOURCE_H`); the code tests those, never the platform.
3. **Platform guards for what WASI lacks by design**: `defined(__wasi__)` added to the platform list the file already keeps for the same gap, or a small `#if defined(__wasi__)` alternative next to the real implementation.
4. **No behaviour change anywhere else**: every hunk is conditional.
5. **Scope is what the module needs**: the libraries linked into `llvm-driver` with clang, lld and llvm-ar, plus the one libc++abi function that link needs; not a port of every tool.

**Moving to another LLVM version:** in a clone of the fork with both tags fetched from upstream (`git fetch https://github.com/llvm/llvm-project.git tag llvmorg-22.1.8 tag llvmorg-<new>`), `git rebase --onto llvmorg-<new> llvmorg-22.1.8 rltools-wasi-22.1.8`, resolve conflicts (first ask whether upstream or wasi-libc removed the need for a hunk), push the branch under a new name, and set `RL_TOOLS_IDE_LLVM_BRANCH` and `RL_TOOLS_IDE_LLVM_UPSTREAM` in `CMakeLists.txt`; the verify stage then proves the module.

## Stages

| Stage | Target | What it does |
|---|---|---|
| 0 | (configure) | FetchContent clones llvm-project (the fork's branch) and wasi-libc at configure time, one ref at depth 1 (`cmake/fetch_source.cmake`) |
| 1 | `host` | Native `llvm-tblgen`, `llvm-min-tblgen`, `clang-tblgen` of the pinned version, compiled by the distro clang; the cross build cannot run the ones it builds |
| 2 | `sysroot` | compiler-rt builtins, wasi-libc, then libc++abi and libc++ into `prefix/usr`, compiled by the distro clang through `cmake/Toolchain-WASI.cmake`; `resource/` holds the host clang's headers next to the wasm32 builtins |
| 3 | `llvm-wasm` | clang + lld + llvm-ar as `llvm-driver`, MinSizeRel, ThinLTO, threads off, wasi-libc's mman/getpid/signal emulation, 8 MiB stack first in memory, 4 GiB memory ceiling, `DEFAULT_SYSROOT=/usr`, `CLANG_RESOURCE_DIR=/usr` (the browser mounts the sysroot at `/usr`), built with `SOURCE_DATE_EPOCH` = commit time |
| 4 | `package` | `out/llvm.wasm`, deterministic ustar `out/sysroot.tar` (sysroot + the stage 3 resource headers merged into `include/`), `toolchain.json`, `SHA256SUMS`; copied to `static/ide/build/toolchain` |
| 5 | `verify` | `tools/ide/bundle.sh`, parity build of the pendulum program with the distro clang against the same sysroot, `tests/src/ide/toolchain.mjs` (imports, exports, memory ceiling, identity, stack probe), `tests/src/ide/pipeline.mjs --reference`, `tests/src/ide/provenance.sh` |

## Layout on disk

```
<repository>/.dependencies/<build directory name>/
  llvm_project-src                    the fork's branch tip (depth 1)
  wasi_libc-src
$RL_TOOLS_IDE_TOOLCHAIN_DIR/          default /vm/data/rl-tools/ide-toolchain, override with the env var or -DRL_TOOLS_IDE_TOOLCHAIN_DIR
  build/                              this superbuild's own tree (the -B directory)
  host/                               stage 1
  sysroot/{compiler-rt,wasi-libc,libcxx}   stage 2 build trees
  prefix/usr                          the sysroot
  resource/                           host clang headers + wasm32 builtins
  wasm/                               stage 3 (bin/llvm, usr/include, SOURCE_DATE_EPOCH)
  out/                                stage 4 package + VERIFIED stamp
  parity/training.wasm                stage 5 reference program
```

## Options

| Cache variable | Default | Meaning |
|---|---|---|
| `RL_TOOLS_IDE_TOOLCHAIN_DIR` | env or `/vm/data/rl-tools/ide-toolchain` | build trees and outputs (sources live in `.dependencies`) |
| `RL_TOOLS_OFFLINE_BUILD` | OFF | never touch the network; the checkouts must already be present |
| `RL_TOOLS_IDE_JOBS` | 5 | parallel jobs per sub-build, also the ThinLTO backend threads of the module link |
| `RL_TOOLS_IDE_LTO` | `thin` | `full` gives the smallest module but its single-threaded link peaked at 16.4 GB on 22.1.8 and thrashes a 15 GB machine; `off` links fastest with the largest module |
| `RL_TOOLS_IDE_ASSERTIONS` | ON | LLVM assertions in the module, as in the deployed one |
| `RL_TOOLS_IDE_HOST_LLVM_BIN` | `/usr/lib/llvm-22/bin` | directory with `clang`, `clang++`, `wasm-ld`, `llvm-ar`, `llvm-ranlib`, `llvm-nm` |
| `RL_TOOLS_IDE_BUILD_HOST_CLANG` | OFF | build the cross compiler from the pinned source with the system compiler (stage 1 grows by about an hour); for machines without a suitable distro clang |
| `RL_TOOLS_IDE_OUTPUT_DIR` | `static/ide/build/toolchain` | where the package is copied |

## Measured on the reference VM (8 cores, 15 GB, NFS-backed disk, LLVM 22.1.8)

| Step | Time |
|---|---|
| fetch (once; the 150,000-file checkout on NFS dominates, the transfer is 280 MB) | 25 min |
| stage 1 tablegen | 2.5 min |
| stage 2 sysroot | 6 min |
| stage 3 compile (2956 objects) | 21 min at 5 jobs, 35 min at 3 |
| stage 3 ThinLTO link | a few minutes |
| package and verify | 3 min |

The result: `llvm.wasm` 75.0 MB (clang, lld, llvm-ar; 30 WASI imports), `sysroot.tar` 25.3 MB; the module compiles a template recursion 466 deep under Node's default stack.

## Reproducibility

Same pins, same host clang version, same flags give the same bytes: absolute paths are mapped away with `-ffile-prefix-map`, the module is stripped, the archive carries fixed ownership and the commit time as mtime. Checked on the reference VM: two stage 3 builds from empty trees (Ubuntu clang 22.1.2, ThinLTO, 3 jobs) produced the identical `llvm.wasm`. `toolchain.json` records the host compiler and linker versions so a byte difference between two hosts has a visible cause. Check a second build with `sha256sum -c static/ide/build/toolchain/SHA256SUMS`.
