# Browser IDE toolchain (from source)

Builds the compiler the browser IDE (`static/ide`) runs: clang, lld and llvm-ar for `wasm32-wasip1` as one WASI module (`llvm.wasm`), and the sysroot both the module and the programs it compiles link against (`sysroot.tar`). Nothing prebuilt enters the pipeline: the sources are git tags fetched by CMake at depth 1, the cross compiler is the distro's clang and lld, and the only change to upstream is our own small patch series in `patches/`.

```
sudo apt install clang-22 lld-22 ninja-build nodejs         # Ubuntu 26.04; ninja and node are optional
cmake -S tools/ide/toolchain -B /vm/data/rl-tools/ide-toolchain/build -G Ninja   # fetches the two pinned checkouts (about 1.5 GB)
cmake --build /vm/data/rl-tools/ide-toolchain/build            # about 40 min at 5 jobs after the fetch; --target sysroot stops after stage 2
```

The pins are the literals at the top of `CMakeLists.txt` (`RL_TOOLS_IDE_LLVM_TAG`, `RL_TOOLS_IDE_WASI_LIBC_TAG`, `RL_TOOLS_IDE_HOST_LLVM`); a 40-character commit works in place of a tag for content-addressed pinning. `tests/src/ide/provenance.sh` checks the served package against them.

The sources are FetchContent checkouts, handled exactly like every other RLtools dependency: they land in `<repository>/.dependencies/<build directory name>/` (`llvm_project-src`, `wasi_libc-src`), are shared with a main RLtools build directory of the same name, and survive a deleted build directory. `-DRL_TOOLS_OFFLINE_BUILD=ON` reuses them without touching the network; `-DFETCHCONTENT_SOURCE_DIR_LLVM_PROJECT=<dir>` (and `..._WASI_LIBC`) points at an existing checkout instead, for example a local mirror; `-DFETCHCONTENT_BASE_DIR=<dir>` moves the tree.

## Stages

| Stage | Target | What it does |
|---|---|---|
| 0 | (configure), `llvm-wasi-source` | FetchContent clones llvm-project and wasi-libc at configure time (one ref, depth 1, no other tags; `cmake/fetch_source.cmake`); `llvm-wasi-source` keeps a detached worktree `llvm_project-wasi-src` next to them with `patches/*.patch` applied in order (`cmake/prepare_wasi_source.cmake`, idempotent, resets when the series or the pin changes) |
| 1 | `host` | Native `llvm-tblgen`, `llvm-min-tblgen`, `clang-tblgen` of the pinned version, compiled by the distro clang; the cross build cannot run the ones it builds |
| 2 | `sysroot` | compiler-rt builtins and wasi-libc from the pristine checkout, then libc++abi and libc++ from the patched worktree (patch 0008) into `prefix/usr`, compiled by the distro clang through `cmake/Toolchain-WASI.cmake`; `resource/` holds the host clang's headers next to the wasm32 builtins |
| 3 | `llvm-wasm` | clang + lld + llvm-ar as `llvm-driver` from the patched worktree, MinSizeRel, LTO, threads off, wasi-libc's mman/getpid/signal emulation, 8 MiB stack first in memory, 4 GiB memory ceiling, `DEFAULT_SYSROOT=/usr`, `CLANG_RESOURCE_DIR=/usr` (the browser mounts the sysroot at `/usr`), built with `SOURCE_DATE_EPOCH` = commit time |
| 4 | `package` | `out/llvm.wasm`, deterministic ustar `out/sysroot.tar` (sysroot + the stage 3 resource headers merged into `include/`), `toolchain.json`, `SHA256SUMS`; copied to `static/ide/build/toolchain` |
| 5 | `verify` | `tools/ide/bundle.sh`, parity build of the pendulum program with the distro clang against the same sysroot, `tests/src/ide/toolchain.mjs` (imports, exports, memory ceiling, identity, stack probe), `tests/src/ide/pipeline.mjs --reference`, `tests/src/ide/provenance.sh` |

## The patch series

`patches/*.patch` is our own series (eight patches, 17 files, about +190/−20 lines), maintained as commits and exported with `git format-patch`; `patches/README.md` explains each patch, the rules the series follows (wasi-libc's emulation libraries before code changes, feature checks for what depends on wasi-libc's configuration, `__wasi__` guards in the files' existing platform lists for what WASI lacks by design), and how to rebase it onto another LLVM version with `rebase_patches.sh`. Every hunk is conditional, so no other platform changes. `toolchain.json` records each patch's SHA-256; a modified series fails the provenance test.

## Layout on disk

```
<repository>/.dependencies/<build directory name>/
  llvm_project-src                    pristine checkout (stage 1)
  llvm_project-wasi-src               worktree + patch (stages 2 and 3)
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
| `RL_TOOLS_IDE_JOBS` | 5 | parallel jobs per sub-build |
| `RL_TOOLS_IDE_LTO` | `thin` | `full` gives the smallest module but its single-threaded link peaked at 16.4 GB on 22.1.8 and thrashes a 15 GB machine; `off` links fastest with the largest module |
| `RL_TOOLS_IDE_ASSERTIONS` | ON | LLVM assertions in the module, as in the deployed one |
| `RL_TOOLS_IDE_HOST_LLVM_BIN` | `/usr/lib/llvm-22/bin` | directory with `clang`, `clang++`, `wasm-ld`, `llvm-ar`, `llvm-ranlib`, `llvm-nm` |
| `RL_TOOLS_IDE_BUILD_HOST_CLANG` | OFF | build the cross compiler from the pinned source with the system compiler (stage 1 grows by about an hour); for machines without a suitable distro clang |
| `RL_TOOLS_IDE_OUTPUT_DIR` | `static/ide/build/toolchain` | where the package is copied |

## Measured on the reference VM (8 cores, 15 GB, NFS-backed disk, LLVM 22.1.8)

| Step | Time |
|---|---|
| fetch (once; the 150,000-file checkout on NFS dominates, the transfer is 278 MB) | 25 min |
| stage 1 tablegen | 2.5 min |
| stage 2 sysroot | 6 min |
| stage 3 compile (2956 objects) | 21 min at 5 jobs, 35 min at 3 |
| stage 3 ThinLTO link | a few minutes |
| package and verify | 3 min |

The result: `llvm.wasm` 75.0 MB (clang, lld, llvm-ar; 30 WASI imports), `sysroot.tar` 25.3 MB; the module compiles a template recursion 466 deep under Node's default stack.

## Reproducibility

Same pins, same host clang version, same flags give the same bytes: absolute paths are mapped away with `-ffile-prefix-map`, the module is stripped, the archive carries fixed ownership and the commit time as mtime. `toolchain.json` records the host compiler and linker versions so a byte difference between two hosts has a visible cause. Check a second build with `sha256sum -c static/ide/build/toolchain/SHA256SUMS`.
