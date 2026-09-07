# Browser IDE toolchain

This standalone CMake project builds native tablegen tools, a WASI sysroot, and clang + lld as one `llvm.wasm` command module. It packages a candidate bundle, verifies that bundle through CTest, and stages the verified files for [the IDE](../../../static/ide/README.md).

Prerequisites are CMake 3.26+, GNU tar, git, Python 3, Node.js 22+, and native clang/lld 22 with `clang`, `clang++`, `wasm-ld`, `llvm-ar`, `llvm-ranlib`, and `llvm-nm`. Ninja is optional; every sub-build inherits the selected CMake generator. An installed Chrome/Chromium or Firefox enables the browser check. Python scripts use the repository's `.venv` and need no packages.

```bash
# Provision these tools yourself; on a suitable Ubuntu release:
sudo apt install cmake clang-22 lld-22 llvm-22 nodejs
node --version                       # must be 22 or newer
python3 -m venv .venv                 # only if .venv does not exist
cmake -S tools/ide/toolchain -B /vm/data/rl-tools/ide-toolchain/build
cmake --build /vm/data/rl-tools/ide-toolchain/build --parallel 5
```

The default target is `stage`. A failed or unavailable verification prerequisite prevents staging and leaves no `VERIFIED` stamp. Source building and packaging can be requested independently without Node:

| Target | Product |
|---|---|
| `host` | Native tablegen tools |
| `sysroot` | compiler-rt builtins, wasi-libc, libc++abi, libc++ |
| `llvm-wasm` | WASI clang + lld and resource headers |
| `package` | Candidate `toolchain/{llvm.wasm,sysroot.tar,toolchain.json,SHA256SUMS}` |
| `bundle` | Candidate RLtools headers, example sources/settings, and header manifest |
| `parity` | Host-compiled `parity/training.wasm` with a compiler-generated header depfile |
| `verify` | Unit/build-graph checks, provenance, module identity, minimum stack depth, and a short compile/run check; browser smoke check when available |
| `verify-training` | Full learning check and stdout parity with the host-compiled program, excluding timing lines |
| `stage` | Verified candidate copied into the served bundle |

## Sources

[dependencies.json](../dependencies.json) owns the repository URLs, branch/tag references, upstream versions, target and CPU. CMake FetchContent checks out the LLVM fork's `rltools-wasi-22.1.8` branch and wasi-libc's `wasi-sdk-34` tag at configure time. The fetched commits are recorded in `source-revisions.json` and the package manifest.

The shared `cmake/fetch_source.cmake` fetches one ref at depth 1 with no tags. Reconfiguration follows the branch tip; an existing checkout is retained with a warning if the network is unavailable. `-DRL_TOOLS_OFFLINE_BUILD=ON` uses existing sources without a network request. Local changes are preserved and cause a clear failure.

FetchContent defaults to `<build>/_deps`; `-DFETCHCONTENT_BASE_DIR=<directory>` selects a persistent source cache. `-DFETCHCONTENT_SOURCE_DIR_LLVM_PROJECT=<checkout>` and `-DFETCHCONTENT_SOURCE_DIR_WASI_LIBC=<checkout>` use existing checkouts. Both the toolchain and optional host bootstrap use this workflow.

The build checks that source HEADs still match the configured revisions and that the checkouts are clean. A revision change clears the affected generated trees and installed sysroot; LLVM changes also clear native tablegen tools. Files removed by a dependency revision therefore cannot survive installation.

The LLVM fork provides WASI host support and libc++abi thread-local destruction without changing code generation. The cross toolchain enables wasi-libc's mmap/getpid/signal emulation.

The browser module includes clang and lld. Tool exclusions are derived from the fetched LLVM/Clang tool directories and the small enabled set. Native archiving tools remain prerequisites for building LLVM and the sysroot.

## Directories and configuration

All sub-builds belong to the `-B` directory:

```text
build/
  _deps/
  host/
  sysroot/{compiler-rt,wasi-libc,libcxx}/
  prefix/usr/
  resource/
  wasm/
  bundle/
    toolchain/
    examples/
    examples.json
    rl_tools_include.tar
    manifest.json
  parity/training.wasm
  VERIFIED
```

| CMake cache variable | Default | Meaning |
|---|---|---|
| `FETCHCONTENT_BASE_DIR` | `<build>/_deps` | Dependency source cache |
| `RL_TOOLS_OFFLINE_BUILD` | OFF | Use existing sources without network access |
| `RL_TOOLS_IDE_HOST_LLVM_BIN` | `/usr/lib/llvm-22/bin` | Native compiler tools |
| `RL_TOOLS_IDE_JOBS` | 5 | Jobs per stage, limited to 1–5 |
| `RL_TOOLS_IDE_LTO` | `thin` | `thin`, `full`, or `off` |
| `RL_TOOLS_IDE_ASSERTIONS` | ON | Assertions in the WASM compiler |
| `RL_TOOLS_IDE_OUTPUT_DIR` | `<build>/bundle` | Candidate **bundle root**, including its `toolchain/` subdirectory |
| `RL_TOOLS_IDE_STAGE_DIR` | `static/ide/build` | Destination for verified files |

Reconfiguring LTO updates the actual compiler and linker cache flags. Flags map both build and dependency source paths to stable prefixes. Builds use the configured LLVM commit time as `SOURCE_DATE_EPOCH`; archives fix ordering, ownership, and timestamps. Manifests record the selected flags, source identities, host compiler/linker versions, sizes, and hashes. Identical configuration does not rewrite configure metadata.

`RL_TOOLS_IDE_TOOLCHAIN_DIR` is replaced by `-B`. The old `RL_TOOLS_IDE_BUILD_HOST_CLANG` mode has a separate bootstrap entry point:

```bash
cmake -S tools/ide/bootstrap -B /vm/data/rl-tools/ide-host
cmake --build /vm/data/rl-tools/ide-host --parallel 5
cmake -S tools/ide/toolchain -B /vm/data/rl-tools/ide-toolchain/build \
    -DRL_TOOLS_IDE_HOST_LLVM_BIN=/vm/data/rl-tools/ide-host/host/bin
```

The bootstrap uses the system C/C++ compiler and the same LLVM branch declaration. Different toolchain `-B` directories have independent generated trees.

## Tests

```bash
node --test tests/src/ide/*.test.mjs
.venv/bin/python tests/src/ide/build_graph_test.py

# Quick verification and explicit full training/parity:
cmake --build /vm/data/rl-tools/ide-toolchain/build --target verify --parallel 5
cmake --build /vm/data/rl-tools/ide-toolchain/build --target verify-training --parallel 5

# The maximum-depth diagnostic is deliberately outside the default test suite:
node tests/src/ide/toolchain.mjs --bundle /vm/data/rl-tools/ide-toolchain/build/bundle --bisect-stack

# All package consumers take the same bundle root:
node tests/src/ide/pipeline.mjs --bundle /vm/data/rl-tools/ide-toolchain/build/bundle --example smoke
node tests/src/ide/pipeline.mjs --bundle /vm/data/rl-tools/ide-toolchain/build/bundle \
    --example pendulum_sac --reference /vm/data/rl-tools/ide-toolchain/build/parity/training.wasm
.venv/bin/python tests/src/ide/browser_test.py --bundle /vm/data/rl-tools/ide-toolchain/build/bundle
tests/src/ide/provenance.sh /vm/data/rl-tools/ide-toolchain/build/bundle
```

The main RLtools configure and this superbuild use the same CTest registration. Main-project tests check for artifacts at execution time and report missing prerequisites as skips; newly generated bundles require no reconfigure. Superbuild verification requires its selected checks to pass. CTest enforces timeouts; training tests have a longer timeout and the `ide-training` label. Quick checks have the `ide-verify` label.

The browser harness serves the selected candidate through its local HTTP server, even when the bundle is outside the repository. `--example pendulum_sac` requests full training in the browser; the default is the short `smoke` example. `RL_TOOLS_BROWSER_ARGS` adds browser flags when required by the host environment.

[examples.json](../../../static/ide/examples.json) owns compiler arguments, run arguments, and validation expectations for both examples. The browser and Node share the validator. The build-graph test uses small local source/compiler fixtures to test invalidation, reconfiguration, artifact recovery, and staging; it does not validate LLVM code generation.
