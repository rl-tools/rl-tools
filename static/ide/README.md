# RLtools IDE (browser)

Compiles an RLtools training program with clang running inside the browser (LLVM built for `wasm32-wasip1` from source by `tools/ide/toolchain`), runs the resulting WASI module in a Web Worker, and plots the evaluation lines the loop steps print. Nothing runs on a server; there are no bindings between the page and the program. The program is a plain `main()` that reads its seed from `argv`, writes to stdout, and also builds natively via the CMake target `rl_environments_pendulum_sac_wasi` (`src/rl/environments/pendulum/sac/wasi/`).

## Build
```
sudo apt install clang-22 lld-22 ninja-build nodejs           # the wasm cross compiler (Ubuntu 26.04); ninja and node are optional
cmake -S tools/ide/toolchain -B /vm/data/rl-tools/ide-toolchain/build -G Ninja   # fetches the llvm-project fork + wasi-libc into .dependencies
cmake --build /vm/data/rl-tools/ide-toolchain/build           # about 40 min: builds and verifies build/toolchain/{llvm.wasm,sysroot.tar,toolchain.json}
tools/ide/bundle.sh                                           # include/rl_tools -> build/rl_tools_include.tar, example program, manifest (the superbuild's verify stage runs it too)
python3 -m http.server -d . 8000                              # from the repository root, then open http://localhost:8000/static/ide/
```
The toolchain superbuild (`tools/ide/toolchain/`, see its README) is a standalone CMake project, not built by the main RLtools configure. Its sources are FetchContent checkouts in the repository's `.dependencies/<build directory name>/` like every other RLtools dependency, pinned at the top of its `CMakeLists.txt`: RLtools' fork of llvm-project at the branch that carries the WASI commit, and stock wasi-libc; build trees live outside the repository in `$RL_TOOLS_IDE_TOOLCHAIN_DIR` (default `/vm/data/rl-tools/ide-toolchain`). Everything served lands in `static/ide/build/` (ignored by git). Plain HTTP is enough: no cross-origin isolation headers are required because nothing uses threads.

## Tests
```
node --test tests/src/ide/*.test.mjs                          # unit tests of wasi.js, tar.js, protocol.js (ctest test_ide_unit)
node tests/src/ide/toolchain.mjs                              # module shape, compiler identity vs toolchain.json, stack probe (ctest test_ide_toolchain)
node tests/src/ide/pipeline.mjs [--reference training.wasm]   # compile + run the example through the page's modules, optional parity with the host-compiled program (ctest test_ide_pipeline)
python3 tests/src/ide/browser_test.py                         # the same through the real workers in headless Chrome or Firefox, via static/ide/test.html (ctest test_ide_browser)
tests/src/ide/provenance.sh                                   # the served package matches the pins and its hashes (ctest test_ide_toolchain_provenance)
```
Node is the distro package (Ubuntu: `sudo apt install nodejs`); the browser test needs only the standard library and an installed browser.

## Layout
- `wasi.js` implements WASI preview1 for one in-memory directory tree preopened at `/`, stdio as byte sinks, no threads and no sockets; the compiler imports 29 of its 46 functions. An optional trace hook logs every syscall.
- `tar.js` reads the header and sysroot archives
- `filesystem.js` converts path maps to directory trees and back
- `process.js` runs one WASI command module: argv in, stdout/stderr lines out, one directory tree at `/`
- `toolchain.js` asks the clang driver for its job plan (`-###`) and runs each job (`clang++ -cc1`, `wasm-ld`) as its own process, since WASI cannot spawn
- `runtime.js` runs a compiled program
- `protocol.js` parses the evaluation and timing lines
- `worker.js` hosts compile and run off the main thread; `main.js` wires the page; `test.html` drives the same workers for the browser test

The sysroot is mounted read-only at `/usr`, the compiler's temporary files go to `/tmp`, and the project (editor contents plus `include/rl_tools`) sits at `/`, so the arguments read like a shell invocation: `-std=c++17 -O2 -fno-exceptions -Iinclude training.cpp -o training.wasm`. `-fno-exceptions` is required: the WASI libc++ has no exception support.
