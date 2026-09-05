# RLtools IDE (browser)

Compiles an RLtools training program with clang running inside the browser (LLVM 21 built for `wasm32-wasip1`), runs the resulting WASI module in a Web Worker, and plots the evaluation lines the loop steps print. Nothing runs on a server; there are no bindings between the page and the program. The program is a plain `main()` that reads its seed from `argv`, writes to stdout, and also builds natively via the CMake target `rl_environments_pendulum_sac_wasi` (`src/rl/environments/pendulum/sac/wasi/`).

## Build
```
sudo apt install clang-22 lld-22                       # the wasm cross compiler (Ubuntu 26.04)
cmake -S tools/ide/toolchain -B /vm/data/rl-tools/ide-toolchain/build -G Ninja
cmake --build /vm/data/rl-tools/ide-toolchain/build    # fetches pinned llvm-project + wasi-libc, builds build/toolchain/{llvm.wasm,sysroot.tar,toolchain.json}; --target sources fetches only
tools/ide/bundle.sh                                    # include/rl_tools -> build/rl_tools_include.tar, example program, manifest
python3 -m http.server -d . 8000                       # from the repository root, then open http://localhost:8000/static/ide/
```
The toolchain superbuild (`tools/ide/toolchain/`) is a standalone CMake project with the source pins inline at the top; it is not built by the main RLtools configure. Sources and build trees live outside the repository in `$RL_TOOLS_IDE_TOOLCHAIN_DIR` (default `/vm/data/rl-tools/ide-toolchain`). Everything served lands in `static/ide/build/` (ignored by git). Plain HTTP is enough: no cross-origin isolation headers are required because nothing uses threads.

## Test without a browser
```
node tests/src/ide/pipeline.mjs      # Node >= 18 from the distro package (Ubuntu: sudo apt install nodejs): compiles and runs the example through the same modules the page uses (ctest test_ide_pipeline when node is found)
```
The end-to-end path (workers, transfers, chart) is exercised in a real browser; a headless Chrome harness is part of the toolchain plan.

## Layout
- `tar.js` reads the header and sysroot archives
- `filesystem.js` converts path maps to the shim's in-memory directory trees and back
- `process.js` runs one WASI command module: argv in, stdout/stderr lines out, one directory tree at `/`
- `toolchain.js` asks the clang driver for its job plan (`-###`) and runs each job (`clang++ -cc1`, `wasm-ld`) as its own process, since WASI cannot spawn
- `runtime.js` runs a compiled program
- `protocol.js` parses the evaluation and timing lines
- `worker.js` hosts compile and run off the main thread; `main.js` wires the page

The sysroot is mounted at `/usr`, the compiler's temporary files go to `/tmp`, and the project (editor contents plus `include/rl_tools`) sits at `/`, so the arguments read like a shell invocation: `-std=c++17 -O2 -fno-exceptions -Iinclude training.cpp -o training.wasm`. `-fno-exceptions` is required: the WASI libc++ has no exception support.
