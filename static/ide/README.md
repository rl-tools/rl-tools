# RLtools IDE

The IDE compiles C++ with a WASI build of clang and lld inside a Web Worker, then runs the resulting WASI command module in a separate worker. The default example is a plain pendulum SAC training program; its stdout supplies the evaluation plot. Compilation and training happen in the browser.

Build the compiler and candidate bundle using the [toolchain instructions](../../tools/ide/toolchain/README.md). The default build verifies the candidate and stages it in `static/ide/build/`. Serve the repository root with the virtual environment's Python:

```bash
.venv/bin/python -m http.server -d . 8000
```

Open `http://localhost:8000/static/ide/`. Plain HTTP is sufficient; cross-origin isolation is unnecessary. Compile and Run execute one operation at a time. Stop cancels either operation. Worker failures and timeouts discard the compiler worker; a subsequent compile reloads its assets.

The argument field supports single/double quotes and backslash escapes. It performs no shell expansion. Arguments stay as an array after parsing, and output discovery uses the final clang driver job. The default arguments come from [examples.json](examples.json); `-fno-exceptions` is required by the sysroot.

The compiler worker owns the compiled module and the read-only `/usr` and `/include` mounts. Each compilation gets fresh source files and `/tmp`. Archives use ustar. Both the browser and Node tests use `toolchain.js`, `runtime.js`, and the same WASI implementation. The page and browser test also share `worker_client.js`, which handles request IDs, errors, timeouts, and cancellation.

Tests, custom bundle paths, optional host compiler bootstrapping, and the full training/parity check are described in the [toolchain README](../../tools/ide/toolchain/README.md).
