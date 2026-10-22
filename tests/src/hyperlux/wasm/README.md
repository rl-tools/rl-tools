# Run the WASM golden generator

From the repository root, on macOS arm64 or Linux x86_64 with CMake 3.25 or newer:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DRL_TOOLS_HYPERLUX_ENABLE_WASM_GOLDENS=ON
cmake --build build --target hyperlux_golden_wasm -j5
./build/tests/src/hyperlux/wasm/generate_goldens
```

WASM goldens are opt-in because the pinned wasi-sdk and Wasmtime downloads are large. Tests and raytracing must also be enabled; check the `wasm goldens` entry in the configure summary for availability and the reason if disabled.

Default Wasmtime location: `.dependencies/build/wasm/wasmtime-v*-*/wasmtime` (exact path in the configure summary).

```bash
wasmtime=/absolute/path/to/wasmtime
scene=$(./build/src/conta/conta conta:<sha1 from golden_manifest.h>)
output="$PWD/build/goldens"
mkdir -p "$output"

"$wasmtime" run \
  -C compiler=cranelift -C cache=n -C parallel-compilation=n \
  -W exceptions=y -W nan-canonicalization=y \
  -W simd=n -W relaxed-simd=n -W threads=n \
  -S inherit-env=n -S inherit-stdin=n \
  --dir "$(dirname "$scene")::/scene" \
  --dir "$output::/output" \
  build/tests/src/hyperlux/wasm/generate_golden.wasm \
  --scene "/scene/$(basename "$scene")" --output-dir /output
```

The `--dir` options expose the host directories inside the module as `/scene` and `/output`. This writes into `build/goldens` with the runtime profile from [render.sh.in](render.sh.in).

The `generate_goldens` script writes into the configured corpus directory by default; `--output-dir DIR` selects another destination and `--dump-dir DIR` writes raw comparison outputs.
