# Run the WASM golden generator

From the repository root, on macOS arm64 or Linux x86_64:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rendering_raytracing_golden_wasm -j5
```

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
  build/tests/src/rendering/raytracing/wasm/generate_golden.wasm \
  --scene "/scene/$(basename "$scene")" --output-dir /output
```

The `--dir` options expose the host directories inside the module as `/scene` and `/output`. This writes into `build/goldens` with the runtime profile from [render.sh.in](render.sh.in).

To generate into the configured corpus directory, run `./build/tests/src/rendering/raytracing/wasm/generate_goldens`; `--output-dir DIR` selects another destination and `--dump-dir DIR` writes raw comparison outputs.
