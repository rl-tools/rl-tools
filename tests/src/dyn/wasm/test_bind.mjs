import { readFileSync } from 'fs';
import { resolve } from 'path';
import * as jsfive from 'jsfive';

const checkpoint_path = process.argv[2];
const wasm_path = process.argv[3] || './build/test_dyn_h5_bind.js';
if(!checkpoint_path){ console.error("Usage: node test_bind.mjs <checkpoint.h5> [wasm.js]"); process.exit(1); }

const h5_buf = readFileSync(resolve(checkpoint_path));
const h5 = new jsfive.File(h5_buf.buffer);

let test_input_ds, expected_output_ds;
try { test_input_ds = h5.get('example/input'); } catch(e) {}
try { expected_output_ds = h5.get('example/output'); } catch(e) {}
if(!test_input_ds) try { test_input_ds = h5.get('test_input'); } catch(e) {}
if(!expected_output_ds) try { expected_output_ds = h5.get('expected_output'); } catch(e) {}
if(!test_input_ds || !expected_output_ds){ console.error("FAIL: could not find example data in checkpoint"); process.exit(1); }
const input_data = new Float32Array(test_input_ds.value.flat(Infinity));
const expected_output = new Float32Array(expected_output_ds.value.flat(Infinity));
const input_shape = test_input_ds.shape;

console.log("Ground truth from jsfive:");
console.log("  input shape:", input_shape, "total:", input_data.length);
console.log("  expected output:", Array.from(expected_output).map(v => v.toFixed(6)));
console.log("  input first 5:", Array.from(input_data.slice(0, 5)).map(v => v.toFixed(4)));

const createModule = (await import(resolve(wasm_path))).default;
const Module = await createModule();

const h5_raw = readFileSync(resolve(checkpoint_path));
Module.FS.writeFile('/checkpoint.h5', new Uint8Array(h5_raw));
if(!Module.load_model('/checkpoint.h5', input_shape)){
    console.error("FAIL: could not load model");
    process.exit(1);
}

function max_diff(a, b){
    let m = 0;
    for(let i = 0; i < Math.min(a.length, b.length); i++){
        const d = Math.abs(a[i] - b[i]);
        if(d > m) m = d;
    }
    return m;
}

function report(label, actual){
    const diff = max_diff(actual, expected_output);
    const pass = diff < 1e-4;
    console.log(`[${label}]  output: ${Array.from(actual).map(v => v.toFixed(6))}  max_diff: ${diff.toExponential(2)}  ${pass ? 'PASS' : 'FAIL'}`);
    return pass;
}

let all_pass = true;

// Test 1: emscripten::val element-by-element copy (the original broken path)
const out_val = new Float32Array(Module.evaluate_via_val(input_data));
all_pass &= report('val copy ', out_val);

// Test 2: typed_memory_view heap copy (the proposed fix)
const out_heap = new Float32Array(Module.evaluate_via_heap(input_data));
all_pass &= report('heap copy', out_heap);

console.log("\n" + (all_pass ? "ALL PASS" : "SOME FAILED"));
process.exit(all_pass ? 0 : 1);
