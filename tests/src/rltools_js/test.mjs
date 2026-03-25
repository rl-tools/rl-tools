import { load } from '../../../static/rltools.js/main.js'
import assert from 'node:assert'
import fs from 'node:fs'
import path from 'node:path'

const data_dir = path.join(import.meta.dirname, '../../data')

let passed = 0
let failed = 0

function run_test(name, fixture_file){
    process.stdout.write(`  ${name}... `)
    try {
        const buffer = fs.readFileSync(path.join(data_dir, fixture_file))
        const model = load(buffer.buffer)
        assert(model !== null, `${name}: model is null`)
        console.log('OK')
        passed++
    } catch (e) {
        console.log(`FAIL: ${e.message}`)
        failed++
    }
}

console.log('rltools.js tests:')
run_test('Conv2d + MaxPool2d + AvgPool2d + Dense', 'rltools_js_conv2d.h5')
run_test('Conv2d + Flatten + Dense', 'rltools_js_flatten.h5')
run_test('Parallel (Dense + Dense + Head)', 'rltools_js_parallel.h5')

console.log(`\n${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)
