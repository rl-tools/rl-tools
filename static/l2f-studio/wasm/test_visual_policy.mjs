import assert from "node:assert/strict"
import {readFileSync} from "node:fs"
import {execFileSync} from "node:child_process"
import {fileURLToPath} from "node:url"
import createSimulator from "../external/blob/l2f-interface.js"
import createInference from "../external/blob/dyn-inference.js"
import {packVisualPair, visualPolicyProfile} from "../visual_policy.js"

const checkpoint = process.argv[2]
assert(checkpoint, "Usage: node test_visual_policy.mjs checkpoint.h5")
const examples = JSON.parse(execFileSync(fileURLToPath(new URL("../../../.venv/bin/python", import.meta.url)), ["-c", `
import h5py, json, sys
with h5py.File(sys.argv[1]) as f:
    print(json.dumps({
        "images": f["example/inputs/0"][0].tolist(),
        "sensors": f["example/inputs/1"][0].tolist(),
        "actions": f["example/outputs/0"][0].tolist()
    }))
`, checkpoint], {maxBuffer: 16 * 1024 * 1024}))

const inference = await createInference({wasmBinary: readFileSync(new URL("../external/blob/dyn-inference.wasm", import.meta.url))})
inference.FS.writeFile("/checkpoint.h5", readFileSync(checkpoint))
const model = new inference.DynInference()
assert(model.load("/checkpoint.h5"))
assert(model.verify().pass)
const profile = visualPolicyProfile({
    meta: JSON.parse(model.get_meta()), input_shapes: model.get_input_shapes(), output_dim: model.get_output_dim(),
})
assert.equal(profile.state_type, "VisualState")
assert(profile.observation.endsWith(".InertialVelocity"))
let maxDifference = 0
for(let sample = 0; sample < examples.images.length; sample++) {
    const image = examples.images[sample]
    const current = image.flatMap(row => row.slice(0, 80).flatMap(pixel => pixel.slice(0, 3)))
    const target = image.flatMap(row => row.slice(0, 80).flatMap(pixel => pixel.slice(3, 6)))
    const packed = packVisualPair(current, target, 80, 50)
    assert.deepEqual(Array.from(packed), image.flat(2))
    for(let repeat = 0; repeat < 3; repeat++) {
        const actual = Array.from(model.evaluate_tuple([packed, new Float32Array(examples.sensors[sample])]))
        assert.equal(actual.length, 4)
        assert(actual.every(Number.isFinite))
        maxDifference = Math.max(maxDifference, ...actual.map((v, i) => Math.abs(v - examples.actions[sample][i])))
    }
}
assert(maxDifference < 1e-5)
model.destroy()
model.delete()

const simulator = await createSimulator({wasmBinary: readFileSync(new URL("../external/blob/l2f-interface.wasm", import.meta.url))})
const state = new simulator.VisualState(12)
state.initial_state()
const initial = Array.from(state.get_policy_observation())
assert.equal(initial.length, 65)
assert(initial.slice(0, 3).every((v, i) => v === [0, 0, 1][i]))
assert(initial.slice(62).every(v => v === 0))
assert(Math.abs(initial[40] - 9.81) < 1e-5)
const before = state.get_state()
for(let i = 0; i < 10; i++) {
    state.observe()
    state.get_policy_observation()
}
assert.equal(state.get_state(), before)
const parameters = JSON.parse(state.get_parameters())
assert.deepEqual(parameters.ctbr_controller.rate_limit, [4, 4, 2])
assert.equal(parameters.integration.dt, 0.01)
const dynamics = parameters.dynamics
const [c, b, a] = dynamics.rotor_thrust_coefficients[0]
const hoverCommand = (-b + Math.sqrt(b * b - 4 * a * (c - dynamics.mass * 9.81 / 4))) / (2 * a)
const hoverAction = 2 * hoverCommand - 1
const restingState = JSON.parse(state.get_state())
assert.equal(restingState.q_estimate.length, 4)
assert.equal(restingState.linear_acceleration_body_history.length, 8)
assert.equal(restingState.gyro.length, 3)
restingState.rpm.fill(hoverCommand)
state.set_state(JSON.stringify(restingState))
for(let step = 0; step < 200; step++) {
    for(let axis = 0; axis < 4; axis++) {
        state.set_action(axis, axis === 0 ? hoverAction : 0)
    }
    state.step()
    assert(Array.from(state.get_policy_observation()).every(Number.isFinite))
}
assert(Math.hypot(...state.get_position()) < 1e-3)
assert(Math.abs(state.get_policy_observation()[6] - hoverAction) < 1e-6)
assert.equal(state.get_policy_observation()[7], 0)
state.sample_initial_state()
assert(Array.from(state.get_policy_observation()).every(Number.isFinite))
assert(Array.from(state.get_policy_observation()).slice(62).every(v => v === 0))
state.delete()

const legacy = new simulator.State(12)
legacy.initial_state()
legacy.observe()
assert.equal(legacy.get_observation().length, 21 + 64 * 4)
legacy.delete()
console.log(`Paired image, single-sample inference, CTBR hover, sensor/reset and legacy checks passed; max action difference ${maxDifference}`)
