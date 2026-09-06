import { test } from "node:test";
import assert from "node:assert/strict";
import { parseLine } from "../../../static/ide/protocol.js";

test("evaluation lines", () => {
    assert.deepEqual(parseLine("Step: 10000/10000 Mean return: -187.561 Mean episode length: 200"), { type: "evaluation", step: 10000, stepLimit: 10000, meanReturn: -187.561, meanEpisodeLength: 200 });
    assert.deepEqual(parseLine("Step: 0/10000 Mean return: -1.5e+03 Mean episode length: 200 (extra)"), { type: "evaluation", step: 0, stepLimit: 10000, meanReturn: -1500, meanEpisodeLength: 200 });
    assert.equal(parseLine("Step: 0/10000 Mean return: -nan Mean episode length: 200").meanReturn, NaN);
});

test("timing lines", () => {
    assert.deepEqual(parseLine("Loop step: 1000, env step: 1000, SPS: 412.7"), { type: "steps_per_second", step: 1000, environmentStep: 1000, stepsPerSecond: 412.7 });
    assert.deepEqual(parseLine("Time: 25.3s"), { type: "total_time", seconds: 25.3 });
});

test("unrelated lines", () => {
    assert.equal(parseLine(""), null);
    assert.equal(parseLine("Step 1/2"), null);
    assert.equal(parseLine("warning: unused variable"), null);
});
