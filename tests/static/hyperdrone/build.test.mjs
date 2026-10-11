import assert from "node:assert/strict";
import {spawnSync} from "node:child_process";
import {existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {fileURLToPath} from "node:url";
import {test} from "node:test";

const script = fileURLToPath(new URL("../../../src/rl/environments/hyperdrone/web/build.sh", import.meta.url));
const default_configuration = "high-mb4-aa1-480x270";

function fixture(t){
    const root = mkdtempSync(join(tmpdir(), "hyperdrone-build-"));
    t.after(() => rmSync(root, {recursive: true, force: true}));
    const bin = join(root, "bin");
    const logs = join(root, "logs");
    const dependencies = join(root, "dependencies");
    const build = join(root, "build output");
    const header = join(build, "hyperdrone_web_configurations.h");
    for(const path of [bin, logs, join(dependencies, "stb"), join(dependencies, "json/include/nlohmann")]){
        mkdirSync(path, {recursive: true});
    }
    for(const name of ["em++", "cmake", "emcmake"]){
        writeFileSync(join(bin, name), [
            "#!/bin/sh",
            'tool=${0##*/}',
            'if [ "$tool" = cmake ] && [ "${1-}" = --build ]; then tool=cmake-build; fi',
            'printf \'%s\\n\' "$@" > "$HYPERDRONE_TEST_LOG_DIR/$tool"',
            'printf \'%s\\n\' "$EMCC_CORES" "$BINARYEN_CORES" > "$HYPERDRONE_TEST_LOG_DIR/$tool.cores"',
            "",
        ].join("\n"), {mode: 0o755});
    }
    writeFileSync(join(bin, "git"), "#!/bin/sh\nexit 99\n", {mode: 0o755});
    writeFileSync(join(dependencies, "stb/stb_image.h"), "");
    writeFileSync(join(dependencies, "json/include/nlohmann/json.hpp"), "");
    return {
        build, header, logs,
        run(...args){
            const result = spawnSync("/bin/bash", [script, ...args], {
                encoding: "utf8",
                env: {...process.env, PATH: `${bin}:/usr/bin:/bin`, HYPERDRONE_TEST_LOG_DIR: logs,
                    EMCC_CORES: "99", BINARYEN_CORES: "99",
                    RL_TOOLS_WASM_DEPS_DIR: dependencies, RL_TOOLS_WASM_BUILD_DIR: build},
            });
            assert.equal(result.error, undefined);
            return result;
        },
        compiled(){
            const fidelity = {Low: "low", Medium: "medium", High: "high", VeryHigh: "very-high"};
            return [...readFileSync(header, "utf8").matchAll(/Config<rendering::raytracing::(\w+),\s*(\d+),\s*(\d+),\s*(\d+)>/g)]
                .map(([, shading, motion, aa, scale]) => `${fidelity[shading]}-mb${motion}-aa${aa}-${480 * scale}x${270 * scale}`);
        },
    };
}

function assert_jobs(f, jobs){
    const args = readFileSync(join(f.logs, "cmake-build"), "utf8").trim().split("\n");
    assert.deepEqual(args.slice(-2), ["-j", String(jobs)]);
    for(const tool of ["em++", "emcmake", "cmake-build"]){
        assert.equal(readFileSync(join(f.logs, `${tool}.cores`), "utf8"), `${jobs}\n${jobs}\n`, tool);
    }
}

test("full, single, and sparse builds generate exactly the selected instantiations", t => {
    const f = fixture(t);
    const listed = f.run("--list-configurations");
    assert.equal(listed.status, 0, listed.stderr);
    const supported = listed.stdout.trim().split("\n");
    assert.equal(supported.length, 32);
    assert.equal(new Set(supported).size, 32);
    assert.ok(supported.includes(default_configuration));
    const sparse = ["very-high-mb4-aa2-960x540", "low-mb1-aa1-480x270"];
    for(const args of [[], [default_configuration], sparse]){
        const result = f.run(...args);
        assert.equal(result.status, 0, result.stderr);
        assert.deepEqual(f.compiled(), args.length ? args : supported);
        const compiler_args = readFileSync(join(f.logs, "em++"), "utf8").trim().split("\n");
        assert.ok(compiler_args.some((arg, index) => arg === "-I" && compiler_args[index + 1] === f.build));
        assert.ok(compiler_args.find(arg => arg.startsWith("-sEXPORTED_FUNCTIONS=")).split(/=|,/).includes("_demo_configurations"));
        assert.equal(compiler_args.at(-1), join(f.build, "hyperdrone_web.js"));
        assert_jobs(f, 5);
    }
});

test("job options work before, between, and after configurations and override toolchain limits", t => {
    const f = fixture(t);
    const selected = [default_configuration, "low-mb1-aa1-480x270"];
    for(const args of [
        ["-j12", ...selected],
        [selected[0], "-j", "12", selected[1]],
        [...selected, "-j12"],
    ]){
        const result = f.run(...args);
        assert.equal(result.status, 0, result.stderr);
        assert.deepEqual(f.compiled(), selected);
        assert_jobs(f, 12);
    }
    const result = f.run("-j", "12");
    assert.equal(result.status, 0, result.stderr);
    assert.equal(f.compiled().length, 32);
    assert_jobs(f, 12);
});

test("help, listing, and invalid selections never invoke the toolchain or generate a header", t => {
    const f = fixture(t);
    for(const option of ["--help", "--list-configurations"]){
        const result = f.run(option);
        assert.equal(result.status, 0, result.stderr);
        assert.ok(result.stdout.length > 0);
    }
    for(const [args, error] of [
        [["high-mb4-aa1-123x456"], /Unknown configuration/],
        [[default_configuration, "not-a-configuration"], /Unknown configuration/],
        [[default_configuration, default_configuration], /Duplicate configuration/],
    ]){
        const result = f.run(...args);
        assert.notEqual(result.status, 0);
        assert.match(result.stderr, error);
    }
    for(const args of [["-j"], ["-j0"], ["-j", "0"], ["-j-1"], ["-j", "-1"], ["-jabc"], ["-j", "1.5"], [default_configuration, "-j", "invalid"]]){
        const result = f.run(...args);
        assert.notEqual(result.status, 0);
        assert.match(result.stderr, /positive integer|job count|jobs/i);
    }
    assert.deepEqual(readdirSync(f.logs), []);
    assert.equal(existsSync(f.header), false);
});
