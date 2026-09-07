#!/usr/bin/env python3
"""Exercise the actual superbuild graph with tiny local compiler/source fixtures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[3]


def write(path, content, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(0o755)


class BuildGraph(unittest.TestCase):
    def test_incremental_build_and_verification(self):
        with tempfile.TemporaryDirectory(prefix="rl-tools-ide-graph-") as temporary:
            root = Path(temporary)
            repo = root / "repository"
            for relative in ["tools/ide", "tests/src/ide", "static/ide"]:
                shutil.copytree(REPOSITORY / relative, repo / relative, ignore=shutil.ignore_patterns("build", "external", "__pycache__"))
            for relative in ["include/rl_tools/rl_tools.h", "src/rl/environments/pendulum/sac/wasi/training.cpp"]:
                write(repo / relative, (REPOSITORY / relative).read_text())
            external, metadata, binary = root / "sources", root / "metadata", root / "bin"
            host_resource, build = root / "host-resource", root / "build"
            candidate, served = root / "candidate with spaces", root / "served"
            cmake_file = repo / "tools/ide/toolchain/CMakeLists.txt"
            for name in ["llvm-project", "wasi-libc"]:
                write(metadata / name / "HEAD", "a" * 40 if name == "llvm-project" else "c" * 40)
            write(metadata / "repository/HEAD", "a" * 40)
            write(host_resource / "include/stddef.h", "host headers\n")
            os.utime(host_resource / "include/stddef.h", (1700000000, 1700000000))
            write(host_resource / "include/removed.h", "removed host header\n")
            write(binary / "git", f"""#!{sys.executable}
import sys
from pathlib import Path
args=sys.argv[1:]
directory=Path(args[args.index('-C')+1]) if '-C' in args else Path.cwd()
name=directory.name if directory.name in ['llvm-project','wasi-libc'] else 'repository'
metadata=Path({str(metadata)!r})/name
if 'symbolic-ref' in args: raise SystemExit(1)
if 'status' in args: raise SystemExit(0)
if '--absolute-git-dir' in args: print(metadata)
elif '--format=%ct' in args: print(1700000000)
elif 'HEAD' in args: print((metadata/'HEAD').read_text())
else: raise SystemExit('unexpected git call: '+str(args))
""", True)
            host_script = f"""#!{sys.executable}
import sys
if '-print-resource-dir' in sys.argv: print({str(host_resource)!r})
elif '-dumpversion' in sys.argv: print('22.1.8')
else: print('clang version 22.1.8 (build graph fixture)')
"""
            for tool in ["clang", "clang++", "wasm-ld", "llvm-ar", "llvm-ranlib", "llvm-nm"]:
                write(binary / tool, host_script, True)
            self.environment = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"]}
            llvm = external / "llvm-project"
            write(llvm / "cmake/Modules/LLVMVersion.cmake", "set(LLVM_VERSION_MAJOR 22)\nset(LLVM_VERSION_MINOR 1)\nset(LLVM_VERSION_PATCH 8)\n")
            for directory in ["llvm/tools/llvm-driver", "llvm/tools/llvm-ar", "llvm/tools/new-tool", "clang/tools/driver", "clang/tools/new-tool"]:
                write(llvm / directory / "CMakeLists.txt", "")
            produce = root / "produce.cmake"
            write(produce, 'file(READ "${INPUT}" content)\nget_filename_component(directory "${OUTPUT}" DIRECTORY)\nfile(MAKE_DIRECTORY "${directory}")\nfile(CONFIGURE OUTPUT "${OUTPUT}" CONTENT "${content}\\n${FLAGS}\\n" @ONLY)\n')
            llvm_cmake = r"""
cmake_minimum_required(VERSION 3.24)
project(graph_llvm NONE)
file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/configured.log" "${CMAKE_C_FLAGS}\n")
foreach(tool llvm-tblgen llvm-min-tblgen clang-tblgen llvm)
    add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/bin/${tool}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/input.txt" "-DOUTPUT=${CMAKE_BINARY_DIR}/bin/${tool}" "-DFLAGS=${CMAKE_C_FLAGS}" -P "@PRODUCE@"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/input.txt" VERBATIM)
    if(tool STREQUAL "llvm")
        add_custom_target(llvm-driver DEPENDS "${CMAKE_BINARY_DIR}/bin/${tool}")
    else()
        add_custom_target(${tool} DEPENDS "${CMAKE_BINARY_DIR}/bin/${tool}")
    endif()
endforeach()
add_custom_command(OUTPUT "${CMAKE_BINARY_DIR}/usr/include/stddef.h"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/headers.txt" "-DOUTPUT=${CMAKE_BINARY_DIR}/usr/include/stddef.h" -P "@PRODUCE@"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/headers.txt" VERBATIM)
add_custom_target(clang-resource-headers DEPENDS "${CMAKE_BINARY_DIR}/usr/include/stddef.h")
"""
            write(llvm / "llvm/CMakeLists.txt", llvm_cmake.replace("@PRODUCE@", str(produce)))
            write(llvm / "llvm/input.txt", "module")
            write(llvm / "llvm/headers.txt", "compiler headers")
            libraries = {
                llvm / "compiler-rt": ["lib/wasm32-unknown-wasip1/libclang_rt.builtins.a"],
                external / "wasi-libc": ["lib/wasm32-wasip1/libc.a"],
                llvm / "runtimes": ["lib/wasm32-wasip1/libc++.a", "lib/wasm32-wasip1/libc++abi.a"],
            }
            for source, outputs in libraries.items():
                text = 'cmake_minimum_required(VERSION 3.24)\nproject(graph_runtime NONE)\nfile(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/configured.log" "${CMAKE_C_FLAGS}\\n")\n'
                for index, output in enumerate(outputs):
                    destination, filename = str(Path(output).parent), Path(output).name
                    text += f'''
add_custom_command(OUTPUT "${{CMAKE_BINARY_DIR}}/{filename}"
    COMMAND "${{CMAKE_COMMAND}}" "-DINPUT=${{CMAKE_CURRENT_SOURCE_DIR}}/input.txt" "-DOUTPUT=${{CMAKE_BINARY_DIR}}/{filename}" "-DFLAGS=${{CMAKE_C_FLAGS}}" -P "{produce}"
    DEPENDS "${{CMAKE_CURRENT_SOURCE_DIR}}/input.txt" VERBATIM)
add_custom_target(library_{index} ALL DEPENDS "${{CMAKE_BINARY_DIR}}/{filename}")
install(FILES "${{CMAKE_BINARY_DIR}}/{filename}" DESTINATION "{destination}")
'''
                write(source / "CMakeLists.txt", text)
                write(source / "input.txt", source.name)
            configure = ["cmake", "-S", str(repo / "tools/ide/toolchain"), "-B", str(build),
                         "-DFETCHCONTENT_SOURCE_DIR_LLVM_PROJECT=" + str(external / "llvm-project"),
                         "-DFETCHCONTENT_SOURCE_DIR_WASI_LIBC=" + str(external / "wasi-libc"),
                         "-DRL_TOOLS_IDE_HOST_LLVM_BIN=" + str(binary),
                         "-DRL_TOOLS_IDE_OUTPUT_DIR=" + str(candidate),
                         "-DRL_TOOLS_IDE_STAGE_DIR=" + str(served),
                         "-DIDE_NODE_EXECUTABLE=" + str(root / "missing-node")]
            self.run_command(configure)
            def build_targets(*targets):
                return self.run_command(["cmake", "--build", str(build), "--target", *targets, "--parallel", "5"])
            build_targets("package", "bundle")
            self.assertTrue((build / "host/bin/llvm-tblgen").is_file())
            self.assertTrue((build / "wasm/bin/llvm").is_file())
            self.assertFalse(served.exists())
            self.assertFalse((build / "VERIFIED").exists())
            manifest = json.loads((candidate / "toolchain/toolchain.json").read_text())
            self.assertIn("-DLLVM_ENABLE_PROJECTS=clang;lld", manifest["cmake_args_llvm_wasm"])
            self.assertNotIn("lld", manifest["cmake_args_llvm_wasm"])
            self.assertIn("-DLLVM_TOOL_NEW_TOOL_BUILD=OFF", manifest["cmake_args_llvm_wasm"])
            self.assertIn("-DLLVM_TOOL_LLVM_AR_BUILD=OFF", manifest["cmake_args_llvm_wasm"])
            provenance = ["ctest", "--test-dir", str(build), "-R", "^test_ide_toolchain_provenance$", "--output-on-failure", "--timeout", "20"]
            self.run_command(provenance)
            manifest_file = candidate / "toolchain/toolchain.json"
            original_manifest = manifest_file.read_text()
            manifest["llvm_commit"] = "d" * 40
            manifest_file.write_text(json.dumps(manifest))
            result = self.run_command(provenance, success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("llvm_commit: expected", result.stdout)
            manifest_file.write_text(original_manifest)
            module = candidate / "toolchain/llvm.wasm"
            original_module = module.read_bytes()
            module.write_bytes(original_module + b"corrupt")
            result = self.run_command(provenance, success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("sha256 mismatch", result.stdout)
            module.write_bytes(original_module)
            build_targets("package", "bundle")
            outputs = [candidate / name for name in ["toolchain/llvm.wasm", "toolchain/sysroot.tar", "toolchain/toolchain.json", "rl_tools_include.tar", "manifest.json"]]
            outputs += [build / "resource/include/stddef.h", build / "resource/lib/wasm32-unknown-wasip1/libclang_rt.builtins.a"]
            timestamps = [path.stat().st_mtime_ns for path in outputs]
            configurations = [path / "configured.log" for path in [llvm / "llvm", *libraries]]
            configured = [path.read_text() for path in configurations]
            result = build_targets("package", "bundle")
            self.assertNotIn("Generating resource/", result.stdout)
            self.assertEqual(configured, [path.read_text() for path in configurations])
            self.assertEqual(timestamps, [path.stat().st_mtime_ns for path in outputs])
            write(cmake_file, cmake_file.read_text() + "\n")
            build_targets("package", "bundle")
            self.assertEqual(timestamps, [path.stat().st_mtime_ns for path in outputs])
            (host_resource / "include/removed.h").unlink()
            build_targets("resource-dir")
            self.assertFalse((build / "resource/include/removed.h").exists())
            self.run_command(configure + ["-DRL_TOOLS_IDE_LTO=off"])
            build_targets("package")
            flags = next(line for line in (build / "wasm/CMakeCache.txt").read_text().splitlines() if line.startswith("CMAKE_C_FLAGS:"))
            self.assertNotIn("-flto", flags)
            manifest = json.loads((candidate / "toolchain/toolchain.json").read_text())
            self.assertEqual(manifest["lto"], "off")
            self.assertTrue(all("-flto" not in argument for argument in manifest["cmake_args_llvm_wasm"]))
            (candidate / "toolchain/llvm.wasm").unlink()
            (build / "wasm/bin/llvm").unlink()
            build_targets("package")
            self.assertTrue((candidate / "toolchain/llvm.wasm").is_file())
            original = (candidate / "rl_tools_include.tar").read_bytes()
            write(repo / "include/rl_tools/new.h", "new header")
            build_targets("bundle")
            self.assertNotEqual(original, (candidate / "rl_tools_include.tar").read_bytes())
            with tarfile.open(candidate / "rl_tools_include.tar") as archive:
                self.assertEqual(archive.extractfile("rl_tools/new.h").read(), b"new header")
            (repo / "include/rl_tools/new.h").unlink()
            build_targets("bundle")
            with tarfile.open(candidate / "rl_tools_include.tar") as archive:
                self.assertNotIn("rl_tools/new.h", archive.getnames())
            libc_log = external / "wasi-libc/configured.log"
            before = libc_log.read_text()
            stale_header = build / "prefix/usr/include/removed.h"
            write(stale_header, "removed by the next dependency revision")
            write(metadata / "wasi-libc/HEAD", "b" * 40)
            write(cmake_file, cmake_file.read_text().replace("set(wasi_libc_ref wasi-sdk-34)", "set(wasi_libc_ref next-ref)"))
            build_targets("package")
            self.assertFalse(stale_header.exists())
            with tarfile.open(candidate / "toolchain/sysroot.tar") as archive:
                self.assertNotIn("include/removed.h", archive.getnames())
            self.assertGreater(len(libc_log.read_text()), len(before))
            self.assertEqual(json.loads((candidate / "toolchain/toolchain.json").read_text())["wasi_libc_commit"], "b" * 40)
            result = self.run_command(["cmake", "--build", str(build), "--target", "stage", "--parallel", "5"], success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing-node", result.stdout)
            self.assertFalse((build / "VERIFIED").exists())
            self.assertFalse(served.exists())

            # The verification result alone is a fixture; staging still uses the real producer and file dependencies.
            write(build / "VERIFIED", "preverified build graph fixture")
            build_targets("stage")
            (served / "toolchain/llvm.wasm").unlink()
            build_targets("stage")
            self.assertEqual((served / "toolchain/llvm.wasm").read_bytes(), (candidate / "toolchain/llvm.wasm").read_bytes())
            staged_manifest = (served / "manifest.json").read_bytes()
            runtime = repo / "static/ide/worker_client.js"
            runtime.unlink()
            result = self.run_command(["cmake", "--build", str(build), "--target", "stage", "--parallel", "5"], success=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing-node", result.stdout)
            self.assertFalse((build / "VERIFIED").exists())
            self.assertEqual(staged_manifest, (served / "manifest.json").read_bytes())
            other = root / "other-build"
            other_configure = configure.copy()
            other_configure[other_configure.index("-B") + 1] = str(other)
            self.run_command(other_configure)
            self.assertFalse((other / "wasm/bin/llvm").exists())
            self.assertTrue((build / "wasm/bin/llvm").exists())

    def test_prerequisites_are_checked_at_execution(self):
        self.environment = os.environ.copy()
        with tempfile.TemporaryDirectory(prefix="rl-tools-ide-prerequisites-") as temporary:
            root = Path(temporary)
            artifact = root / "artifact"
            command = ["cmake", "-DPROGRAM=" + sys.executable, "-DARGUMENTS=-c;print('artifact appeared')",
                       "-DFILES=" + str(artifact), "-P", str(REPOSITORY / "tests/src/ide/run_test.cmake")]
            self.assertIn("SKIP: IDE prerequisite", self.run_command(command).stdout)
            artifact.touch()
            result = self.run_command(command)
            self.assertIn("artifact appeared", result.stdout)
            self.assertNotIn("SKIP", result.stdout)
            write(root / "skip.py", "print('SKIP: IDE prerequisite unavailable')\nraise SystemExit(1)\n")
            write(root / "CMakeLists.txt", f'''
cmake_minimum_required(VERSION 3.24)
project(verification NONE)
enable_testing()
include("{REPOSITORY}/tests/src/ide/CMakeLists.txt")
ide_register_test(required PROGRAM "{sys.executable}" ARGS "{root}/skip.py" REQUIRED TRUE TIMEOUT 20)
ide_register_test(optional PROGRAM "{sys.executable}" ARGS "{root}/skip.py" REQUIRED FALSE TIMEOUT 20)
''')
            self.run_command(["cmake", "-S", str(root), "-B", str(root / "build")])
            ctest = ["ctest", "--test-dir", str(root / "build"), "--output-on-failure", "--timeout", "20", "-R"]
            self.assertNotEqual(self.run_command(ctest + ["^required$"], success=False).returncode, 0)
            self.assertIn("Skipped", self.run_command(ctest + ["^optional$"]).stdout)

    def run_command(self, command, success=True):
        result = subprocess.run(command, env=self.environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        if success and result.returncode != 0:
            self.fail(" ".join(command) + "\n" + result.stdout[-18000:])
        return result


if __name__ == "__main__":
    unittest.main()
