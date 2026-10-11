import SwiftUI
import UIKit

@main
struct BenchmarkApp: App {
    var body: some Scene {
        WindowGroup {
            BenchmarkView()
        }
    }
}

struct BenchmarkView: View {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var run = BenchmarkRun()

    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            Text("RLtools Benchmark")
                .font(.largeTitle.bold())
            Text("4,096 cameras · 64 × 64 · Metal\nThe full benchmark takes at least 26 minutes. Keep this app open until it finishes.")
            Button("Run benchmark") {
                run.start(quickCheck: false)
            }
            .buttonStyle(.borderedProminent)
            .disabled(run.running)
            Button("Quick check") {
                run.start(quickCheck: true)
            }
            .buttonStyle(.bordered)
            .disabled(run.running)
            Text("Quick check runs 24 short cases with all 4,096 cameras.")
                .font(.footnote)
                .foregroundStyle(.secondary)
            Text(run.status)
                .textSelection(.enabled)
            Text("Results: Files → On My iPhone → RLtools Benchmark. Copy the entire run folder to your Mac.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(24)
        .onAppear {
            if CommandLine.arguments.contains("--quick-check") {
                run.start(quickCheck: true)
            } else if CommandLine.arguments.contains("--benchmark") {
                run.start(quickCheck: false)
            }
        }
        .onChange(of: scenePhase) { phase in
            if phase != .active {
                run.markInterrupted()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)) { _ in
            run.recordThermalState()
        }
    }
}

@MainActor
final class BenchmarkRun: ObservableObject {
    @Published private(set) var running = false
    @Published private(set) var status = "Ready. Run a quick check first."
    private var directory: URL?
    private var metadata: [String: Any] = [:]

    func start(quickCheck: Bool) {
        guard !running else {
            return
        }
        guard let scene = Bundle.main.url(forResource: "ProcTHOR-Train-1", withExtension: "glb"),
              let build = Bundle.main.url(forResource: "build", withExtension: "json") else {
            status = "Bundled benchmark resources are missing."
            return
        }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss-SSS"
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let mode = quickCheck ? "quick-check" : "benchmark"
        let output = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("\(formatter.string(from: Date()))-\(mode)", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: build, to: output.appendingPathComponent("build.json"))
        } catch {
            status = "Cannot prepare results folder: \(error.localizedDescription)"
            return
        }
        var system = utsname()
        uname(&system)
        let model = withUnsafeBytes(of: &system.machine) {
            String(cString: $0.baseAddress!.assumingMemoryBound(to: CChar.self))
        }
        directory = output
        metadata = [
            "mode": mode,
            "started_at": ISO8601DateFormatter().string(from: Date()),
            "device_model": model,
            "ios_version": UIDevice.current.systemVersion,
            "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "thermal_start": ProcessInfo.processInfo.thermalState.rawValue,
            "thermal_max": ProcessInfo.processInfo.thermalState.rawValue,
            "interrupted": false,
            "status": "running"
        ]
        guard saveMetadata() else {
            return
        }
        running = true
        status = quickCheck ? "Running quick check…" : "Running benchmark…"
        UIApplication.shared.isIdleTimerDisabled = true
        DispatchQueue.global(qos: .userInitiated).async {
            let result = autoreleasepool {
                scene.path.withCString { scenePath in
                    output.path.withCString { outputPath in
                        rl_tools_paper_benchmark_run(scenePath, outputPath, quickCheck)
                    }
                }
            }
            Task { @MainActor in
                self.finish(result: result)
            }
        }
    }

    func markInterrupted() {
        guard running else {
            return
        }
        metadata["interrupted"] = true
        status = "Run interrupted. Let it finish, then run again for valid timings."
        saveMetadata()
    }

    func recordThermalState() {
        guard running else {
            return
        }
        metadata["thermal_max"] = max(metadata["thermal_max"] as? Int ?? 0, ProcessInfo.processInfo.thermalState.rawValue)
    }

    private func finish(result: Int32) {
        running = false
        UIApplication.shared.isIdleTimerDisabled = false
        let interrupted = metadata["interrupted"] as? Bool ?? false
        metadata["exit_code"] = result
        metadata["finished_at"] = ISO8601DateFormatter().string(from: Date())
        metadata["thermal_end"] = ProcessInfo.processInfo.thermalState.rawValue
        metadata["status"] = result != 0 ? "failed" : interrupted ? "interrupted" : "complete"
        if result != 0 {
            status = "Benchmark failed (\(result)). See benchmark.log in the results folder."
        } else if interrupted {
            status = "Finished with an interruption. Run again for valid timings."
        } else {
            status = "Finished. Results saved in \(directory!.lastPathComponent)."
        }
        saveMetadata()
    }

    @discardableResult
    private func saveMetadata() -> Bool {
        do {
            let data = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: directory!.appendingPathComponent("device.json"), options: .atomic)
            return true
        } catch {
            status = "Cannot save device metadata: \(error.localizedDescription)"
            return false
        }
    }
}
