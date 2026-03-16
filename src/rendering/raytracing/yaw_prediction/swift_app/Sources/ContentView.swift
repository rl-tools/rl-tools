import SwiftUI
#if os(iOS)
import UniformTypeIdentifiers
import simd
import UIKit
#endif

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var frozenFrame: CGImage?
    @State private var prediction: (px: Double, py: Double, roll: Double)?
    @State private var predictor: OpaquePointer?
    @State private var inferenceTimer: Timer?
    @State private var showNativeResolution = false
    @State private var modelLoaded = false
    @State private var modelStatusText = "Model not loaded"
    #if os(iOS)
    @State private var datasetRecorder = DatasetRecorder(imageSize: 64)
    @State private var referenceTransform: simd_float4x4?
    @State private var datasetStatusText = "Dataset capture idle"
    @State private var datasetDirectory: URL?
    @State private var isDatasetRecording = false
    @State private var isExportPresented = false
    @State private var showFilePicker = false
    #endif

    private let imageSize = 64
    #if os(iOS)
    private let bundledModelName = "yaw-predictor5-beta"
    #endif

    var body: some View {
        VStack(spacing: 16) {
            HStack(spacing: 16) {
                VStack {
                    Text("Reference").font(.caption)
                    Group {
                        if let frozen = frozenFrame {
                            Image(showNativeResolution ? downscale(frozen, to: imageSize) : frozen,
                                  scale: 1, label: Text("Reference"))
                                .resizable()
                                .interpolation(.none)
                                .aspectRatio(contentMode: .fit)
                        } else {
                            Rectangle()
                                .fill(Color.gray.opacity(0.3))
                                .overlay(Text("No capture yet").foregroundColor(.secondary))
                        }
                    }
                    .frame(width: 240, height: 240)
                    .clipped()
                }

                VStack {
                    Text("Live").font(.caption)
                    Group {
                        if let frame = camera.currentFrame {
                            Image(displayImage(frame), scale: 1, label: Text("Live"))
                                .resizable()
                                .interpolation(.none)
                                .aspectRatio(contentMode: .fit)
                        } else {
                            Rectangle()
                                .fill(Color.gray.opacity(0.3))
                                .overlay(Text(camera.trackingReady ? "Camera starting..." : "Initializing AR...").foregroundColor(.secondary))
                        }
                    }
                    .frame(width: 240, height: 240)
                    .clipped()
                }
            }

            VStack(spacing: 4) {
                Text(modelStatusText)
                    .font(.system(size: 14, weight: .medium, design: .monospaced))
                    .foregroundColor(modelLoaded ? .green : .secondary)
                if let pred = prediction {
                    let hfov = camera.horizontalFOV
                    let pxDeg = normalizedDisplacementToDegrees(pred.px, fovDegrees: hfov)
                    let pyDeg = normalizedDisplacementToDegrees(pred.py, fovDegrees: hfov)
                    let rollDeg = pred.roll * 180.0
                    Text(String(format: "H: %+.1f\u{00B0}  V: %+.1f\u{00B0}  R: %+.1f\u{00B0}", pxDeg, pyDeg, rollDeg))
                        .font(.system(size: 18, weight: .bold, design: .monospaced))
                } else {
                    Text("No prediction yet")
                        .font(.system(size: 18, weight: .bold, design: .monospaced))
                        .foregroundColor(.secondary)
                }
                #if os(iOS)
                if let gt = computeGroundTruth() {
                    Text(String(format: "GT H: %+.1f\u{00B0}  V: %+.1f\u{00B0}  R: %+.1f\u{00B0}", gt.horizontalDegrees, gt.verticalDegrees, gt.rollDegrees))
                        .font(.system(size: 18, weight: .bold, design: .monospaced))
                        .foregroundColor(.blue)
                    if let pred = prediction {
                        let hfov = camera.horizontalFOV
                        let errH = normalizedDisplacementToDegrees(pred.px, fovDegrees: hfov) - gt.horizontalDegrees
                        let errV = normalizedDisplacementToDegrees(pred.py, fovDegrees: hfov) - gt.verticalDegrees
                        let errR = pred.roll * 180.0 - gt.rollDegrees
                        Text(String(format: "Err: %+.1f\u{00B0}  %+.1f\u{00B0}  %+.1f\u{00B0}", errH, errV, errR))
                            .font(.system(size: 18, weight: .bold, design: .monospaced))
                            .foregroundColor(.red)
                    }
                }
                #endif
            }

            HStack {
                Button("Capture Reference") {
                    captureFrame()
                }
                #if os(macOS)
                .keyboardShortcut(.space, modifiers: [])
                #endif

                Toggle("64\u{00D7}64", isOn: $showNativeResolution)

                #if os(iOS)
                Button("Load Model") {
                    showFilePicker = true
                }
                #endif
            }

            #if os(iOS)
            VStack(spacing: 8) {
                HStack {
                    Button(isDatasetRecording ? "Stop Dataset" : "Start Dataset") {
                        toggleDatasetRecording()
                    }
                    Button("Export Dataset") {
                        isExportPresented = true
                    }
                    .disabled(datasetDirectory == nil)
                }
                Text(datasetStatusText)
                    .font(.system(size: 14, weight: .medium, design: .monospaced))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
            #endif
        }
        .padding()
        .onAppear {
            camera.start()
            #if os(macOS)
            loadModelFromArgs()
            #else
            loadBundledModel()
            #endif
        }
        .onDisappear {
            inferenceTimer?.invalidate()
            camera.stop()
            if let p = predictor {
                yaw_predictor_destroy(p)
            }
        }
        #if os(iOS)
        .onReceive(camera.$latestARFrame.compactMap { $0 }) { frame in
            guard isDatasetRecording else { return }
            updateDatasetStatus(with: frame)
            do {
                if try datasetRecorder.considerSample(frame: frame, horizontalFOVDegrees: camera.horizontalFOV) {
                    updateDatasetStatus(with: frame, didCapture: true)
                }
            } catch {
                datasetStatusText = "Dataset write failed: \(error.localizedDescription)"
                isDatasetRecording = false
            }
        }
        .sheet(isPresented: $isExportPresented) {
            if let datasetDirectory {
                ActivityViewController(activityItems: [datasetDirectory])
            }
        }
        #endif
        #if os(iOS)
        .fileImporter(isPresented: $showFilePicker,
                      allowedContentTypes: [
                        UTType(filenameExtension: "tar") ?? .data,
                        UTType(filenameExtension: "h5") ?? .data
                      ],
                      allowsMultipleSelection: false) { result in
            if case .success(let urls) = result, let url = urls.first {
                loadModel(from: url.path)
            }
        }
        #endif
    }

    #if os(macOS)
    private func loadModelFromArgs() {
        let args = CommandLine.arguments
        guard args.count > 1 else {
            print("Usage: YawPredictor <path/to/model.h5> [--fov <degrees>]")
            return
        }
        if let fovIdx = args.firstIndex(of: "--fov"), fovIdx + 1 < args.count,
           let fov = Double(args[fovIdx + 1]) {
            camera.horizontalFOV = fov
        }
        loadModel(from: args[1])
    }
    #endif

    #if os(iOS)
    private func loadBundledModel() {
        if let url = Bundle.main.url(forResource: bundledModelName, withExtension: "tar") {
            _ = loadModel(from: url.path)
        } else {
            modelLoaded = false
            modelStatusText = "Bundled model missing: \(bundledModelName).tar"
            print("Bundled model not found: \(bundledModelName).tar")
        }
    }
    #endif

    @discardableResult
    private func loadModel(from path: String) -> Bool {
        if let p = predictor {
            yaw_predictor_destroy(p)
            predictor = nil
        }
        predictor = yaw_predictor_create(path)
        modelLoaded = predictor != nil
        if modelLoaded {
            modelStatusText = "Model loaded: \((path as NSString).lastPathComponent)"
            print("Loaded model from: \(path)")
        } else {
            modelStatusText = "Failed to load: \((path as NSString).lastPathComponent)"
            print("Failed to load model from: \(path)")
        }
        return modelLoaded
    }

    #if os(iOS)
    private func computeGroundTruth() -> AttitudeGroundTruth? {
        guard let refT = referenceTransform, let curT = camera.currentTransform else { return nil }
        return computeGroundTruthDegrees(reference: refT, current: curT, horizontalFOVDegrees: camera.horizontalFOV)
    }
    #endif

    private func displayImage(_ image: CGImage) -> CGImage {
        let cropped = centerSquareCrop(image)
        if showNativeResolution {
            return downscale(cropped, to: imageSize)
        }
        return cropped
    }

    private func captureFrame() {
        guard let frame = camera.currentFrame else { return }
        frozenFrame = centerSquareCrop(deepCopyCGImage(frame))
        #if os(iOS)
        referenceTransform = camera.currentTransform
        #endif
        inferenceTimer?.invalidate()
        inferenceTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = computeInference()
                DispatchQueue.main.async {
                    prediction = result
                }
            }
        }
    }

    #if os(iOS)
    private func toggleDatasetRecording() {
        if isDatasetRecording {
            do {
                datasetDirectory = try datasetRecorder.stop(horizontalFOVDegrees: camera.horizontalFOV)
                isDatasetRecording = false
                let count = datasetRecorder.sampleCount
                datasetStatusText = "Dataset stopped. Saved \(count) samples."
            } catch {
                datasetStatusText = "Failed to finalize dataset: \(error.localizedDescription)"
                isDatasetRecording = false
            }
            return
        }

        guard let frame = camera.latestARFrame else {
            datasetStatusText = "AR frame unavailable. Wait for tracking."
            return
        }

        do {
            try datasetRecorder.start(frame: frame, horizontalFOVDegrees: camera.horizontalFOV)
            frozenFrame = centerSquareCrop(deepCopyCGImage(frame.image))
            referenceTransform = frame.transform
            prediction = nil
            datasetDirectory = datasetRecorder.sessionPaths?.root
            isDatasetRecording = true
            datasetStatusText = "Dataset started. Saved frame 0. Rotate while keeping translation low."
        } catch {
            datasetStatusText = "Failed to start dataset: \(error.localizedDescription)"
        }
    }

    private func updateDatasetStatus(with frame: CapturedARFrame, didCapture: Bool = false) {
        guard let status = datasetRecorder.status(for: frame) else { return }
        let driftMM = status.translationDriftMeters * 1000.0
        let gate = status.isTranslationAcceptable ? "stable" : "drifting"
        if didCapture {
            datasetStatusText = String(
                format: "Captured %d samples. Drift %.0f mm (%@).",
                status.sampleCount + 1,
                driftMM,
                gate
            )
            return
        }
        if let age = status.lastCaptureAgeSeconds {
            datasetStatusText = String(
                format: "Dataset live: %d samples, drift %.0f mm (%@), %.2fs since last save.",
                status.sampleCount,
                driftMM,
                gate,
                age
            )
        } else {
            datasetStatusText = String(
                format: "Dataset live: %d samples, drift %.0f mm (%@).",
                status.sampleCount,
                driftMM,
                gate
            )
        }
    }
    #endif

    private func centerSquareCrop(_ image: CGImage) -> CGImage {
        let w = image.width
        let h = image.height
        let side = min(w, h)
        let x = (w - side) / 2
        let y = (h - side) / 2
        return image.cropping(to: CGRect(x: x, y: y, width: side, height: side)) ?? image
    }

    private func downscale(_ image: CGImage, to size: Int) -> CGImage {
        guard let ctx = CGContext(
            data: nil,
            width: size, height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        ) else { return image }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return ctx.makeImage() ?? image
    }

    private func computeInference() -> (px: Double, py: Double, roll: Double)? {
        guard let frozen = frozenFrame,
              let live = camera.currentFrame,
              let p = predictor else { return nil }

        let croppedLive = centerSquareCrop(live)
        let pixelsA = cgImageToFloatRGB(frozen, size: imageSize)
        let pixelsB = cgImageToFloatRGB(croppedLive, size: imageSize)

        var output: [Float] = [0, 0, 0]
        pixelsA.withUnsafeBufferPointer { a in
            pixelsB.withUnsafeBufferPointer { b in
                output.withUnsafeMutableBufferPointer { o in
                    yaw_predictor_evaluate(p, a.baseAddress, b.baseAddress, o.baseAddress)
                }
            }
        }

        return (Double(output[0]), Double(output[1]), Double(output[2]))
    }
}

#if os(iOS)
struct ActivityViewController: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
#endif

private func deepCopyCGImage(_ image: CGImage) -> CGImage {
    let w = image.width
    let h = image.height
    guard let ctx = CGContext(
        data: nil,
        width: w, height: h,
        bitsPerComponent: 8,
        bytesPerRow: w * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ) else { return image }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage() ?? image
}

private func cgImageToFloatRGB(_ image: CGImage, size: Int) -> [Float] {
    let bytesPerPixel = 4
    var pixelData = [UInt8](repeating: 0, count: size * size * bytesPerPixel)

    guard let ctx = CGContext(
        data: &pixelData,
        width: size,
        height: size,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerPixel * size,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ) else {
        return [Float](repeating: 0, count: size * size * 3)
    }

    ctx.interpolationQuality = .high
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

    var result = [Float](repeating: 0, count: size * size * 3)
    for i in 0..<(size * size) {
        result[i * 3 + 0] = Float(pixelData[i * 4 + 0]) / 255.0
        result[i * 3 + 1] = Float(pixelData[i * 4 + 1]) / 255.0
        result[i * 3 + 2] = Float(pixelData[i * 4 + 2]) / 255.0
    }
    return result
}
