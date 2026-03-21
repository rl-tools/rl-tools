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
                            AnnotatedImageView(
                                image: showNativeResolution ? downscale(frozen, to: imageSize) : frozen,
                                label: "Reference",
                                showsCenter: true,
                                markers: []
                            )
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
                            AnnotatedImageView(
                                image: displayImage(frame),
                                label: "Live",
                                showsCenter: false,
                                markers: liveImageMarkers
                            )
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

            #if os(iOS)
            if let translationIndicator = translationIndicator {
                TranslationOverlayView(indicator: translationIndicator)
            }
            #endif

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

    private var liveImageMarkers: [ImageMarker] {
        var markers: [ImageMarker] = []
        if let gt = computeGroundTruth() {
            markers.append(
                ImageMarker(
                    label: "GT",
                    horizontalNorm: gt.horizontalNorm,
                    verticalNorm: gt.verticalNorm,
                    color: .green
                )
            )
        }
        if let pred = prediction {
            markers.append(
                ImageMarker(
                    label: "Pred",
                    horizontalNorm: pred.px,
                    verticalNorm: pred.py,
                    color: .red
                )
            )
        }
        return markers
    }

    private var translationIndicator: TranslationIndicator? {
        guard let referenceTransform, let currentTransform = camera.currentTransform else { return nil }
        return computeTranslationIndicator(reference: referenceTransform, current: currentTransform)
    }
    #else
    private var liveImageMarkers: [ImageMarker] {
        if let pred = prediction {
            return [
                ImageMarker(
                    label: "Pred",
                    horizontalNorm: pred.px,
                    verticalNorm: pred.py,
                    color: .red
                )
            ]
        }
        return []
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

private struct ImageMarker: Identifiable {
    let id = UUID()
    let label: String
    let horizontalNorm: Double
    let verticalNorm: Double
    let color: Color
}

#if os(iOS)
private struct TranslationIndicator {
    let right: Double
    let up: Double
    let depth: Double
    let magnitudeMeters: Double
}
#endif

private struct AnnotatedImageView: View {
    let image: CGImage
    let label: String
    let showsCenter: Bool
    let markers: [ImageMarker]

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Image(image, scale: 1, label: Text(label))
                    .resizable()
                    .interpolation(.none)
                    .aspectRatio(contentMode: .fit)

                if showsCenter {
                    CrosshairView(color: Color.white.opacity(0.75))
                        .position(x: geometry.size.width * 0.5, y: geometry.size.height * 0.5)
                }

                ForEach(markers) { marker in
                    MarkerView(label: marker.label, color: marker.color)
                        .position(position(for: marker, in: geometry.size))
                }
            }
        }
    }

    private func position(for marker: ImageMarker, in size: CGSize) -> CGPoint {
        let clampedHorizontal = min(max(marker.horizontalNorm, -1.0), 1.0)
        let clampedVertical = min(max(marker.verticalNorm, -1.0), 1.0)
        let x = size.width * 0.5 * (1.0 - clampedHorizontal)
        let y = size.height * 0.5 * (1.0 + clampedVertical)
        return CGPoint(x: x, y: y)
    }
}

private struct CrosshairView: View {
    let color: Color

    var body: some View {
        ZStack {
            Rectangle()
                .fill(color)
                .frame(width: 20, height: 2)
            Rectangle()
                .fill(color)
                .frame(width: 2, height: 20)
        }
    }
}

private struct MarkerView: View {
    let label: String
    let color: Color

    var body: some View {
        VStack(spacing: 4) {
            CrosshairView(color: color)
            Text(label)
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(color)
                .padding(.horizontal, 4)
                .padding(.vertical, 2)
                .background(.black.opacity(0.55))
                .clipShape(RoundedRectangle(cornerRadius: 4))
        }
    }
}

#if os(iOS)
private struct TranslationOverlayView: View {
    let indicator: TranslationIndicator

    var body: some View {
        let rightLength = CGFloat(abs(indicator.rightNormalized)) * 26.0
        let upLength = CGFloat(abs(indicator.upNormalized)) * 26.0
        let depthStrength = CGFloat(abs(indicator.depthNormalized))
        let depthColor = indicator.depthNormalized >= 0 ? Color.orange : Color.cyan
        let depthLabel = indicator.depthNormalized >= 0 ? "Near" : "Far"

        VStack(alignment: .leading, spacing: 8) {
            ZStack {
                RoundedRectangle(cornerRadius: 14)
                    .fill(.black.opacity(0.55))
                HStack(spacing: 14) {
                    ZStack {
                        Circle()
                            .stroke(Color.white.opacity(0.16), lineWidth: 1)
                            .frame(width: 64, height: 64)
                        axisArrow(horizontal: true, positive: indicator.rightNormalized >= 0, length: rightLength, color: .pink)
                            .frame(width: 64, height: 64)
                        axisArrow(horizontal: false, positive: indicator.upNormalized >= 0, length: upLength, color: .yellow)
                            .frame(width: 64, height: 64)
                        Circle()
                            .fill(Color.white.opacity(0.9))
                            .frame(width: 6, height: 6)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        ZStack {
                            Circle()
                                .stroke(depthColor.opacity(0.35), lineWidth: 1)
                                .frame(width: 28, height: 28)
                            Circle()
                                .stroke(depthColor.opacity(0.18), lineWidth: 6)
                                .frame(width: 28 + depthStrength * 18, height: 28 + depthStrength * 18)
                            Circle()
                                .fill(depthColor.opacity(0.85))
                                .frame(width: 8 + depthStrength * 10, height: 8 + depthStrength * 10)
                        }
                        Text(depthLabel)
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundColor(depthColor)
                        Text(String(format: "%.0f mm", indicator.magnitudeMeters * 1000.0))
                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                            .foregroundColor(.white.opacity(0.8))
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
            }
            .frame(width: 132, height: 84)
        }
    }

    private func axisArrow(horizontal: Bool, positive: Bool, length: CGFloat, color: Color) -> some View {
        GeometryReader { geometry in
            let center = CGPoint(x: geometry.size.width * 0.5, y: geometry.size.height * 0.5)
            ZStack {
                Path { path in
                    path.move(to: center)
                    if horizontal {
                        path.addLine(to: CGPoint(x: center.x + (positive ? length : -length), y: center.y))
                    } else {
                        path.addLine(to: CGPoint(x: center.x, y: center.y + (positive ? -length : length)))
                    }
                }
                .stroke(color, style: StrokeStyle(lineWidth: 3, lineCap: .round))

                if length > 1 {
                    Path { path in
                        let tip: CGPoint
                        let a: CGPoint
                        let b: CGPoint
                        if horizontal {
                            let x = center.x + (positive ? length : -length)
                            tip = CGPoint(x: x, y: center.y)
                            a = CGPoint(x: x + (positive ? -7 : 7), y: center.y - 4)
                            b = CGPoint(x: x + (positive ? -7 : 7), y: center.y + 4)
                        } else {
                            let y = center.y + (positive ? -length : length)
                            tip = CGPoint(x: center.x, y: y)
                            a = CGPoint(x: center.x - 4, y: y + (positive ? 7 : -7))
                            b = CGPoint(x: center.x + 4, y: y + (positive ? 7 : -7))
                        }
                        path.move(to: tip)
                        path.addLine(to: a)
                        path.addLine(to: b)
                        path.closeSubpath()
                    }
                    .fill(color)
                }
            }
        }
    }
}

private extension TranslationIndicator {
    var rightNormalized: Double {
        max(-1.0, min(1.0, right / 0.05))
    }

    var upNormalized: Double {
        max(-1.0, min(1.0, up / 0.05))
    }

    var depthNormalized: Double {
        max(-1.0, min(1.0, depth / 0.05))
    }
}

private func computeTranslationIndicator(
    reference: simd_float4x4,
    current: simd_float4x4
) -> TranslationIndicator {
    let referencePosition = simd_make_float3(reference.columns.3.x, reference.columns.3.y, reference.columns.3.z)
    let currentPosition = simd_make_float3(current.columns.3.x, current.columns.3.y, current.columns.3.z)
    let deltaWorld = currentPosition - referencePosition
    let currentRotation = simd_float3x3(
        simd_make_float3(current.columns.0.x, current.columns.0.y, current.columns.0.z),
        simd_make_float3(current.columns.1.x, current.columns.1.y, current.columns.1.z),
        simd_make_float3(current.columns.2.x, current.columns.2.y, current.columns.2.z)
    )
    let deltaPhone = currentRotation.transpose * deltaWorld
    return TranslationIndicator(
        right: Double(deltaPhone.x),
        up: Double(deltaPhone.y),
        depth: Double(-deltaPhone.z),
        magnitudeMeters: Double(simd_length(deltaWorld))
    )
}
#endif
