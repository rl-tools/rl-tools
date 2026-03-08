import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var frozenFrame: CGImage?
    @State private var predictedAngle: Double?
    @State private var predictor: OpaquePointer?
    @State private var inferenceTimer: Timer?

    private let imageSize = 64
    // Training FOV: cos_fov=0.66, half-angle = atan(0.66/2), full FOV in degrees
    private let trainingFOV = 2.0 * atan(0.33) * 180.0 / .pi // ~36.5°

    var body: some View {
        VStack(spacing: 16) {
            HStack(spacing: 16) {
                // Left: frozen image
                ZStack {
                    if let frozen = frozenFrame {
                        Image(frozen, scale: 1, label: Text("Frozen"))
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                    } else {
                        Rectangle()
                            .fill(Color.gray.opacity(0.3))
                            .overlay(Text("Press Space to capture")
                                .foregroundColor(.secondary))
                    }
                }
                .frame(width: 320, height: 240)
                .clipped()
                .border(Color.blue, width: 2)

                // Right: live camera (cropped to training FOV)
                ZStack {
                    if let frame = camera.currentFrame {
                        let cropped = cropToTrainingFOV(frame)
                        Image(cropped, scale: 1, label: Text("Live"))
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                    } else {
                        Rectangle()
                            .fill(Color.gray.opacity(0.3))
                            .overlay(Text("Camera starting...")
                                .foregroundColor(.secondary))
                    }
                }
                .frame(width: 320, height: 240)
                .clipped()
                .border(Color.green, width: 2)
            }

            // Angle display
            if let angle = predictedAngle {
                Text(String(format: "Predicted yaw: %.1f\u{00B0}", angle))
                    .font(.system(size: 28, weight: .bold, design: .monospaced))
                    .foregroundColor(.primary)
            } else {
                Text("No prediction yet")
                    .font(.system(size: 28, weight: .bold, design: .monospaced))
                    .foregroundColor(.secondary)
            }

            Text("Space: capture reference | Esc: quit | Camera FOV: \(String(format: "%.0f", camera.horizontalFOV))\u{00B0} \u{2192} crop to \(String(format: "%.0f", trainingFOV))\u{00B0}")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .frame(minWidth: 700, minHeight: 350)
        .background(KeyPressHandler(onSpace: captureFrame, onEscape: quit))
        .onAppear {
            camera.start()
            loadModel()
        }
        .onDisappear {
            inferenceTimer?.invalidate()
            camera.stop()
            if let p = predictor {
                yaw_predictor_destroy(p)
            }
        }
    }

    private func loadModel() {
        let args = CommandLine.arguments
        guard args.count > 1 else {
            print("Usage: YawPredictor <path/to/yaw-predictor.h5> [--fov <degrees>]")
            return
        }
        let h5Path = args[1]

        // Parse optional --fov argument
        if let fovIdx = args.firstIndex(of: "--fov"), fovIdx + 1 < args.count,
           let fov = Double(args[fovIdx + 1]) {
            camera.horizontalFOV = fov
        }

        print("Loading model from: \(h5Path)")
        print("Training FOV: \(String(format: "%.1f", trainingFOV))\u{00B0}, Camera FOV: \(String(format: "%.1f", camera.horizontalFOV))\u{00B0}")
        predictor = yaw_predictor_create(h5Path)
        if predictor == nil {
            print("Failed to load model")
        }
    }

    private func captureFrame() {
        guard let frame = camera.currentFrame else { return }
        // Deep copy: CGImage from CIContext may share backing CVPixelBuffer that gets recycled
        frozenFrame = deepCopyCGImage(cropToTrainingFOV(frame))
        inferenceTimer?.invalidate()
        inferenceTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            runInference()
        }
    }

    // Center-crop the camera image so its angular extent matches the training FOV
    private func cropToTrainingFOV(_ image: CGImage) -> CGImage {
        let cameraFOV = camera.horizontalFOV
        guard cameraFOV > 0, trainingFOV < cameraFOV else { return image }

        let w = Double(image.width)
        let h = Double(image.height)
        let cameraHalfRad = cameraFOV * .pi / 360.0
        let trainingHalfRad = trainingFOV * .pi / 360.0

        // Fraction of frame width that corresponds to training FOV
        let cropFraction = tan(trainingHalfRad) / tan(cameraHalfRad)
        let cropW = w * cropFraction
        // Square crop: use same angular extent vertically
        let cropH = cropW  // square output
        let cropSide = min(cropW, min(cropH, min(w, h)))

        let x = (w - cropSide) / 2.0
        let y = (h - cropSide) / 2.0
        let rect = CGRect(x: x, y: y, width: cropSide, height: cropSide)

        return image.cropping(to: rect) ?? image
    }

    private func runInference() {
        guard let frozen = frozenFrame,
              let live = camera.currentFrame,
              let p = predictor else { return }

        let croppedLive = cropToTrainingFOV(live)

        let pixelsA = cgImageToFloatRGB(frozen, size: imageSize)
        let pixelsB = cgImageToFloatRGB(croppedLive, size: imageSize)

        var sinCos: [Float] = [0, 0]
        pixelsA.withUnsafeBufferPointer { a in
            pixelsB.withUnsafeBufferPointer { b in
                sinCos.withUnsafeMutableBufferPointer { out in
                    yaw_predictor_evaluate(p, a.baseAddress, b.baseAddress, out.baseAddress)
                }
            }
        }

        let angle = atan2(Double(sinCos[0]), Double(sinCos[1])) * 180.0 / .pi
        predictedAngle = angle
    }

    private func quit() {
        NSApplication.shared.terminate(nil)
    }
}

// Force a deep copy of a CGImage so it doesn't share a recycled CVPixelBuffer
private func deepCopyCGImage(_ image: CGImage) -> CGImage {
    let w = image.width
    let h = image.height
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil,
        width: w, height: h,
        bitsPerComponent: 8,
        bytesPerRow: w * 4,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ) else { return image }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage() ?? image
}

// Convert CGImage to NHWC float RGB array [1, size, size, 3] in [0,1]
private func cgImageToFloatRGB(_ image: CGImage, size: Int) -> [Float] {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * size
    var pixelData = [UInt8](repeating: 0, count: size * size * bytesPerPixel)

    guard let ctx = CGContext(
        data: &pixelData,
        width: size,
        height: size,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ) else {
        return [Float](repeating: 0, count: size * size * 3)
    }

    ctx.interpolationQuality = .high
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

    // RGBA -> RGB float, NHWC layout
    var result = [Float](repeating: 0, count: size * size * 3)
    for i in 0..<(size * size) {
        result[i * 3 + 0] = Float(pixelData[i * 4 + 0]) / 255.0
        result[i * 3 + 1] = Float(pixelData[i * 4 + 1]) / 255.0
        result[i * 3 + 2] = Float(pixelData[i * 4 + 2]) / 255.0
    }
    return result
}

// Invisible view that captures key events
struct KeyPressHandler: NSViewRepresentable {
    let onSpace: () -> Void
    let onEscape: () -> Void

    func makeNSView(context: Context) -> KeyCaptureView {
        let view = KeyCaptureView()
        view.onSpace = onSpace
        view.onEscape = onEscape
        return view
    }

    func updateNSView(_ nsView: KeyCaptureView, context: Context) {
        nsView.onSpace = onSpace
        nsView.onEscape = onEscape
    }

    class KeyCaptureView: NSView {
        var onSpace: (() -> Void)?
        var onEscape: (() -> Void)?

        override var acceptsFirstResponder: Bool { true }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            window?.makeFirstResponder(self)
        }

        override func keyDown(with event: NSEvent) {
            switch event.keyCode {
            case 49: // space
                onSpace?()
            case 53: // escape
                onEscape?()
            default:
                super.keyDown(with: event)
            }
        }
    }
}
