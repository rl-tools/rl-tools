import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var frozenFrame: CGImage?
    @State private var predictedAngle: Double?
    @State private var predictor: OpaquePointer?
    @State private var inferenceTimer: Timer?

    private let imageSize = 64

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

                // Right: live camera
                ZStack {
                    if let frame = camera.currentFrame {
                        Image(frame, scale: 1, label: Text("Live"))
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

            Text("Space: capture reference | Esc: quit")
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
            print("Usage: YawPredictor <path/to/yaw-predictor.h5>")
            return
        }
        let h5Path = args[1]
        print("Loading model from: \(h5Path)")
        predictor = yaw_predictor_create(h5Path)
        if predictor == nil {
            print("Failed to load model")
        }
    }

    private func captureFrame() {
        guard let frame = camera.currentFrame else { return }
        frozenFrame = frame
        inferenceTimer?.invalidate()
        inferenceTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            runInference()
        }
    }

    private func runInference() {
        guard let frozen = frozenFrame,
              let live = camera.currentFrame,
              let p = predictor else { return }

        let pixelsA = cgImageToFloatRGB(frozen, size: imageSize)
        let pixelsB = cgImageToFloatRGB(live, size: imageSize)

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
