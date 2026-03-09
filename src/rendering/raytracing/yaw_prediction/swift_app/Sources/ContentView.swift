import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var frozenFrame: CGImage?
    @State private var predictedAngle: Double?
    @State private var predictor: OpaquePointer?
    @State private var inferenceTimer: Timer?
    @State private var showNativeResolution = false

    private let imageSize = 64

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
                                .overlay(Text("Camera starting...").foregroundColor(.secondary))
                        }
                    }
                    .frame(width: 240, height: 240)
                    .clipped()
                }
            }

            if let angle = predictedAngle {
                Text(String(format: "Predicted yaw: %.1f\u{00B0}", angle))
                    .font(.system(size: 28, weight: .bold, design: .monospaced))
            } else {
                Text("No prediction yet")
                    .font(.system(size: 28, weight: .bold, design: .monospaced))
                    .foregroundColor(.secondary)
            }

            HStack {
                Button("Capture Reference") {
                    captureFrame()
                }
                .keyboardShortcut(.space, modifiers: [])

                Toggle("64\u{00D7}64", isOn: $showNativeResolution)
            }
        }
        .padding()
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
            print("Usage: YawPredictor <path/to/yaw-predictor2.h5> [--fov <degrees>]")
            return
        }
        let h5Path = args[1]

        if let fovIdx = args.firstIndex(of: "--fov"), fovIdx + 1 < args.count,
           let fov = Double(args[fovIdx + 1]) {
            camera.horizontalFOV = fov
        }

        predictor = yaw_predictor_create(h5Path)
        if predictor == nil {
            print("Failed to load model from: \(h5Path)")
        }
    }

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
        inferenceTimer?.invalidate()
        inferenceTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            runInference()
        }
    }

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

    private func runInference() {
        guard let frozen = frozenFrame,
              let live = camera.currentFrame,
              let p = predictor else { return }

        let croppedLive = centerSquareCrop(live)
        let pixelsA = cgImageToFloatRGB(frozen, size: imageSize)
        let pixelsB = cgImageToFloatRGB(croppedLive, size: imageSize)

        let displacement: Float = pixelsA.withUnsafeBufferPointer { a in
            pixelsB.withUnsafeBufferPointer { b in
                yaw_predictor_evaluate(p, a.baseAddress, b.baseAddress)
            }
        }

        predictedAngle = Double(displacement) * camera.horizontalFOV / 2.0
    }
}

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
