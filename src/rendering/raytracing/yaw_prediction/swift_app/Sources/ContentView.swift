import SwiftUI
#if os(iOS)
import UniformTypeIdentifiers
import simd
#endif

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @State private var frozenFrame: CGImage?
    @State private var prediction: (px: Double, py: Double, roll: Double)?
    @State private var predictor: OpaquePointer?
    @State private var inferenceTimer: Timer?
    @State private var showNativeResolution = false
    @State private var modelLoaded = false
    #if os(iOS)
    @State private var referenceTransform: simd_float4x4?
    @State private var showFilePicker = false
    #endif

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
                                .overlay(Text(camera.trackingReady ? "Camera starting..." : "Initializing AR...").foregroundColor(.secondary))
                        }
                    }
                    .frame(width: 240, height: 240)
                    .clipped()
                }
            }

            VStack(spacing: 4) {
                if let pred = prediction {
                    let hfov = camera.horizontalFOV
                    let pxDeg = pred.px * hfov / 2.0
                    let pyDeg = pred.py * hfov / 2.0
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
                    Text(String(format: "GT H: %+.1f\u{00B0}  V: %+.1f\u{00B0}  R: %+.1f\u{00B0}", gt.px, gt.py, gt.roll))
                        .font(.system(size: 18, weight: .bold, design: .monospaced))
                        .foregroundColor(.blue)
                    if let pred = prediction {
                        let hfov = camera.horizontalFOV
                        let errH = pred.px * hfov / 2.0 - gt.px
                        let errV = pred.py * hfov / 2.0 - gt.py
                        let errR = pred.roll * 180.0 - gt.roll
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
        .fileImporter(isPresented: $showFilePicker,
                      allowedContentTypes: [UTType(filenameExtension: "h5") ?? .data],
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
        if let url = Bundle.main.url(forResource: "yaw-predictor4", withExtension: "tar") {
            loadModel(from: url.path)
        } else {
            print("Bundled model not found")
        }
    }
    #endif

    private func loadModel(from path: String) {
        if let p = predictor {
            yaw_predictor_destroy(p)
            predictor = nil
        }
        predictor = yaw_predictor_create(path)
        modelLoaded = predictor != nil
        if !modelLoaded {
            print("Failed to load model from: \(path)")
        }
    }

    #if os(iOS)
    private func computeGroundTruth() -> (px: Double, py: Double, roll: Double)? {
        guard let refT = referenceTransform, let curT = camera.currentTransform else { return nil }
        let rel = simd_inverse(refT) * curT
        // Camera B's forward (-Z) projected into camera A's frame
        // z2 = -rel.columns.2 (third column negated = forward direction)
        let z2x = -rel.columns.2.x
        let z2y = -rel.columns.2.y
        let z2z = -rel.columns.2.z
        let hfov = camera.horizontalFOV * .pi / 180.0
        let tanHalfH = tan(hfov / 2.0)
        let tanHalfV = tanHalfH // square crop => same fov
        let px = Double(z2x / (z2z * Float(tanHalfH))) * hfov / 2.0 * 180.0 / .pi
        let py = Double(z2y / (z2z * Float(tanHalfV))) * hfov / 2.0 * 180.0 / .pi
        // Roll: rotation around forward axis
        // Shortest-arc decomposition: align (0,0,1) to z2, then extract remaining roll
        // Simplified: use atan2 on the rotated x-axis projected onto the plane perpendicular to z2
        let re00 = rel.columns.0.x
        let re10 = rel.columns.1.x
        let roll = Double(atan2(re10, re00)) * 180.0 / .pi
        return (-py, px, -roll)
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
