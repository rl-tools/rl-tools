import Foundation
#if os(iOS)
import ImageIO
import MobileCoreServices
import simd
import UniformTypeIdentifiers

struct AttitudeGroundTruth {
    let horizontalDegrees: Double
    let verticalDegrees: Double
    let rollDegrees: Double
}

struct DatasetStatus {
    let translationDriftMeters: Double
    let lastCaptureAgeSeconds: Double?
    let sampleCount: Int
    let isTranslationAcceptable: Bool
}

final class DatasetRecorder {
    struct SessionPaths {
        let root: URL
        let framesDirectory: URL
        let metadata: URL
        let summary: URL
    }

    private struct FrameRecord: Encodable {
        let index: Int
        let timestamp: Double
        let image: String
        let translationDriftMeters: Double
        let translationFromSessionOrigin: [Double]
        let worldTransform: [[Double]]
    }

    private struct SessionSummary: Encodable {
        let createdAt: String
        let imageSize: Int
        let horizontalFOVDegrees: Double
        let maxTranslationDriftMeters: Double
        let minAngularSeparationDegrees: Double
        let minSampleSpacingSeconds: Double
        let framesDirectory: String
        let metadata: String
        let frameCount: Int
    }

    let imageSize: Int
    let maxTranslationDriftMeters: Double
    let minAngularSeparationDegrees: Double
    let minSampleSpacingSeconds: Double

    private(set) var sessionPaths: SessionPaths?
    private(set) var sessionOriginTransform: simd_float4x4?
    private(set) var sampleCount = 0

    private var lastSavedTransform: simd_float4x4?
    private var lastSavedTimestamp: TimeInterval?

    init(
        imageSize: Int,
        maxTranslationDriftMeters: Double = 0.05,
        minAngularSeparationDegrees: Double = 3.0,
        minSampleSpacingSeconds: Double = 0.35
    ) {
        self.imageSize = imageSize
        self.maxTranslationDriftMeters = maxTranslationDriftMeters
        self.minAngularSeparationDegrees = minAngularSeparationDegrees
        self.minSampleSpacingSeconds = minSampleSpacingSeconds
    }

    func start(frame: CapturedARFrame, horizontalFOVDegrees: Double) throws {
        let fm = FileManager.default
        let root = try Self.makeSessionRoot()
        let frames = root.appendingPathComponent("frames", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        try fm.createDirectory(at: frames, withIntermediateDirectories: true)

        let metadata = root.appendingPathComponent("frames.jsonl")
        fm.createFile(atPath: metadata.path, contents: nil)

        sessionPaths = SessionPaths(
            root: root,
            framesDirectory: frames,
            metadata: metadata,
            summary: root.appendingPathComponent("session.json")
        )
        sessionOriginTransform = frame.transform
        lastSavedTransform = nil
        lastSavedTimestamp = nil
        sampleCount = 0

        _ = try saveFrame(frame, translationDriftMeters: 0.0, translationFromSessionOrigin: .zero)
        try writeSummary(horizontalFOVDegrees: horizontalFOVDegrees)
    }

    func stop(horizontalFOVDegrees: Double) throws -> URL? {
        guard let root = sessionPaths?.root else { return nil }
        try writeSummary(horizontalFOVDegrees: horizontalFOVDegrees)
        return root
    }

    func status(for frame: CapturedARFrame) -> DatasetStatus? {
        guard let sessionOriginTransform else { return nil }
        let rel = simd_inverse(sessionOriginTransform) * frame.transform
        let translation = translationVector(rel)
        let drift = simd_length(translation)
        let age = lastSavedTimestamp.map { frame.timestamp - $0 }
        return DatasetStatus(
            translationDriftMeters: drift,
            lastCaptureAgeSeconds: age,
            sampleCount: sampleCount,
            isTranslationAcceptable: drift <= maxTranslationDriftMeters
        )
    }

    func considerSample(frame: CapturedARFrame, horizontalFOVDegrees: Double) throws -> Bool {
        guard let sessionOriginTransform else { return false }
        let rel = simd_inverse(sessionOriginTransform) * frame.transform
        let translation = translationVector(rel)
        let drift = simd_length(translation)
        guard drift <= maxTranslationDriftMeters else { return false }

        if let lastSavedTimestamp, frame.timestamp - lastSavedTimestamp < minSampleSpacingSeconds {
            return false
        }

        if let lastSavedTransform {
            let delta = simd_inverse(lastSavedTransform) * frame.transform
            let angularSeparation = rotationAngleDegrees(delta)
            guard angularSeparation >= minAngularSeparationDegrees else { return false }
        }

        _ = try saveFrame(frame, translationDriftMeters: drift, translationFromSessionOrigin: translation)
        try writeSummary(horizontalFOVDegrees: horizontalFOVDegrees)
        return true
    }

    private func writeSummary(horizontalFOVDegrees: Double) throws {
        guard let paths = sessionPaths else { return }
        let summary = SessionSummary(
            createdAt: ISO8601DateFormatter().string(from: Date()),
            imageSize: imageSize,
            horizontalFOVDegrees: horizontalFOVDegrees,
            maxTranslationDriftMeters: maxTranslationDriftMeters,
            minAngularSeparationDegrees: minAngularSeparationDegrees,
            minSampleSpacingSeconds: minSampleSpacingSeconds,
            framesDirectory: paths.framesDirectory.lastPathComponent,
            metadata: paths.metadata.lastPathComponent,
            frameCount: sampleCount
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(summary)
        try data.write(to: paths.summary, options: .atomic)
    }

    private static func makeSessionRoot() throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let name = "attitude-dataset-\(formatter.string(from: Date()))"
        return docs.appendingPathComponent(name, isDirectory: true)
    }

    private static func appendJSONLine<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(value)
        guard let handle = try? FileHandle(forWritingTo: url) else {
            throw CocoaError(.fileWriteUnknown)
        }
        defer { try? handle.close() }
        try handle.seekToEnd()
        try handle.write(contentsOf: data)
        try handle.write(contentsOf: Data([0x0a]))
    }

    private static func writeJPEG(_ image: CGImage, to url: URL, quality: CGFloat) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        let options: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: quality
        ]
        CGImageDestinationAddImage(destination, image, options as CFDictionary)
        if !CGImageDestinationFinalize(destination) {
            throw CocoaError(.fileWriteUnknown)
        }
    }

    private func saveFrame(
        _ frame: CapturedARFrame,
        translationDriftMeters: Double,
        translationFromSessionOrigin: simd_double3
    ) throws -> Int {
        guard let paths = sessionPaths else { return sampleCount }
        let index = sampleCount
        let filename = String(format: "frame_%05d.jpg", index)
        let imageURL = paths.framesDirectory.appendingPathComponent(filename)
        try Self.writeJPEG(centerSquareCrop(frame.image), to: imageURL, quality: 0.9)

        let record = FrameRecord(
            index: index,
            timestamp: frame.timestamp,
            image: "frames/\(filename)",
            translationDriftMeters: translationDriftMeters,
            translationFromSessionOrigin: [translationFromSessionOrigin.x, translationFromSessionOrigin.y, translationFromSessionOrigin.z],
            worldTransform: matrixRows(frame.transform)
        )
        try Self.appendJSONLine(record, to: paths.metadata)

        sampleCount += 1
        lastSavedTransform = frame.transform
        lastSavedTimestamp = frame.timestamp
        return index
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

func computeGroundTruthDegrees(
    reference: simd_float4x4,
    current: simd_float4x4,
    horizontalFOVDegrees: Double
) -> AttitudeGroundTruth? {
    let rel = simd_inverse(reference) * current
    let z2x = -rel.columns.2.x
    let z2y = -rel.columns.2.y
    let z2z = -rel.columns.2.z
    guard abs(z2z) > 1e-6 else { return nil }
    let hfov = horizontalFOVDegrees * .pi / 180.0
    let tanHalfH = tan(hfov / 2.0)
    let tanHalfV = tanHalfH
    let px = normalizedDisplacementToDegrees(Double(z2x / (z2z * Float(tanHalfH))), fovDegrees: horizontalFOVDegrees)
    let py = normalizedDisplacementToDegrees(Double(z2y / (z2z * Float(tanHalfV))), fovDegrees: horizontalFOVDegrees)
    let re00 = rel.columns.0.x
    let re10 = rel.columns.1.x
    let roll = Double(atan2(re10, re00)) * 180.0 / .pi
    return AttitudeGroundTruth(horizontalDegrees: -py, verticalDegrees: px, rollDegrees: -roll)
}

func normalizedDisplacementToDegrees(_ value: Double, fovDegrees: Double) -> Double {
    let halfFOVRadians = fovDegrees * .pi / 360.0
    return atan(value * tan(halfFOVRadians)) * 180.0 / .pi
}

private func matrixRows(_ matrix: simd_float4x4) -> [[Double]] {
    [
        [Double(matrix.columns.0.x), Double(matrix.columns.1.x), Double(matrix.columns.2.x), Double(matrix.columns.3.x)],
        [Double(matrix.columns.0.y), Double(matrix.columns.1.y), Double(matrix.columns.2.y), Double(matrix.columns.3.y)],
        [Double(matrix.columns.0.z), Double(matrix.columns.1.z), Double(matrix.columns.2.z), Double(matrix.columns.3.z)],
        [Double(matrix.columns.0.w), Double(matrix.columns.1.w), Double(matrix.columns.2.w), Double(matrix.columns.3.w)]
    ]
}

private func translationVector(_ transform: simd_float4x4) -> simd_double3 {
    simd_make_double3(
        Double(transform.columns.3.x),
        Double(transform.columns.3.y),
        Double(transform.columns.3.z)
    )
}

private func rotationAngleDegrees(_ transform: simd_float4x4) -> Double {
    let rotation = simd_quatf(transform)
    let clamped = max(-1.0, min(1.0, Double(rotation.real)))
    return 2.0 * acos(clamped) * 180.0 / .pi
}
#endif
