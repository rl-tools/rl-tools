import AVFoundation
import CoreImage
#if os(iOS)
import ARKit
#endif

final class CameraManager: NSObject, ObservableObject {
    @Published var currentFrame: CGImage?
    @Published var horizontalFOV: Double = 65.0
    @Published var currentYaw: Double?
    @Published var trackingReady = false

    private let context = CIContext()

    #if os(iOS)
    private let arSession = ARSession()
    #else
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera.queue")
    #endif

    func start() {
        #if os(iOS)
        arSession.delegate = self
        let config = ARWorldTrackingConfiguration()
        config.isAutoFocusEnabled = true
        arSession.run(config)
        #else
        session.sessionPreset = .medium
        guard let camera = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            print("No camera available")
            return
        }
        if session.canAddInput(input) {
            session.addInput(input)
        }
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        queue.async {
            self.session.startRunning()
        }
        #endif
    }

    func stop() {
        #if os(iOS)
        arSession.pause()
        #else
        session.stopRunning()
        #endif
    }
}

#if os(iOS)
extension CameraManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let tracking = frame.camera.trackingState
        let transform = frame.camera.transform
        let yaw = Double(-atan2(transform.columns.2.x, transform.columns.2.z))

        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            .oriented(.right)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }

        DispatchQueue.main.async {
            if case .normal = tracking {
                self.trackingReady = true
                self.currentYaw = yaw
            }
            if self.trackingReady {
                self.currentFrame = cgImage
            }
        }
    }
}
#endif

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        #if os(macOS)
        let mirrored = ciImage.transformed(by: CGAffineTransform(scaleX: -1, y: 1)
            .translatedBy(x: -ciImage.extent.width, y: 0))
        guard let cgImage = context.createCGImage(mirrored, from: mirrored.extent) else { return }
        #else
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        #endif

        DispatchQueue.main.async {
            self.currentFrame = cgImage
        }
    }
}
