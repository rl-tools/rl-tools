import AVFoundation
import CoreImage

final class CameraManager: NSObject, ObservableObject {
    @Published var currentFrame: CGImage?
    @Published var horizontalFOV: Double = 65.0

    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "camera.queue")
    private let context = CIContext()

    func start() {
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

        #if os(iOS)
        if let connection = output.connection(with: .video) {
            if #available(iOS 17.0, *) {
                connection.videoRotationAngle = 90
            } else {
                connection.videoOrientation = .portrait
            }
            if camera.position == .front {
                connection.isVideoMirrored = true
            }
        }
        #endif

        queue.async {
            self.session.startRunning()
        }
    }

    func stop() {
        session.stopRunning()
    }
}

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
