//
//    MIT License
//
//    Copyright (c) 2020 John Boiles
//    Copyright (c) 2020 Ryohei Ikegami
//    Copyright (c) 2020 Philipp Matthes
//
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.

import Foundation
import AVFoundation
import AppKit
import Vision
import CoreText
import Accelerate
import VideoToolbox


class Stream: NSObject, Object {
    var objectID: CMIOObjectID = 0
    let name = "YOLO Cam"
    let width = 320
    let height = 240
    let frameRate = 30

    private var sequenceNumber: UInt64 = 0
    private var queueAlteredProc: CMIODeviceStreamQueueAlteredProc?
    private var queueAlteredRefCon: UnsafeMutableRawPointer?
    
    private lazy var captureSession: AVCaptureSession = {
        let session = AVCaptureSession()
        session.sessionPreset = .qvga320x240
        return session
    }()
    
    private lazy var visionModel = try! VNCoreMLModel(for: YOLOv3Tiny().model)

    private lazy var formatDescription: CMVideoFormatDescription? = {
        var formatDescription: CMVideoFormatDescription?
        let error = CMVideoFormatDescriptionCreate(
            allocator: kCFAllocatorDefault,
            codecType: kCVPixelFormatType_32ARGB,
            width: Int32(width), height: Int32(height),
            extensions: nil,
            formatDescriptionOut: &formatDescription)
        guard error == noErr else {
            log("CMVideoFormatDescriptionCreate Error: \(error)")
            return nil
        }
        return formatDescription
    }()

    private lazy var clock: CFTypeRef? = {
        var clock: Unmanaged<CFTypeRef>? = nil

        let error = CMIOStreamClockCreate(
            kCFAllocatorDefault,
            "YOLO Cam clock" as CFString,
            Unmanaged.passUnretained(self).toOpaque(),
            CMTimeMake(value: 1, timescale: 10),
            100, 10,
            &clock);
        guard error == noErr else {
            log("CMIOStreamClockCreate Error: \(error)")
            return nil
        }
        return clock?.takeUnretainedValue()
    }()

    private lazy var queue: CMSimpleQueue? = {
        var queue: CMSimpleQueue?
        let error = CMSimpleQueueCreate(
            allocator: kCFAllocatorDefault,
            capacity: 30,
            queueOut: &queue)
        guard error == noErr else {
            log("CMSimpleQueueCreate Error: \(error)")
            return nil
        }
        return queue
    }()

    lazy var properties: [Int : Property] = [
        kCMIOObjectPropertyName: Property(name),
        kCMIOStreamPropertyFormatDescription: Property(formatDescription!),
        kCMIOStreamPropertyFormatDescriptions: Property([formatDescription!] as CFArray),
        kCMIOStreamPropertyDirection: Property(UInt32(0)),
        kCMIOStreamPropertyFrameRate: Property(Float64(frameRate)),
        kCMIOStreamPropertyFrameRates: Property(Float64(frameRate)),
        kCMIOStreamPropertyMinimumFrameRate: Property(Float64(frameRate)),
        kCMIOStreamPropertyFrameRateRanges: Property(AudioValueRange(mMinimum: Float64(frameRate), mMaximum: Float64(frameRate))),
        kCMIOStreamPropertyClock: Property(CFTypeRefWrapper(ref: clock!)),
    ]

    func start() {
        let device = AVCaptureDevice.default(for: .video)!
        let input = try! AVCaptureDeviceInput(device: device)
        captureSession.addInput(input)
        captureSession.sessionPreset = .qvga320x240
        
        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: .main)
        output.alwaysDiscardsLateVideoFrames = true
        
        captureSession.addOutput(output)
        captureSession.startRunning()
    }

    func stop() {
        captureSession.stopRunning()
    }

    func copyBufferQueue(queueAlteredProc: CMIODeviceStreamQueueAlteredProc?, queueAlteredRefCon: UnsafeMutableRawPointer?) -> CMSimpleQueue? {
        self.queueAlteredProc = queueAlteredProc
        self.queueAlteredRefCon = queueAlteredRefCon
        return self.queue
    }
}

extension Stream: AVCaptureVideoDataOutputSampleBufferDelegate {
    func render(
        observations: [VNRecognizedObjectObservation],
        into pixelBuffer: inout CVPixelBuffer
    ) {
        pixelBuffer.modifyWithContext { [width, height] context in
            for observation in observations {
                let objectBounds = VNImageRectForNormalizedRect(
                    observation.boundingBox, Int(width), Int(height)
                )
                
                context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0.1))
                context.fill(objectBounds)
                
                let description = observation.labels
                    .prefix(3)
                    .map {"\($0.identifier.capitalized) \(Int($0.confidence * 100))%"}
                    .joined(separator: ", ")
                
                let attributedString = NSAttributedString(string: description, attributes: [
                    NSAttributedString.Key.foregroundColor: CGColor(red: 1, green: 1, blue: 1, alpha: 1),
                ]) as CFAttributedString
                let frameSetter = CTFramesetterCreateWithAttributedString(attributedString)
                let framePath = CGMutablePath()
                framePath.addRect(objectBounds)
                let currentRange = CFRangeMake(0, 0)
                let frameRef = CTFramesetterCreateFrame(frameSetter, currentRange, framePath, nil)
                context.textMatrix = .identity
                CTFrameDraw(frameRef, context)
            }
        }
    }
    
    func didReceive(yuvPixelBuffer: CVPixelBuffer) {
        guard
            let queue = queue,
            CMSimpleQueueGetCount(queue) < CMSimpleQueueGetCapacity(queue)
        else {return}
        
        let width = CVPixelBufferGetWidth(yuvPixelBuffer)
        let height = CVPixelBufferGetHeight(yuvPixelBuffer)
        
        DispatchQueue.main.async {
            // Convert pixel buffer to a format, on which we can draw shapes
            var pixelBuffer = CVPixelBuffer.create(size: .init(width: width, height: height))!
            var transferSession: VTPixelTransferSession?
            VTPixelTransferSessionCreate(allocator: kCFAllocatorDefault, pixelTransferSessionOut: &transferSession)
            VTPixelTransferSessionTransferImage(transferSession!, from: yuvPixelBuffer, to: pixelBuffer)

            let request = VNCoreMLRequest(model: self.visionModel, completionHandler: { (request, error) in
                guard let observations = request.results as? [VNRecognizedObjectObservation] else {return}
                
                self.render(observations: observations, into: &pixelBuffer)
                
                let currentTimeNsec = mach_absolute_time()

                var timing = CMSampleTimingInfo(
                    duration: CMTime(value: 1, timescale: CMTimeScale(self.frameRate)),
                    presentationTimeStamp: CMTime(value: CMTimeValue(currentTimeNsec), timescale: CMTimeScale(1000_000_000)),
                    decodeTimeStamp: .invalid
                )

                var error = noErr

                error = CMIOStreamClockPostTimingEvent(timing.presentationTimeStamp, currentTimeNsec, true, self.clock)
                guard error == noErr else {
                    log("CMSimpleQueueCreate Error: \(error)")
                    return
                }

                var formatDescription: CMFormatDescription?
                error = CMVideoFormatDescriptionCreateForImageBuffer(
                    allocator: kCFAllocatorDefault,
                    imageBuffer: pixelBuffer,
                    formatDescriptionOut: &formatDescription)
                guard error == noErr else {
                    log("CMVideoFormatDescriptionCreateForImageBuffer Error: \(error)")
                    return
                }

                var sampleBufferUnmanaged: Unmanaged<CMSampleBuffer>? = nil
                error = CMIOSampleBufferCreateForImageBuffer(
                    kCFAllocatorDefault,
                    pixelBuffer,
                    formatDescription,
                    &timing,
                    self.sequenceNumber,
                    UInt32(kCMIOSampleBufferNoDiscontinuities),
                    &sampleBufferUnmanaged
                )
                guard error == noErr else {
                    log("CMIOSampleBufferCreateForImageBuffer Error: \(error)")
                    return
                }

                CMSimpleQueueEnqueue(queue, element: sampleBufferUnmanaged!.toOpaque())
                self.queueAlteredProc?(self.objectID, sampleBufferUnmanaged!.toOpaque(), self.queueAlteredRefCon)

                self.sequenceNumber += 1
            })
            
            request.imageCropAndScaleOption = .scaleFill
            let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            try! requestHandler.perform([request])
        }
    }
    
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard
            let yuvPixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else {return}
        
        didReceive(yuvPixelBuffer: yuvPixelBuffer)
    }
}
