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
    let width = 1280
    let height = 720
    let frameRate = 30

    private var mostRecentlyEnqueuedVNRequest: VNRequest?
    private let dispatchSemaphore = DispatchSemaphore(value: 1)
    private var sequenceNumber: UInt64 = 0
    private var queueAlteredProc: CMIODeviceStreamQueueAlteredProc?
    private var queueAlteredRefCon: UnsafeMutableRawPointer?
    
    private lazy var capture = VideoCapture()
    
    private lazy var visionModel = try! VNCoreMLModel(for: YOLOv3TinyInt8LUT().model)

    private lazy var formatDescription: CMVideoFormatDescription? = {
        var formatDescription: CMVideoFormatDescription?
        guard CMVideoFormatDescriptionCreate(
            allocator: kCFAllocatorDefault,
            codecType: kCVPixelFormatType_32ARGB,
            width: Int32(width),
            height: Int32(height),
            extensions: nil,
            formatDescriptionOut: &formatDescription
        ) == noErr else {return nil}
        return formatDescription
    }()

    private lazy var clock: CFTypeRef? = {
        var clock: Unmanaged<CFTypeRef>? = nil
        guard CMIOStreamClockCreate(
            kCFAllocatorDefault,
            "YOLO Cam clock" as CFString,
            Unmanaged.passUnretained(self).toOpaque(),
            CMTimeMake(value: 1, timescale: 10),
            100,
            10,
            &clock
        ) == noErr else {return nil}
        return clock?.takeUnretainedValue()
    }()

    private lazy var queue: CMSimpleQueue? = {
        var queue: CMSimpleQueue?
        guard CMSimpleQueueCreate(
            allocator: kCFAllocatorDefault,
            capacity: 30,
            queueOut: &queue
        ) == noErr else {return nil}
        return queue
    }()

    lazy var properties: [Int : Property] = [
        kCMIOObjectPropertyName: Property(name),
        kCMIOStreamPropertyFormatDescription: Property(formatDescription!),
        kCMIOStreamPropertyFormatDescriptions: Property([formatDescription!] as CFArray),
        kCMIOStreamPropertyDirection: Property(UInt32(0)),
        kCMIOStreamPropertyFrameRate: Property(Float64(frameRate)),
        kCMIOStreamPropertyFrameRates: Property(Float64(frameRate)),
        kCMIOStreamPropertyMinimumFrameRate: Property(Float64(0)),
        kCMIOStreamPropertyFrameRateRanges: Property(AudioValueRange(
            mMinimum: Float64(0), mMaximum: Float64(frameRate)
        )),
        kCMIOStreamPropertyClock: Property(CFTypeRefWrapper(ref: clock!)),
    ]

    func start() {
        capture.delegate = self
        
        capture.setUp()
        capture.start()
    }

    func stop() {
        capture.stop()
    }

    func copyBufferQueue(
        queueAlteredProc: CMIODeviceStreamQueueAlteredProc?,
        queueAlteredRefCon: UnsafeMutableRawPointer?
    ) -> CMSimpleQueue? {
        self.queueAlteredProc = queueAlteredProc
        self.queueAlteredRefCon = queueAlteredRefCon
        return self.queue
    }
}

extension Stream: VideoCaptureDelegate {
    func videoCapture(
        _ capture: VideoCapture,
        didCapture pixelBuffer: CVPixelBuffer?,
        with sampleTimingInfo: CMSampleTimingInfo
    ) {
        guard
            var pixelBuffer = pixelBuffer,
            let queue = queue,
            CMSimpleQueueGetCount(queue) < CMSimpleQueueGetCapacity(queue)
        else {return}
        
        DispatchQueue.global(qos: .userInteractive).async {
            let request = VNCoreMLRequest(model: self.visionModel) { (request, error) in
                self.dispatchSemaphore.signal()
                
                guard let observations = request.results
                    as? [VNRecognizedObjectObservation] else {return}
                
                for observation in observations {
                    let renderer = ObservationRenderer(observation: observation)
                    renderer.render(into: &pixelBuffer)
                }

                let currentTimeNsec = mach_absolute_time()
                var timing = sampleTimingInfo
                
                guard CMIOStreamClockPostTimingEvent(
                    timing.presentationTimeStamp,
                    currentTimeNsec,
                    true,
                    self.clock
                ) == noErr else {return}

                var formatDescription: CMFormatDescription?
                guard CMVideoFormatDescriptionCreateForImageBuffer(
                    allocator: kCFAllocatorDefault,
                    imageBuffer: pixelBuffer,
                    formatDescriptionOut: &formatDescription
                ) == noErr else {return}

                var sampleBufferUnmanaged: Unmanaged<CMSampleBuffer>? = nil
                guard CMIOSampleBufferCreateForImageBuffer(
                    kCFAllocatorDefault,
                    pixelBuffer,
                    formatDescription,
                    &timing,
                    self.sequenceNumber,
                    UInt32(kCMIOSampleBufferNoDiscontinuities),
                    &sampleBufferUnmanaged
                ) == noErr else {return}

                CMSimpleQueueEnqueue(queue, element: sampleBufferUnmanaged!.toOpaque())
                self.queueAlteredProc?(
                    self.objectID,
                    sampleBufferUnmanaged!.toOpaque(),
                    self.queueAlteredRefCon
                )

                self.sequenceNumber += 1
            }
            request.imageCropAndScaleOption = .scaleFill
            
            let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            
            // Ensure, that only one request is enqueued and that
            // only the most recent request gets executed
            self.mostRecentlyEnqueuedVNRequest = request
            self.dispatchSemaphore.wait()
            guard request == self.mostRecentlyEnqueuedVNRequest else {
                self.dispatchSemaphore.signal()
                return
            }
            
            try! requestHandler.perform([request])
        }
    }
}
