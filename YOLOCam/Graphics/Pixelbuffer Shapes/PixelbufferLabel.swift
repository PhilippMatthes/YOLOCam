//
//    MIT License
//
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
import AppKit
import Vision

public struct PixelbufferLabel: PixelbufferRenderable {
    public let text: String
    
    public var font = NSFont(name: "Arial", size: 18)!
    public var color = CGColor(red: 1, green: 1, blue: 1, alpha: 1)
    
    public func render(into pixelBuffer: inout CVPixelBuffer, boundedBy bounds: CGRect) {
        pixelBuffer.modifyWithContext { context in
            let attributedString = NSAttributedString(
                string: text,
                attributes: [
                    NSAttributedString.Key.foregroundColor: CGColor(
                        red: 1, green: 1, blue: 1, alpha: 1
                    ),
                    NSAttributedString.Key.font: NSFont(name: "Arial", size: 24)!
                ]
            ) as CFAttributedString
                
            let frameSetter = CTFramesetterCreateWithAttributedString(attributedString)
            let framePath = CGMutablePath()
            framePath.addRect(bounds)
            let currentRange = CFRangeMake(0, 0)
            let frameRef = CTFramesetterCreateFrame(
                frameSetter,
                currentRange,
                framePath,
                nil
            )
            context.textMatrix = .identity
            CTFrameDraw(frameRef, context)
        }
    }
}
