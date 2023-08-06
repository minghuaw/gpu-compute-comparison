//
//  main.swift
//  metal-compute
//
//  Created by Minghua Wu on 2023-08-04.
//

import Foundation
import MetalKit

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

let defaultLibrary = device.makeDefaultLibrary()!
let shader = defaultLibrary.makeFunction(name: "add")

let commandBuffer = commandQueue.makeCommandBuffer()!
let encoder = commandBuffer.makeComputeCommandEncoder()!
encoder.setComputePipelineState(try device.makeComputePipelineState(function: defaultLibrary.makeFunction(name: "add")!))

let left: [Float] = [1.0, 2.0];
let leftBuffer = device.makeBuffer(bytes: left as [Float], length: MemoryLayout<Float>.stride * left.count, options: [])
encoder.setBuffer(leftBuffer, offset: 0, index: 0)

let right: [Float] = [3.0, 4.0];
let rightBuffer = device.makeBuffer(bytes: right as [Float], length: MemoryLayout<Float>.stride * right.count, options: [])
encoder.setBuffer(rightBuffer, offset: 0, index: 1)

let outputBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * left.count, options: [])!
encoder.setBuffer(outputBuffer, offset: 0, index: 2)

let numThreadGroups = MTLSize(width: 1, height: 1, depth: 1)
let threadsPerThreadGroup = MTLSize(width: 2, height: 1, depth: 1)
encoder.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerThreadGroup)
encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

//let result = outputBuffer.contents().load(as: [Float].self)
let ptr = outputBuffer.contents();
let length = left.count;
let array_ptr = ptr.bindMemory(to: Float.self, capacity: length)
let buffer = UnsafeBufferPointer(start: array_ptr, count: length)
let output = Array(buffer)
print("output: ", output)
