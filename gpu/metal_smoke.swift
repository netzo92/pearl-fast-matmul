import Foundation
import Metal

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal GPU") }
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void prefix_dot(device const char* a [[buffer(0)]],
                       device const char* b [[buffer(1)]],
                       device int* out [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
    if (index >= 256) return;
    uint i=index/32, j=index%32;
    int sum=0;
    for (uint chunk=0;chunk<16;chunk++) {
        for (uint s=chunk*128;s<(chunk+1)*128;s++)
            sum+=int(a[i*2048+s])*int(b[j*2048+s]);
        out[chunk*256+index]=sum;
    }
}
"""
let library = try device.makeLibrary(source: source, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "prefix_dot")!)
var a = [Int8](repeating:0,count:16384)
var b = [Int8](repeating:0,count:65536)
for i in a.indices { a[i] = Int8((i * 7) % 127 - 63) }
for i in b.indices { b[i] = Int8((i * 13) % 127 - 63) }
let ab = a.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
let bb = b.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
let output = device.makeBuffer(length: 16*256*MemoryLayout<Int32>.stride, options: .storageModeShared)!
let queue = device.makeCommandQueue()!
let command = queue.makeCommandBuffer()!
let encoder = command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(ab,offset:0,index:0); encoder.setBuffer(bb,offset:0,index:1); encoder.setBuffer(output,offset:0,index:2)
encoder.dispatchThreads(MTLSize(width:256,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:pipeline.threadExecutionWidth,height:1,depth:1))
encoder.endEncoding();command.commit();command.waitUntilCompleted()
if let error=command.error {fatalError("GPU command failed: \(error)")}
let actual=output.contents().bindMemory(to:Int32.self,capacity:16*256)
for i in 0..<8 {for j in 0..<32 {
    var sum:Int32=0
    for chunk in 0..<16 {
        for s in chunk*128..<(chunk+1)*128 {sum+=Int32(a[i*2048+s])*Int32(b[j*2048+s])}
        precondition(actual[chunk*256+i*32+j]==sum,"Exact prefix mismatch")
    }
}}
print("Metal device: \(device.name)")
print("PASS: 4096 integer prefix entries match CPU exactly (8x32 tile, k2048, chunk128)")
print("Synthetic operands; GPU execution check only, no Pearl noise/verifier or performance claim")
