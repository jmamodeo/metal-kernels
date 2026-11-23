// SPDX-License-Identifier: Apache-2.0
// Times the Swift host of flash_attention_hd32 at the SAM 3 encoder shape,
// [1, 8, 5184, 32], no bias, default scale.
//
//   swift run -c release FlashAttentionHD32Bench [--warmup N] [--iterations N] [--dtype bf16|fp32|all]
//
// Each warmup and timed iteration is one launch in its own command buffer,
// timed from commit to waitUntilCompleted, so samples are per-call GPU wall
// time like bench/flash_attention_hd32.py. Prints median, min, p10 and p90 ms
// per call and TFLOP/s at the median. Nothing is recorded.
import Foundation
import MetalKernels

#if canImport(Metal)
import Metal
#endif

let shape = [1, 8, 5184, 32]
// Defaults match bench/flash_attention_hd32.py.
var warmup = 12
var iterations = 102
var dtypes = FlashAttentionDType.allCases

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("FlashAttentionHD32Bench: \(message)\n".utf8))
    exit(2)
}

var arguments = CommandLine.arguments.dropFirst()
while let flag = arguments.popFirst() {
    guard let value = arguments.popFirst() else { fail("\(flag) needs a value") }
    switch flag {
    case "--warmup":
        guard let n = Int(value), n >= 0 else { fail("--warmup must be a non-negative integer") }
        warmup = n
    case "--iterations":
        guard let n = Int(value), n >= 1 else { fail("--iterations must be a positive integer") }
        iterations = n
    case "--dtype":
        switch value {
        case "bf16": dtypes = [.bfloat16]
        case "fp32": dtypes = [.float32]
        case "all": dtypes = FlashAttentionDType.allCases
        default: fail("--dtype must be bf16, fp32 or all")
        }
    default:
        fail("unknown argument \(flag)")
    }
}

#if canImport(Metal)
func run(warmup: Int, iterations: Int, dtypes: [FlashAttentionDType]) throws {
    let kernel = try FlashAttentionHD32Kernel()
    let geometry = try FlashAttentionHD32.launchGeometry(q: shape, k: shape, v: shape)
    let flops = FlashAttentionHD32.flopsPerCall(geometry)
    print("device: \(kernel.device.name)")
    print("shape: \(shape), warmup \(warmup), timed \(iterations)")
    print("dtype      median ms     min ms     p10 ms     p90 ms   TFLOP/s")

    // Deterministic inputs in [-1, 1), rounded to bf16 so both dtypes see the
    // same values. q, k and v have the same shape here.
    var state: UInt64 = 0x5EED
    let values = (0..<geometry.queryElements).map { _ -> Float in
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        let uniform = Float(state >> 40) / Float(1 << 24) * 2 - 1
        return BFloat16Bits.toFloat(BFloat16Bits.from(uniform))
    }

    for dtype in dtypes {
        let bytes = geometry.queryElements * dtype.elementSize
        func buffer(_ data: [Float]?) throws -> MTLBuffer {
            let made: MTLBuffer?
            switch (dtype, data) {
            case (_, nil):
                made = kernel.device.makeBuffer(length: bytes, options: .storageModeShared)
            case (.float32, let data?):
                made = data.withUnsafeBytes {
                    kernel.device.makeBuffer(
                        bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
                }
            case (.bfloat16, let data?):
                made = data.map(BFloat16Bits.from).withUnsafeBytes {
                    kernel.device.makeBuffer(
                        bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
                }
            }
            guard let made else { throw FlashAttentionError.metal("Could not allocate a buffer") }
            return made
        }
        // k and v are rotations of q's values.
        let q = try buffer(values)
        let k = try buffer(Array(values[1...] + values[..<1]))
        let v = try buffer(Array(values[2...] + values[..<2]))
        let out = try buffer(nil)

        func launch() throws -> Double {
            guard let commandBuffer = kernel.commandQueue.makeCommandBuffer(),
                let encoder = commandBuffer.makeComputeCommandEncoder()
            else {
                throw FlashAttentionError.metal("Could not create a Metal command encoder")
            }
            try kernel.encode(
                into: encoder, q: q, k: k, v: v, out: out, qShape: shape, kShape: shape,
                dtype: dtype)
            encoder.endEncoding()
            let start = DispatchTime.now().uptimeNanoseconds
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            if commandBuffer.status == .error {
                let reason = commandBuffer.error.map { "\($0)" } ?? "unknown error"
                throw FlashAttentionError.metal("flash_attention_hd32 failed: \(reason)")
            }
            return Double(elapsed) / 1e6
        }

        for _ in 0..<warmup {
            _ = try launch()
        }
        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            samples.append(try launch())
        }
        let summary = FlashAttentionHD32.TimingSummary(samples: samples)
        print(
            dtype.rawValue.padding(toLength: 10, withPad: " ", startingAt: 0)
                + String(
                    format: " %10.3f %10.3f %10.3f %10.3f %9.2f",
                    summary.medianMs, summary.minMs, summary.p10Ms, summary.p90Ms,
                    summary.tflops(flops: flops)))
    }
}

do {
    try run(warmup: warmup, iterations: iterations, dtypes: dtypes)
} catch {
    FileHandle.standardError.write(Data("FlashAttentionHD32Bench: \(error)\n".utf8))
    exit(1)
}
#else
fail("needs Metal; run it on macOS with Apple silicon")
#endif
