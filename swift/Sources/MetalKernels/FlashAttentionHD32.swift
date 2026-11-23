// SPDX-License-Identifier: Apache-2.0
import Foundation

#if canImport(Metal)
import Metal
#endif

/// Failure while validating or launching `flash_attention_hd32`.
public enum FlashAttentionError: Error, Equatable, CustomStringConvertible, Sendable {
    /// A shape, length, dtype or scale the kernel does not accept.
    case invalidArgument(String)
    /// No Metal device, or the device failed to compile or run the shader.
    case metal(String)

    public var description: String {
        switch self {
        case .invalidArgument(let message):
            return message
        case .metal(let message):
            return message
        }
    }
}

/// Element type of the `q`, `k`, `v`, `bias` and `out` buffers.
public enum FlashAttentionDType: String, CaseIterable, Sendable {
    /// bfloat16, stored as `UInt16` bits.
    case bfloat16
    /// IEEE float32.
    case float32

    /// Bytes per element.
    public var elementSize: Int {
        switch self {
        case .bfloat16: return 2
        case .float32: return 4
        }
    }

    /// Entry point in `kernels/flash_attention_hd32/flash_attention_hd32.metal`.
    public var functionName: String {
        switch self {
        case .bfloat16: return "flash_attention_hd32_bfloat"
        case .float32: return "flash_attention_hd32_float"
        }
    }
}

/// bfloat16 bit conversions that match the shader.
public enum BFloat16Bits {
    /// fp32 -> bf16 bits, round to nearest even. NaN stays NaN. This is the
    /// rounding `fa_hd32_store` applies to a bf16 output.
    public static func from(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        if value.isNaN {
            return UInt16(truncatingIfNeeded: bits >> 16) | 0x0040
        }
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    /// bf16 bits -> fp32. Exact: the bits become the high half of the word.
    public static func toFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }
}

/// Launch rules for FlashAttention head dim 32.
///
/// These mirror `metal_kernels.flash_attention` and the constants in the
/// shader. They need no GPU, so they also build and run off macOS.
public enum FlashAttentionHD32 {
    /// Kernel directory and `.metal` file name under `kernels/`.
    public static let kernelName = "flash_attention_hd32"
    public static let headDim = 32
    /// Mirrors `FA_HD32_BLOCK_Q`: query rows per threadgroup.
    public static let queryBlock = 32
    /// Mirrors `FA_HD32_BLOCK_K`: keys per K/V block.
    public static let keyBlock = 64
    /// Mirrors `FA_HD32_THREADS`: 8 query rows per simdgroup of 32 threads.
    public static let threadgroupSize = queryBlock / 8 * 32
    /// One 64-key K block and one V block staged in bf16 (fp32 inputs stage
    /// twice this). Mirrors `FA_HD32_KV_ELEMENTS`. The shader declares the
    /// stage itself, so the host does not set threadgroup memory.
    public static let threadgroupMemoryBytes = 2 * keyBlock * headDim * 2
    /// Mirrors `FA_HD32_DIM_COUNT`: length of the int32 `dims` buffer.
    public static let dimCount = 9
    /// Buffer offsets in the shader are 32-bit.
    public static let maxElements = Int(Int32.max)

    /// Validated shapes and the 1-D launch grid.
    public struct Geometry: Equatable, Sendable {
        public let batch: Int
        public let heads: Int
        public let queryLength: Int
        public let keyLength: Int
        /// One threadgroup per `queryBlock` query rows of each (batch, head).
        public let threadgroups: Int

        /// `batch * heads * queryLength`.
        public var rows: Int { batch * heads * queryLength }
        /// Grid size in threads.
        public var threads: Int { threadgroups * FlashAttentionHD32.threadgroupSize }
        /// Elements in `q` and `out`.
        public var queryElements: Int { rows * FlashAttentionHD32.headDim }
        /// Elements in `k` and in `v`.
        public var keyElements: Int { batch * heads * keyLength * FlashAttentionHD32.headDim }
        /// `(batch, heads, queryLength, keyLength)`, the shape a bias broadcasts to.
        public var scoreShape: [Int] { [batch, heads, queryLength, keyLength] }
    }

    /// Validate `(batch, heads, sequence, 32)` shapes for `q`, `k` and `v`.
    public static func launchGeometry(q: [Int], k: [Int], v: [Int]) throws -> Geometry {
        guard q.count == 4, k.count == 4, v.count == 4 else {
            throw FlashAttentionError.invalidArgument(
                "q, k, and v must have shape (batch, heads, sequence, 32)")
        }
        let (batch, heads, queryLength) = (q[0], q[1], q[2])
        guard q[3] == headDim, k[3] == headDim, v[3] == headDim else {
            throw FlashAttentionError.invalidArgument(
                "q, k, and v must have head dimension \(headDim)")
        }
        guard k[0] == batch, k[1] == heads, v[0] == batch, v[1] == heads else {
            throw FlashAttentionError.invalidArgument(
                "q, k, and v batch and head dimensions must match")
        }
        guard k[2] == v[2] else {
            throw FlashAttentionError.invalidArgument("k and v sequence lengths must match")
        }
        let keyLength = k[2]
        guard min(batch, heads, queryLength, keyLength) >= 1 else {
            throw FlashAttentionError.invalidArgument(
                "batch, heads, and sequence lengths must be positive")
        }
        guard let queryElements = product([batch, heads, queryLength, headDim]),
            let keyElements = product([batch, heads, keyLength, headDim]),
            max(queryElements, keyElements) <= maxElements
        else {
            throw FlashAttentionError.invalidArgument(
                "inputs must have at most \(maxElements) elements")
        }
        let queryBlocks = (queryLength + queryBlock - 1) / queryBlock
        guard let threads = product([batch, heads, queryBlocks, threadgroupSize]),
            threads <= maxElements
        else {
            throw FlashAttentionError.invalidArgument(
                "the launch grid must have at most \(maxElements) threads")
        }
        return Geometry(
            batch: batch,
            heads: heads,
            queryLength: queryLength,
            keyLength: keyLength,
            threadgroups: batch * heads * queryBlocks
        )
    }

    /// Validate a bias shape against `target` (`Geometry.scoreShape`).
    ///
    /// Returns `shape`, the bias shape left-padded with ones to four
    /// dimensions, and the element `strides` of a contiguous bias of that
    /// shape, with 0 on broadcast dimensions.
    public static func biasStrides(
        biasShape: [Int], target: [Int]
    ) throws -> (shape: [Int], strides: [Int]) {
        let message = "bias must be broadcastable to \(target); got \(biasShape)"
        guard target.count == 4, biasShape.count <= 4 else {
            throw FlashAttentionError.invalidArgument(message)
        }
        let shape = Array(repeating: 1, count: 4 - biasShape.count) + biasShape
        for (size, full) in zip(shape, target) where size != 1 && size != full {
            throw FlashAttentionError.invalidArgument(message)
        }
        guard let elements = product(shape), elements <= maxElements else {
            throw FlashAttentionError.invalidArgument(
                "bias must have at most \(maxElements) elements")
        }
        var strides = [Int](repeating: 0, count: 4)
        var running = 1
        for axis in stride(from: 3, through: 0, by: -1) {
            strides[axis] = shape[axis] == target[axis] && target[axis] > 1 ? running : 0
            running *= shape[axis]
        }
        return (shape, strides)
    }

    /// `scale`, defaulting to `1 / sqrt(32)` like the reference.
    public static func resolveScale(_ scale: Float?) throws -> Float {
        guard let scale else {
            return 1 / Float(headDim).squareRoot()
        }
        guard scale.isFinite else {
            throw FlashAttentionError.invalidArgument("scale must be finite")
        }
        return scale
    }

    /// The int32 `dims` buffer in the shader's `FaHd32Dim` layout. `biasStrides`
    /// is `nil` when there is no bias.
    public static func dims(geometry: Geometry, biasStrides: [Int]?) -> [Int32] {
        let strides = biasStrides ?? [0, 0, 0, 0]
        precondition(strides.count == 4, "bias strides must have four entries")
        return [
            geometry.rows,
            geometry.heads,
            geometry.queryLength,
            geometry.keyLength,
            biasStrides == nil ? 0 : 1,
        ].map { Int32($0) } + strides.map { Int32($0) }
    }

    /// FLOPs per call: two GEMMs of `2 * Lq * Lk * D` per batch and head,
    /// the count `bench/flash_attention_hd32.py` uses.
    public static func flopsPerCall(_ geometry: Geometry) -> Double {
        4 * Double(geometry.batch) * Double(geometry.heads) * Double(geometry.queryLength)
            * Double(geometry.keyLength) * Double(headDim)
    }

    /// Median, min, p10 and p90 of per-call times in ms, as the Python bench
    /// reports them.
    public struct TimingSummary: Equatable, Sendable {
        public let medianMs: Double
        public let minMs: Double
        public let p10Ms: Double
        public let p90Ms: Double

        /// `samples` must be non-empty. Percentiles interpolate linearly
        /// between order statistics, like `bench.record.percentile`.
        public init(samples: [Double]) {
            precondition(!samples.isEmpty, "no timing samples")
            let sorted = samples.sorted()
            func percentile(_ fraction: Double) -> Double {
                let position = Double(sorted.count - 1) * fraction
                let low = Int(position.rounded(.down))
                let high = Int(position.rounded(.up))
                return sorted[low] + (sorted[high] - sorted[low]) * (position - Double(low))
            }
            medianMs = percentile(0.5)
            minMs = sorted[0]
            p10Ms = percentile(0.1)
            p90Ms = percentile(0.9)
        }

        /// TFLOP/s at the median.
        public func tflops(flops: Double) -> Double {
            flops / (medianMs * 1e9)
        }
    }

    /// Product of `values`, or `nil` on overflow.
    static func product(_ values: [Int]) -> Int? {
        var result = 1
        for value in values {
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            if overflow {
                return nil
            }
            result = next
        }
        return result
    }
}

#if canImport(Metal)
/// Metal host for `flash_attention_hd32`.
///
/// Compiles `kernels/flash_attention_hd32/flash_attention_hd32.metal`, the
/// same source the Python hosts load, with `MTLDevice.makeLibrary(source:)`
/// and launches its `flash_attention_hd32_bfloat` or
/// `flash_attention_hd32_float` entry point. There is no second copy of the
/// shader. Compile once and reuse the instance; it is safe to share across
/// threads because Metal pipeline states and command queues are.
public final class FlashAttentionHD32Kernel: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let pipelines: [FlashAttentionDType: MTLComputePipelineState]

    /// Compile the shared source for `device`, or the system default device.
    public init(device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw FlashAttentionError.metal("No Metal device is available")
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw FlashAttentionError.metal("Could not create a Metal command queue")
        }
        let source = try MetalKernels.metalSource(named: FlashAttentionHD32.kernelName)
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: MTLCompileOptions())
        } catch {
            throw FlashAttentionError.metal(
                "Could not compile \(FlashAttentionHD32.kernelName).metal: \(error)")
        }
        var pipelines: [FlashAttentionDType: MTLComputePipelineState] = [:]
        for dtype in FlashAttentionDType.allCases {
            guard let function = library.makeFunction(name: dtype.functionName) else {
                throw FlashAttentionError.metal("Missing Metal function \(dtype.functionName)")
            }
            let pipeline: MTLComputePipelineState
            do {
                pipeline = try device.makeComputePipelineState(function: function)
            } catch {
                throw FlashAttentionError.metal(
                    "Could not build the \(dtype.functionName) pipeline: \(error)")
            }
            guard pipeline.maxTotalThreadsPerThreadgroup >= FlashAttentionHD32.threadgroupSize
            else {
                throw FlashAttentionError.metal(
                    "\(dtype.functionName) allows \(pipeline.maxTotalThreadsPerThreadgroup) threads per threadgroup; it needs \(FlashAttentionHD32.threadgroupSize)"
                )
            }
            pipelines[dtype] = pipeline
        }
        self.device = device
        self.commandQueue = commandQueue
        self.pipelines = pipelines
    }

    /// Encode one launch of `softmax(q kᵀ * scale + bias) v` into `encoder`.
    ///
    /// `q` and `out` hold a contiguous `qShape` tensor, `(batch, heads,
    /// query_length, 32)`; `k` and `v` hold contiguous `kShape` tensors,
    /// `(batch, heads, key_length, 32)`. All buffers hold `dtype` elements
    /// (bf16 as `UInt16` bits) starting at offset 0. `bias`, when given, is a
    /// contiguous tensor of `biasShape`, which must broadcast to `(batch,
    /// heads, query_length, key_length)`; pass it unbroadcast. A `-inf` bias
    /// masks a key and a row with every key masked returns zeros. `scale`
    /// defaults to `1 / sqrt(32)`. `out` must not alias an input.
    ///
    /// The caller commits the command buffer and waits for it.
    public func encode(
        into encoder: MTLComputeCommandEncoder,
        q: MTLBuffer,
        k: MTLBuffer,
        v: MTLBuffer,
        out: MTLBuffer,
        qShape: [Int],
        kShape: [Int],
        vShape: [Int]? = nil,
        bias: MTLBuffer? = nil,
        biasShape: [Int]? = nil,
        scale: Float? = nil,
        dtype: FlashAttentionDType
    ) throws {
        let geometry = try FlashAttentionHD32.launchGeometry(
            q: qShape, k: kShape, v: vShape ?? kShape)
        let scale = try FlashAttentionHD32.resolveScale(scale)
        let strides: [Int]?
        switch (bias, biasShape) {
        case (nil, nil):
            strides = nil
        case (let bias?, let biasShape?):
            let resolved = try FlashAttentionHD32.biasStrides(
                biasShape: biasShape, target: geometry.scoreShape)
            try Self.check(bias, "bias", elements: resolved.shape.reduce(1, *), dtype: dtype)
            strides = resolved.strides
        default:
            throw FlashAttentionError.invalidArgument("bias and biasShape must be given together")
        }
        try Self.check(q, "q", elements: geometry.queryElements, dtype: dtype)
        try Self.check(k, "k", elements: geometry.keyElements, dtype: dtype)
        try Self.check(v, "v", elements: geometry.keyElements, dtype: dtype)
        try Self.check(out, "out", elements: geometry.queryElements, dtype: dtype)
        guard let pipeline = pipelines[dtype] else {
            throw FlashAttentionError.metal("No pipeline for \(dtype.rawValue)")
        }

        var dims = FlashAttentionHD32.dims(geometry: geometry, biasStrides: strides)
        var scaleValue = scale
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        // The shader needs a valid bias buffer even when has_bias is 0, and
        // never reads it then.
        encoder.setBuffer(bias ?? q, offset: 0, index: 3)
        encoder.setBuffer(out, offset: 0, index: 4)
        encoder.setBytes(&dims, length: MemoryLayout<Int32>.stride * dims.count, index: 5)
        encoder.setBytes(&scaleValue, length: MemoryLayout<Float>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: geometry.threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: FlashAttentionHD32.threadgroupSize, height: 1, depth: 1)
        )
    }

    /// Run one launch on `commandQueue` and wait for it.
    ///
    /// Inputs are contiguous fp32 arrays. With `dtype: .bfloat16` they are
    /// rounded to bf16 before the launch and the bf16 output is widened back
    /// to fp32. This copies every tensor; use `encode` on your own buffers
    /// to avoid that.
    public func run(
        q: [Float],
        k: [Float],
        v: [Float],
        qShape: [Int],
        kShape: [Int],
        bias: [Float]? = nil,
        biasShape: [Int]? = nil,
        scale: Float? = nil,
        dtype: FlashAttentionDType = .float32
    ) throws -> [Float] {
        let geometry = try FlashAttentionHD32.launchGeometry(q: qShape, k: kShape, v: kShape)
        try Self.checkCount(q, "q", geometry.queryElements)
        try Self.checkCount(k, "k", geometry.keyElements)
        try Self.checkCount(v, "v", geometry.keyElements)
        var biasBuffer: MTLBuffer?
        if let bias, let biasShape {
            let resolved = try FlashAttentionHD32.biasStrides(
                biasShape: biasShape, target: geometry.scoreShape)
            try Self.checkCount(bias, "bias", resolved.shape.reduce(1, *))
            biasBuffer = try makeBuffer(bias, dtype: dtype, name: "bias")
        } else if bias != nil || biasShape != nil {
            throw FlashAttentionError.invalidArgument("bias and biasShape must be given together")
        }
        let qBuffer = try makeBuffer(q, dtype: dtype, name: "q")
        let kBuffer = try makeBuffer(k, dtype: dtype, name: "k")
        let vBuffer = try makeBuffer(v, dtype: dtype, name: "v")
        guard
            let outBuffer = device.makeBuffer(
                length: geometry.queryElements * dtype.elementSize, options: .storageModeShared)
        else {
            throw FlashAttentionError.metal("Could not allocate the out buffer")
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
            let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw FlashAttentionError.metal("Could not create a Metal command encoder")
        }
        do {
            try encode(
                into: encoder, q: qBuffer, k: kBuffer, v: vBuffer, out: outBuffer,
                qShape: qShape, kShape: kShape, bias: biasBuffer, biasShape: biasShape,
                scale: scale, dtype: dtype)
        } catch {
            encoder.endEncoding()
            throw error
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status == .error {
            let reason = commandBuffer.error.map { "\($0)" } ?? "unknown error"
            throw FlashAttentionError.metal("flash_attention_hd32 failed: \(reason)")
        }

        let count = geometry.queryElements
        switch dtype {
        case .float32:
            let pointer = outBuffer.contents().bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        case .bfloat16:
            let pointer = outBuffer.contents().bindMemory(to: UInt16.self, capacity: count)
            return UnsafeBufferPointer(start: pointer, count: count).map(BFloat16Bits.toFloat)
        }
    }

    private func makeBuffer(_ values: [Float], dtype: FlashAttentionDType, name: String) throws
        -> MTLBuffer
    {
        let buffer: MTLBuffer?
        switch dtype {
        case .float32:
            buffer = values.withUnsafeBytes {
                device.makeBuffer(
                    bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
        case .bfloat16:
            buffer = values.map(BFloat16Bits.from).withUnsafeBytes {
                device.makeBuffer(
                    bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
            }
        }
        guard let buffer else {
            throw FlashAttentionError.metal("Could not allocate the \(name) buffer")
        }
        return buffer
    }

    private static func check(
        _ buffer: MTLBuffer, _ name: String, elements: Int, dtype: FlashAttentionDType
    ) throws {
        let needed = elements * dtype.elementSize
        guard buffer.length >= needed else {
            throw FlashAttentionError.invalidArgument(
                "\(name) buffer has \(buffer.length) bytes; the shape needs \(needed)")
        }
    }

    private static func checkCount(_ values: [Float], _ name: String, _ elements: Int) throws {
        guard values.count == elements else {
            throw FlashAttentionError.invalidArgument(
                "\(name) has \(values.count) elements; its shape needs \(elements)")
        }
    }
}
#endif
