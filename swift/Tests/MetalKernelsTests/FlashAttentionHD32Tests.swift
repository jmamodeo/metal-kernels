// SPDX-License-Identifier: Apache-2.0
import XCTest

#if canImport(Metal)
import Metal
#endif

@testable import MetalKernels

final class FlashAttentionHD32Tests: XCTestCase {
    // Launch rules: mirror tests/test_flash_attention_hd32.py.

    func testLaunchConstantsMatchTheShader() throws {
        let source = try MetalKernels.metalSource(named: FlashAttentionHD32.kernelName)
        XCTAssertTrue(source.contains("FA_HD32_BLOCK_Q = \(FlashAttentionHD32.queryBlock);"))
        XCTAssertTrue(source.contains("FA_HD32_BLOCK_K = \(FlashAttentionHD32.keyBlock);"))
        XCTAssertTrue(source.contains("FA_HD32_HEAD_DIM = \(FlashAttentionHD32.headDim);"))
        XCTAssertTrue(source.contains("FA_HD32_DIM_COUNT = \(FlashAttentionHD32.dimCount),"))
        XCTAssertEqual(FlashAttentionHD32.threadgroupSize, 128)
        XCTAssertEqual(FlashAttentionHD32.threadgroupMemoryBytes, 8192)
        for dtype in FlashAttentionDType.allCases {
            let name = dtype.functionName.dropFirst("flash_attention_hd32_".count)
            XCTAssertTrue(source.contains("FA_HD32_INSTANTIATE(\(name), "), dtype.functionName)
        }
    }

    func testLaunchGeometry() throws {
        let cases: [([Int], [Int], Int)] = [
            ([1, 1, 1, 1], [1, 1, 1, 1], 128),
            ([1, 2, 5, 7], [10, 2, 5, 7], 256),
            ([2, 3, 17, 33], [102, 3, 17, 33], 6 * 128),
            ([1, 2, 50, 7], [100, 2, 50, 7], 2 * 2 * 128),
            ([1, 8, 5184, 5184], [41472, 8, 5184, 5184], 8 * 162 * 128),
            ([1, 8, 201, 5184], [1608, 8, 201, 5184], 8 * 7 * 128),
        ]
        for (shape, expected, threads) in cases {
            let geometry = try FlashAttentionHD32.launchGeometry(
                q: [shape[0], shape[1], shape[2], 32],
                k: [shape[0], shape[1], shape[3], 32],
                v: [shape[0], shape[1], shape[3], 32]
            )
            XCTAssertEqual(
                [geometry.rows, geometry.heads, geometry.queryLength, geometry.keyLength], expected)
            XCTAssertEqual(geometry.threads, threads)
            XCTAssertEqual(geometry.threadgroups * FlashAttentionHD32.threadgroupSize, threads)
        }
    }

    func testLaunchGeometryRejectsBadShapes() {
        let q = [1, 2, 4, 32]
        let kv = [1, 2, 6, 32]
        assertInvalid("shape") { try FlashAttentionHD32.launchGeometry(q: [1, 2, 4], k: kv, v: kv) }
        assertInvalid("head dimension 32") {
            try FlashAttentionHD32.launchGeometry(q: [1, 2, 4, 64], k: [1, 2, 6, 64], v: [1, 2, 6, 64])
        }
        assertInvalid("head dimension 32") {
            try FlashAttentionHD32.launchGeometry(q: q, k: kv, v: [1, 2, 6, 16])
        }
        assertInvalid("batch and head") {
            try FlashAttentionHD32.launchGeometry(q: q, k: [1, 3, 6, 32], v: [1, 3, 6, 32])
        }
        assertInvalid("k and v sequence lengths") {
            try FlashAttentionHD32.launchGeometry(q: q, k: kv, v: [1, 2, 5, 32])
        }
        assertInvalid("positive") {
            try FlashAttentionHD32.launchGeometry(q: [1, 2, 0, 32], k: kv, v: kv)
        }
        assertInvalid("positive") {
            try FlashAttentionHD32.launchGeometry(q: q, k: [1, 2, 0, 32], v: [1, 2, 0, 32])
        }
        assertInvalid("positive") {
            try FlashAttentionHD32.launchGeometry(q: [-1, 2, 4, 32], k: [-1, 2, 6, 32], v: [-1, 2, 6, 32])
        }
        assertInvalid("at most") {
            try FlashAttentionHD32.launchGeometry(q: [1, 1, 1 << 26, 32], k: [1, 1, 1, 32], v: [1, 1, 1, 32])
        }
        // One query per head still launches a full threadgroup.
        assertInvalid("at most") {
            let shape = [1 << 25, 1, 1, 32]
            return try FlashAttentionHD32.launchGeometry(q: shape, k: shape, v: shape)
        }
        assertInvalid("at most") {
            let shape = [Int.max, Int.max, 1, 32]
            return try FlashAttentionHD32.launchGeometry(q: shape, k: shape, v: shape)
        }
    }

    func testBiasStrides() throws {
        let target = [2, 3, 4, 5]
        let cases: [([Int], [Int], [Int])] = [
            ([2, 3, 4, 5], [2, 3, 4, 5], [60, 20, 5, 1]),
            ([4, 5], [1, 1, 4, 5], [0, 0, 5, 1]),
            ([5], [1, 1, 1, 5], [0, 0, 0, 1]),
            ([], [1, 1, 1, 1], [0, 0, 0, 0]),
            ([2, 1, 4, 1], [2, 1, 4, 1], [4, 0, 1, 0]),
            ([1, 3, 1, 5], [1, 3, 1, 5], [0, 5, 0, 1]),
        ]
        for (biasShape, shape, strides) in cases {
            let resolved = try FlashAttentionHD32.biasStrides(biasShape: biasShape, target: target)
            XCTAssertEqual(resolved.shape, shape, "\(biasShape)")
            XCTAssertEqual(resolved.strides, strides, "\(biasShape)")
        }
        let unit = try FlashAttentionHD32.biasStrides(biasShape: [1, 1, 1, 1], target: [1, 1, 1, 1])
        XCTAssertEqual(unit.shape, [1, 1, 1, 1])
        XCTAssertEqual(unit.strides, [0, 0, 0, 0])
    }

    func testBiasStridesRejectsNonBroadcastable() {
        for biasShape in [[2, 3, 4, 6], [3, 4, 4], [2, 3, 1, 5, 1], [1, 2, 3, 4, 5], [7], [2, 2, 4, 5]] {
            assertInvalid("broadcastable") {
                try FlashAttentionHD32.biasStrides(biasShape: biasShape, target: [2, 3, 4, 5])
            }
        }
    }

    func testResolveScale() throws {
        XCTAssertEqual(try FlashAttentionHD32.resolveScale(nil), 1 / Float(32).squareRoot(), accuracy: 1e-7)
        XCTAssertEqual(try FlashAttentionHD32.resolveScale(0.25), 0.25)
        assertInvalid("finite") { try FlashAttentionHD32.resolveScale(.nan) }
        assertInvalid("finite") { try FlashAttentionHD32.resolveScale(.infinity) }
    }

    func testDimsLayout() throws {
        let geometry = try FlashAttentionHD32.launchGeometry(
            q: [2, 3, 4, 32], k: [2, 3, 5, 32], v: [2, 3, 5, 32])
        XCTAssertEqual(
            FlashAttentionHD32.dims(geometry: geometry, biasStrides: nil),
            [24, 3, 4, 5, 0, 0, 0, 0, 0])
        XCTAssertEqual(
            FlashAttentionHD32.dims(geometry: geometry, biasStrides: [0, 0, 5, 1]),
            [24, 3, 4, 5, 1, 0, 0, 5, 1])
        XCTAssertEqual(
            FlashAttentionHD32.dims(geometry: geometry, biasStrides: nil).count,
            FlashAttentionHD32.dimCount)
    }

    func testBFloat16RoundsToNearestEven() {
        XCTAssertEqual(BFloat16Bits.from(1), 0x3F80)
        XCTAssertEqual(BFloat16Bits.from(-2), 0xC000)
        XCTAssertEqual(BFloat16Bits.from(-.infinity), 0xFF80)
        // Halfway cases round to the even mantissa.
        XCTAssertEqual(BFloat16Bits.from(Float(bitPattern: 0x3F80_8000)), 0x3F80)
        XCTAssertEqual(BFloat16Bits.from(Float(bitPattern: 0x3F81_8000)), 0x3F82)
        XCTAssertEqual(BFloat16Bits.from(Float(bitPattern: 0x3F80_8001)), 0x3F81)
        XCTAssertTrue(BFloat16Bits.toFloat(BFloat16Bits.from(.nan)).isNaN)
        for bits: UInt16 in [0x0000, 0x3F80, 0x4049, 0xC2F7, 0x7F7F, 0xFF80] {
            XCTAssertEqual(BFloat16Bits.from(BFloat16Bits.toFloat(bits)), bits)
        }
    }

    // GPU launches: macOS with a Metal device only.

    #if canImport(Metal)
    private static let sharedKernel = Result { try FlashAttentionHD32Kernel() }

    private func kernel() throws -> FlashAttentionHD32Kernel {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("no Metal device")
        }
        return try Self.sharedKernel.get()
    }

    func testMatchesReferenceSmall() throws {
        let kernel = try kernel()
        for dtype in FlashAttentionDType.allCases {
            for shape in [[1, 1, 1, 1], [1, 2, 5, 7], [2, 3, 17, 33], [1, 2, 9, 100], [1, 2, 9, 1000]] {
                try check(kernel, shape: shape, dtype: dtype)
            }
        }
    }

    func testMatchesReferenceTailBlocks() throws {
        let kernel = try kernel()
        for dtype in FlashAttentionDType.allCases {
            try check(kernel, shape: [1, 2, 50, 7], dtype: dtype)
            try check(kernel, shape: [1, 1, 50, 5184 + 17], dtype: dtype)
            try check(kernel, shape: [1, 1, 50, 5184 + 17], biasShape: [1, 1, 50, 5184 + 17], dtype: dtype)
        }
    }

    func testMatchesReferenceWithBias() throws {
        let kernel = try kernel()
        for dtype in FlashAttentionDType.allCases {
            for keyLength in [7, 100, 1000] {
                for biasShape in [[2, 3, 17, keyLength], [keyLength], [17, keyLength], [2, 1, 17, 1], []] {
                    try check(kernel, shape: [2, 3, 17, keyLength], biasShape: biasShape, dtype: dtype)
                }
            }
        }
    }

    func testFullyMaskedRowsAreZero() throws {
        let kernel = try kernel()
        let (batch, heads, queryLength, keyLength) = (1, 2, 40, 100)
        var random = SplitMix(seed: 3)
        let q = random.normals(batch * heads * queryLength * 32)
        let k = random.normals(batch * heads * keyLength * 32)
        let v = random.normals(batch * heads * keyLength * 32)
        // Rows 1 and 35 of every head are fully masked; 35 sits in the
        // partial last query block.
        var bias = [Float](repeating: 0, count: queryLength * keyLength)
        for row in [1, 35] {
            for key in 0..<keyLength {
                bias[row * keyLength + key] = -.infinity
            }
        }
        for dtype in FlashAttentionDType.allCases {
            let out = try kernel.run(
                q: q, k: k, v: v,
                qShape: [batch, heads, queryLength, 32], kShape: [batch, heads, keyLength, 32],
                bias: bias, biasShape: [queryLength, keyLength], dtype: dtype)
            for head in 0..<heads {
                for row in [1, 35] {
                    let start = (head * queryLength + row) * 32
                    XCTAssertEqual(Array(out[start..<start + 32]), [Float](repeating: 0, count: 32))
                }
            }
        }
    }

    func testMatchesReferenceWithExplicitScale() throws {
        let kernel = try kernel()
        for dtype in FlashAttentionDType.allCases {
            try check(kernel, shape: [1, 2, 5, 70], scale: 0.37, dtype: dtype)
        }
    }

    func testSingleKeyReturnsTheValue() throws {
        let kernel = try kernel()
        var random = SplitMix(seed: 5)
        let q = random.normals(2 * 9 * 32)
        let k = random.normals(2 * 32)
        let v = random.normals(2 * 32).map { BFloat16Bits.toFloat(BFloat16Bits.from($0)) }
        for dtype in FlashAttentionDType.allCases {
            let out = try kernel.run(
                q: q, k: k, v: v, qShape: [1, 2, 9, 32], kShape: [1, 2, 1, 32], dtype: dtype)
            for head in 0..<2 {
                for row in 0..<9 {
                    let start = (head * 9 + row) * 32
                    XCTAssertEqual(Array(out[start..<start + 32]), Array(v[head * 32..<head * 32 + 32]))
                }
            }
        }
    }

    func testRunRejectsMismatchedInputs() throws {
        let kernel = try kernel()
        let x = [Float](repeating: 0, count: 64)
        assertInvalid("elements") {
            try kernel.run(q: [0], k: x, v: x, qShape: [1, 1, 2, 32], kShape: [1, 1, 2, 32])
        }
        assertInvalid("together") {
            try kernel.run(q: x, k: x, v: x, qShape: [1, 1, 2, 32], kShape: [1, 1, 2, 32], bias: [0])
        }
        assertInvalid("broadcastable") {
            try kernel.run(
                q: x, k: x, v: x, qShape: [1, 1, 2, 32], kShape: [1, 1, 2, 32],
                bias: [0, 0, 0], biasShape: [3])
        }
    }

    /// Run one case on the GPU and compare it with `referenceAttention`.
    /// Inputs are drawn as bf16 values so both dtypes see the same numbers.
    /// A bias has every third element set to `-inf`.
    private func check(
        _ kernel: FlashAttentionHD32Kernel,
        shape: [Int],
        biasShape: [Int]? = nil,
        scale: Float? = nil,
        dtype: FlashAttentionDType,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let (batch, heads, queryLength, keyLength) = (shape[0], shape[1], shape[2], shape[3])
        var random = SplitMix(seed: UInt64(batch * 1_000_003 + heads * 10_007 + queryLength * 101 + keyLength))
        let bf16 = { (x: Float) in BFloat16Bits.toFloat(BFloat16Bits.from(x)) }
        let q = random.normals(batch * heads * queryLength * 32).map(bf16)
        let k = random.normals(batch * heads * keyLength * 32).map(bf16)
        let v = random.normals(batch * heads * keyLength * 32).map(bf16)
        var bias: [Float]?
        if let biasShape {
            bias = random.normals(biasShape.reduce(1, *)).enumerated().map { index, value in
                index % 3 == 2 ? -.infinity : bf16(value)
            }
        }
        let qShape = [batch, heads, queryLength, 32]
        let kShape = [batch, heads, keyLength, 32]
        let out = try kernel.run(
            q: q, k: k, v: v, qShape: qShape, kShape: kShape,
            bias: bias, biasShape: biasShape, scale: scale, dtype: dtype)
        let expected = try referenceAttention(
            q: q, k: k, v: v, qShape: qShape, kShape: kShape,
            bias: bias, biasShape: biasShape, scale: scale)
        let (rtol, atol): (Float, Float) = dtype == .bfloat16 ? (2e-2, 1e-2) : (1e-3, 1e-4)
        XCTAssertEqual(out.count, expected.count, file: file, line: line)
        var worst: (index: Int, error: Float) = (0, 0)
        for (index, (actual, wanted)) in zip(out, expected).enumerated() {
            let excess = abs(actual - wanted) - (atol + rtol * abs(wanted))
            if !(excess <= worst.error) {
                worst = (index, excess.isNaN ? .infinity : excess)
            }
        }
        XCTAssertEqual(
            worst.error, 0,
            "\(dtype.rawValue) \(shape) bias \(String(describing: biasShape)): element \(worst.index) is \(out[worst.index]), expected \(expected[worst.index])",
            file: file, line: line)
    }
    #endif

    private func assertInvalid<T>(
        _ fragment: String,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ body: () throws -> T
    ) {
        XCTAssertThrowsError(try body(), file: file, line: line) { error in
            guard case FlashAttentionError.invalidArgument(let message) = error else {
                return XCTFail("unexpected error: \(error)", file: file, line: line)
            }
            XCTAssertTrue(message.contains(fragment), "\(message)", file: file, line: line)
        }
    }
}

/// `softmax(q kᵀ * scale + bias) v` on the CPU with fp64 accumulation, like
/// `reference/sdpa.py`. A row whose keys are all `-inf` returns zeros.
func referenceAttention(
    q: [Float], k: [Float], v: [Float], qShape: [Int], kShape: [Int],
    bias: [Float]?, biasShape: [Int]?, scale: Float?
) throws -> [Float] {
    let geometry = try FlashAttentionHD32.launchGeometry(q: qShape, k: kShape, v: kShape)
    let scale = Double(try FlashAttentionHD32.resolveScale(scale))
    let strides = try biasShape.map {
        try FlashAttentionHD32.biasStrides(biasShape: $0, target: geometry.scoreShape).strides
    }
    let d = FlashAttentionHD32.headDim
    var out = [Float](repeating: 0, count: geometry.queryElements)
    var scores = [Double](repeating: 0, count: geometry.keyLength)
    for b in 0..<geometry.batch {
        for h in 0..<geometry.heads {
            let head = b * geometry.heads + h
            for i in 0..<geometry.queryLength {
                let qBase = (head * geometry.queryLength + i) * d
                var maximum = -Double.infinity
                for j in 0..<geometry.keyLength {
                    let kBase = (head * geometry.keyLength + j) * d
                    var dot = 0.0
                    for c in 0..<d {
                        dot += Double(q[qBase + c]) * Double(k[kBase + c])
                    }
                    var score = dot * scale
                    if let bias, let strides {
                        score += Double(bias[b * strides[0] + h * strides[1] + i * strides[2] + j * strides[3]])
                    }
                    scores[j] = score
                    maximum = max(maximum, score)
                }
                guard maximum.isFinite else { continue }
                var sum = 0.0
                var row = [Double](repeating: 0, count: d)
                for j in 0..<geometry.keyLength {
                    let p = exp(scores[j] - maximum)
                    sum += p
                    let vBase = (head * geometry.keyLength + j) * d
                    for c in 0..<d {
                        row[c] += p * Double(v[vBase + c])
                    }
                }
                for c in 0..<d {
                    out[qBase + c] = Float(row[c] / sum)
                }
            }
        }
    }
    return out
}

/// Deterministic normal samples (SplitMix64 + Box-Muller).
struct SplitMix {
    var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func uniform() -> Double {
        (Double(next() >> 11) + 0.5) / Double(1 << 53)
    }

    mutating func normals(_ count: Int) -> [Float] {
        (0..<count).map { _ in
            Float((-2 * log(uniform())).squareRoot() * cos(2 * Double.pi * uniform()))
        }
    }
}

final class FlashAttentionReferenceTests: XCTestCase {
    func testReferenceMatchesHandComputedCase() throws {
        // One head, one query, two keys. q·k = (1, 0); scale 1; bias masks
        // nothing, so p = softmax(1, 0).
        var q = [Float](repeating: 0, count: 32)
        var k = [Float](repeating: 0, count: 64)
        var v = [Float](repeating: 0, count: 64)
        q[0] = 1
        k[0] = 1
        v[0] = 2
        v[32] = 4
        let out = try referenceAttention(
            q: q, k: k, v: v, qShape: [1, 1, 1, 32], kShape: [1, 1, 2, 32],
            bias: nil, biasShape: nil, scale: 1)
        let p0 = exp(1.0) / (exp(1.0) + 1)
        XCTAssertEqual(Double(out[0]), 2 * p0 + 4 * (1 - p0), accuracy: 1e-6)

        let masked = try referenceAttention(
            q: q, k: k, v: v, qShape: [1, 1, 1, 32], kShape: [1, 1, 2, 32],
            bias: [-.infinity, -.infinity], biasShape: [2], scale: 1)
        XCTAssertEqual(masked, [Float](repeating: 0, count: 32))
    }
}
