// SPDX-License-Identifier: Apache-2.0
import XCTest

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

@testable import MetalKernels

final class MetalKernelsTests: XCTestCase {
    func testCheckoutRootContainsTheTemplate() throws {
        unsetenv("METAL_KERNELS_ROOT")
        let template = try MetalKernels.kernelsRoot()
            .appendingPathComponent("_template")
            .appendingPathComponent("kernel.metal")
        let text = try String(contentsOf: template, encoding: .utf8)
        XCTAssertTrue(text.contains("metal-kernels-template: not a kernel"))
    }

    func testRegistryIsEmpty() throws {
        unsetenv("METAL_KERNELS_ROOT")
        XCTAssertEqual(try MetalKernels.kernelNames(), [])
    }

    func testTemplateIsNotAKernel() {
        XCTAssertThrowsError(try MetalKernels.kernelDirectory(named: "_template")) { error in
            guard case KernelSourceError.unknownKernel("_template") = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testRejectsPathSeparators() {
        XCTAssertThrowsError(try MetalKernels.kernelDirectory(named: "../secrets")) { error in
            guard case KernelSourceError.unknownKernel = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testBundledTemplateMatchesCheckout() throws {
        unsetenv("METAL_KERNELS_ROOT")
        let live = try MetalKernels.kernelsRoot()
            .appendingPathComponent("_template")
            .appendingPathComponent("kernel.metal")
        guard let bundledRoot = MetalKernels.bundledKernelsRoot() else {
            XCTFail("kernels directory was not bundled into MetalKernels")
            return
        }
        let bundled = bundledRoot
            .appendingPathComponent("_template")
            .appendingPathComponent("kernel.metal")
        let liveText = try String(contentsOf: live, encoding: .utf8)
        let bundledText = try String(contentsOf: bundled, encoding: .utf8)
        XCTAssertEqual(bundledText, liveText)
    }

    func testEnvironmentOverride() throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("metal-kernels-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        setenv("METAL_KERNELS_ROOT", temporary.path, 1)
        defer { unsetenv("METAL_KERNELS_ROOT") }

        XCTAssertEqual(
            try MetalKernels.kernelsRoot().resolvingSymlinksInPath().path,
            temporary.resolvingSymlinksInPath().path
        )
        XCTAssertEqual(try MetalKernels.kernelNames(), [])
    }

    func testMissingEnvironmentRootThrows() {
        setenv("METAL_KERNELS_ROOT", "/no/such/metal-kernels-root", 1)
        defer { unsetenv("METAL_KERNELS_ROOT") }
        XCTAssertThrowsError(try MetalKernels.kernelsRoot()) { error in
            guard case KernelSourceError.environmentRootMissing = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }
}
