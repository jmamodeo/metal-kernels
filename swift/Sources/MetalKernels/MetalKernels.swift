// SPDX-License-Identifier: Apache-2.0
import Foundation

/// Failure while resolving a kernel's Metal source.
public enum KernelSourceError: Error, Equatable, CustomStringConvertible, Sendable {
    /// `METAL_KERNELS_ROOT` is set, and that path is not a directory.
    case environmentRootMissing(String)
    /// No checkout and no bundled `kernels/` directory was found.
    case rootNotFound
    /// `name` is not a registered kernel directory.
    case unknownKernel(String)
    /// A kernel directory must contain exactly one `.metal` file.
    case metalSourceCount(kernel: String, count: Int)

    public var description: String {
        switch self {
        case .environmentRootMissing(let path):
            return "METAL_KERNELS_ROOT=\(path) is not a directory"
        case .rootNotFound:
            return "Could not find the kernels directory. Set METAL_KERNELS_ROOT or build from a checkout of the repository."
        case .unknownKernel(let name):
            return "Unknown kernel \(name)"
        case .metalSourceCount(let kernel, let count):
            return "Kernel \(kernel) must contain exactly one .metal file; found \(count)"
        }
    }
}

/// Locates Metal kernel sources shared with the Python `metal_kernels` package.
///
/// The `.metal` files under `kernels/` are the only copy of each kernel. This
/// type reads that text from a repository checkout or from the copy bundled
/// into the module. It does not compile shaders and it does not import Metal.
public enum MetalKernels {
    /// Directory of kernel sources.
    ///
    /// Search order:
    /// 1. `METAL_KERNELS_ROOT`, when it names a directory.
    /// 2. A `kernels/` directory found by walking up from this source file.
    ///    This is the live tree during `swift test`.
    /// 3. The `kernels/` directory copied into the module bundle.
    public static func kernelsRoot() throws -> URL {
        if let override = ProcessInfo.processInfo.environment["METAL_KERNELS_ROOT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !override.isEmpty
        {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: override, isDirectory: &isDirectory),
                isDirectory.boolValue
            else {
                throw KernelSourceError.environmentRootMissing(override)
            }
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        if let checkout = kernelsRootFromSourceLocation() {
            return checkout
        }
        if let bundled = bundledKernelsRoot() {
            return bundled
        }
        throw KernelSourceError.rootNotFound
    }

    /// Resource-bundle copy of `kernels/`, when the package bundled it.
    public static func bundledKernelsRoot() -> URL? {
        let roots = [Bundle.module.resourceURL, Bundle.module.bundleURL].compactMap { $0 }
        for root in roots {
            if let found = findKernelsRoot(under: root) {
                return found
            }
        }
        return nil
    }

    /// Registered kernel names. Directories whose names start with `_` or `.`
    /// are skipped, including `kernels/_template`.
    public static func kernelNames() throws -> [String] {
        let root = try kernelsRoot()
        let children = try FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        return try children.compactMap { url in
            guard try isKernelDirectory(url) else { return nil }
            return url.lastPathComponent
        }.sorted()
    }

    /// Directory for one kernel.
    public static func kernelDirectory(named name: String) throws -> URL {
        try validate(name: name)
        let directory = try kernelsRoot().appendingPathComponent(name, isDirectory: true)
        guard try isKernelDirectory(directory) else {
            throw KernelSourceError.unknownKernel(name)
        }
        return directory
    }

    /// UTF-8 text of the kernel's single `.metal` file.
    public static func metalSource(named name: String) throws -> String {
        let directory = try kernelDirectory(named: name)
        let sources = try metalFiles(in: directory)
        guard sources.count == 1, let source = sources.first else {
            throw KernelSourceError.metalSourceCount(kernel: name, count: sources.count)
        }
        return try String(contentsOf: source, encoding: .utf8)
    }

    private static func kernelsRootFromSourceLocation() -> URL? {
        var url = URL(fileURLWithPath: #filePath)
        // MetalKernels.swift lives at swift/Sources/MetalKernels/.
        for _ in 0..<8 {
            url.deleteLastPathComponent()
            let marker = url
                .appendingPathComponent("kernels", isDirectory: true)
                .appendingPathComponent("_template", isDirectory: true)
                .appendingPathComponent("kernel.metal")
            if FileManager.default.fileExists(atPath: marker.path) {
                return url.appendingPathComponent("kernels", isDirectory: true)
            }
        }
        return nil
    }

    private static func findKernelsRoot(under root: URL) -> URL? {
        let direct = root
            .appendingPathComponent("kernels", isDirectory: true)
            .appendingPathComponent("_template", isDirectory: true)
            .appendingPathComponent("kernel.metal")
        if FileManager.default.fileExists(atPath: direct.path) {
            return root.appendingPathComponent("kernels", isDirectory: true)
        }
        guard
            let enumerator = FileManager.default.enumerator(
                at: root,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )
        else {
            return nil
        }
        for case let fileURL as URL in enumerator {
            if fileURL.lastPathComponent == "kernel.metal",
                fileURL.deletingLastPathComponent().lastPathComponent == "_template"
            {
                return fileURL
                    .deletingLastPathComponent()
                    .deletingLastPathComponent()
            }
        }
        return nil
    }

    private static func validate(name: String) throws {
        if name.isEmpty
            || name.hasPrefix("_")
            || name.hasPrefix(".")
            || name == "."
            || name == ".."
            || name.contains("/")
            || name.contains("\\")
            || name.contains("\0")
        {
            throw KernelSourceError.unknownKernel(name)
        }
    }

    private static func isKernelDirectory(_ url: URL) throws -> Bool {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            return false
        }
        let name = url.lastPathComponent
        if name.hasPrefix("_") || name.hasPrefix(".") {
            return false
        }
        return try !metalFiles(in: url).isEmpty
    }

    private static func metalFiles(in directory: URL) throws -> [URL] {
        let children = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        return children.filter { $0.pathExtension == "metal" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
    }
}
