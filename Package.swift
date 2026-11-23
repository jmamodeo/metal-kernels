// swift-tools-version: 6.0
import PackageDescription

// The package root is the repository root so `kernels/` can be bundled.
// SwiftPM only copies resources that live inside the package, and tools
// version 6.0 is what allows a resource path outside the target directory.
// A manifest under swift/ would leave kernels/ outside the package.
//
// `.copy` keeps the .metal files as source text. A process rule would let
// Xcode compile them into a default.metallib, which is not what the Python
// loaders consume. The path is relative to the target directory `swift/`.
//
// `swift test` prints "Found unhandled resource at <repo>/kernels". SwiftPM
// 6 says that when a declared resource sits outside the target directory,
// then bundles the directory anyway. The package tests check that the
// bundled bytes match the checkout.
let package = Package(
    name: "MetalKernels",
    products: [
        .library(name: "MetalKernels", targets: ["MetalKernels"]),
    ],
    targets: [
        .target(
            name: "MetalKernels",
            path: "swift",
            exclude: ["Tests", "README.md", "Sources/FlashAttentionHD32Bench"],
            sources: ["Sources/MetalKernels"],
            resources: [
                .copy("../kernels"),
            ]
        ),
        // Timing mode for the Swift host; see swift/README.md. An executable
        // target, not a second library product.
        .executableTarget(
            name: "FlashAttentionHD32Bench",
            dependencies: ["MetalKernels"],
            path: "swift/Sources/FlashAttentionHD32Bench"
        ),
        .testTarget(
            name: "MetalKernelsTests",
            dependencies: ["MetalKernels"],
            path: "swift/Tests/MetalKernelsTests"
        ),
    ]
)
