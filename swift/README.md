# Swift sources

The `MetalKernels` manifest is `Package.swift` at the repository root, not in this directory. SwiftPM only bundles files inside the package, so the manifest has to see `kernels/`. Sources and tests live here.

From the repository root:

```bash
swift test
```

SwiftPM prints `Found unhandled resource at <repo>/kernels`. That line means the resource is outside the target directory. The package still bundles it, and the tests compare the bundled `.metal` file with the checkout.
