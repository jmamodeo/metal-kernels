# Swift sources

The `MetalKernels` manifest is `Package.swift` at the repository root, not in this directory. SwiftPM only bundles files inside the package, so the manifest has to see `kernels/`. Sources and tests live here.

From the repository root:

```bash
swift test
```

SwiftPM prints `Found unhandled resource at <repo>/kernels`. That line means the resource is outside the target directory. The package still bundles it, and the tests compare the bundled `.metal` file with the checkout.

## flash_attention_hd32 host

`Sources/MetalKernels/FlashAttentionHD32.swift` launches `kernels/flash_attention_hd32/flash_attention_hd32.metal`. `FlashAttentionHD32Kernel` compiles that file at runtime with `MTLDevice.makeLibrary(source:options:)`, so there is no second copy of the shader. It dispatches `flash_attention_hd32_bfloat` or `flash_attention_hd32_float` with the same buffers, `dims` layout and 128-thread grid as the PyTorch MPS launcher.

```swift
import MetalKernels

let attention = try FlashAttentionHD32Kernel()
let out = try attention.run(
    q: q, k: k, v: v,
    qShape: [1, 8, 5184, 32], kShape: [1, 8, 5184, 32],
    bias: bias, biasShape: [5184, 5184],  // optional, passed unbroadcast
    dtype: .bfloat16)
```

- `run` takes contiguous `[Float]` arrays, rounds them to bf16 for `.bfloat16`, waits for the GPU and returns `[Float]`. `encode(into:q:k:v:out:...)` encodes one launch on caller-owned contiguous `MTLBuffer`s (bf16 as `UInt16` bits).
- `FlashAttentionHD32.launchGeometry`, `biasStrides`, `resolveScale` and `dims` mirror `metal_kernels.flash_attention` and build on every platform. `FlashAttentionHD32Kernel` is compiled only where `Metal` can be imported.
- `Tests/MetalKernelsTests/FlashAttentionHD32Tests.swift` checks the launch rules everywhere. On macOS it also compares GPU output with a CPU reference at the kernel README tolerances; those tests skip without a Metal device.
- Timing mode: `swift run -c release FlashAttentionHD32Bench` times `[1, 8, 5184, 32]` with no bias in bf16 and fp32. It runs 12 warmup and 102 timed launches, one command buffer each, and prints median, min, p10 and p90 ms per call and TFLOP/s. `--warmup N`, `--iterations N` and `--dtype bf16|fp32|all` override the defaults. It records nothing.
- On the Mac, `bench/run_mac.sh swift` runs `swift build`, `swift test` and the timing mode.
