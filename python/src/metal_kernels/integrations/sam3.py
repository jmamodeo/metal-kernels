# SPDX-License-Identifier: Apache-2.0
"""Load-time compatibility patches for the pinned SAM 3 checkout.

SAM 3 at ``31cb7e624e4fe95301b1be1d4e809111593d2e45`` hard-codes CUDA in a
handful of places. This module rewrites those calls onto one device (CUDA if
it is present, otherwise MPS, otherwise CPU) so the image path and the
tracker can run with built-in PyTorch ops. It does not fork SAM 3. Call
:func:`install` before ``import sam3``. Uninstalling, or deleting this
module once upstream stops hard-coding CUDA, drops the patches.

Nothing here runs on Metal. MPS support is the intended target of the
patches, and it still has to be confirmed on a Mac with
``bench/sam3_shapes/profile.py``.

What gets patched:

- ``device="cuda"`` on tensor factories, so position-encoding and decoder
  coordinate caches allocate on the shim device.
- ``Tensor.cuda`` and ``Module.cuda``, so the tracker memory read and any
  ``model.cuda()`` move to the shim device.
- ``Tensor.pin_memory``, which raises "Cannot access accelerator device" when
  no CUDA device exists. On CUDA it is left alone.
- ``torch.autocast`` / ``torch.amp.autocast`` when the requested device is
  CUDA and CUDA is not the shim device. An enabled CUDA autocast becomes an
  enabled autocast on the shim device in the requested dtype (bfloat16 in
  SAM 3). An explicitly disabled autocast, such as the decoder FFN, stays
  disabled. PyTorch itself disables a CUDA autocast when CUDA is missing,
  and that leaves the ViT MLP's bfloat16 activation feeding a float32
  linear, which does not finish a forward. Enabling autocast on the shim
  device is what makes the forward complete.
- The ViT MLP's ``aten._addmm_activation``. On CUDA the original fused op is
  kept. On CPU and MPS it is replaced with ``F.linear`` plus ReLU or GELU in
  bfloat16, which is the same cast the fused helper does. ``set_mlp`` can
  force either implementation.
- ``model_builder._setup_device_and_mode``, which only calls ``.cuda()`` and
  would otherwise ignore ``device="mps"``.
- ``Sam3Processor``'s default ``device="cuda"`` when CUDA is not the shim
  device.
- The image entry points (``forward_image``, ``forward_text``,
  ``forward_grounding``, and the processor methods that call them) run under
  shim-device bfloat16 autocast. On CUDA they are not wrapped.
- A Triton stub, installed only when ``import triton`` fails, so
  ``sam3.model.edt`` can be imported. Calling a stubbed kernel raises.
  ``fill_hole_area=0`` does not call it. FlashAttention 3 and
  ``torch.compile`` are unchanged: leave ``use_fa3=False`` and
  ``compile=False``.

Position-encoding caches live in a plain dict, not as buffers, so they are
allocated on whatever device is installed at construction time. Call
:func:`install` before building the model, and build on that same device.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

SAM3_COMMIT = "31cb7e624e4fe95301b1be1d4e809111593d2e45"

_FACTORY_NAMES = (
    "zeros",
    "ones",
    "empty",
    "full",
    "arange",
    "tensor",
    "eye",
    "rand",
    "randn",
    "randint",
)

_IMAGE_AUTOCAST_METHODS = (
    ("sam3.model.vl_combiner", "SAM3VLBackbone", ("forward_image", "forward_text")),
    ("sam3.model.sam3_image", "Sam3Image", ("forward_grounding",)),
    (
        "sam3.model.sam3_image_processor",
        "Sam3Processor",
        ("set_image", "set_image_batch", "set_text_prompt", "add_geometric_prompt"),
    ),
)

_TRITON_STUB_ERROR = (
    "This Triton kernel is not available. metal_kernels.integrations.sam3 "
    "installed a stub so SAM 3 can import without Triton. The image path and "
    "a tracker built with fill_hole_area=0 do not call it."
)


@dataclass
class InstallReport:
    """What :func:`install` did. MPS rows are not measurements."""

    device: str
    mlp: str
    triton_stub: bool
    patches: tuple[str, ...]
    sam3_commit: str = SAM3_COMMIT
    mps_confirmed_on_this_machine: bool = False


@dataclass
class _State:
    installed: bool = False
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    mlp_mode: str = "auto"
    triton_stub: bool = False
    factory_redirects: int = 0
    cuda_calls: int = 0
    pin_memory_calls: int = 0
    autocast_rewrites: int = 0
    originals: dict = field(default_factory=dict)
    hook: object | None = None
    autocast_impl: object | None = None
    patched_modules: list[str] = field(default_factory=list)


_state = _State()


def default_device() -> torch.device:
    """CUDA if it exists, otherwise MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    if callable(is_available) and is_available():
        return torch.device("mps")
    return torch.device("cpu")


def current_device() -> torch.device:
    """Return the device the patches currently target."""
    if not _state.installed:
        raise RuntimeError("Call metal_kernels.integrations.sam3.install() first.")
    return _state.device


def stats() -> dict[str, int]:
    """Counts of calls the patches rewrote. Useful in tests."""
    return {
        "factory_redirects": _state.factory_redirects,
        "cuda_calls": _state.cuda_calls,
        "pin_memory_calls": _state.pin_memory_calls,
        "autocast_rewrites": _state.autocast_rewrites,
    }


def install(
    device: str | torch.device | None = None,
    mlp: str = "auto",
) -> InstallReport:
    """Patch PyTorch and, when SAM 3 is imported, its CUDA-only call sites.

    ``mlp`` is ``"auto"`` (fused ``aten._addmm_activation`` on CUDA, linear
    plus GELU or ReLU otherwise), ``"fused"``, or ``"linear_gelu"``.
    """
    if mlp not in ("auto", "fused", "linear_gelu"):
        raise ValueError(f"mlp must be auto, fused, or linear_gelu, not {mlp!r}")
    target = torch.device(device) if device is not None else default_device()
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("install(device='cuda') requires torch.cuda.is_available().")
    if target.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        is_available = getattr(mps, "is_available", None)
        if not (callable(is_available) and is_available()):
            raise RuntimeError("install(device='mps') requires a Mac with MPS available.")

    if "sam3" in sys.modules and not _state.installed:
        warnings.warn(
            "install() ran after SAM 3 was imported. Autocast decorators already "
            "captured torch.autocast. Import metal_kernels.integrations.sam3 and "
            "call install() before import sam3.",
            RuntimeWarning,
            stacklevel=2,
        )
    _state.device = target
    _state.mlp_mode = mlp
    if not _state.installed:
        _ensure_triton()
        _patch_torch()
        _state.hook = _ImportHook()
        sys.meta_path.insert(0, _state.hook)
        _state.installed = True
    _patch_already_imported()
    _bind_mlp()
    return _report()


def set_mlp(mlp: str) -> str:
    """Switch the ViT MLP implementation and return the one now in use."""
    if not _state.installed:
        raise RuntimeError("Call install() before set_mlp().")
    if mlp not in ("auto", "fused", "linear_gelu"):
        raise ValueError(f"mlp must be auto, fused, or linear_gelu, not {mlp!r}")
    _state.mlp_mode = mlp
    _bind_mlp()
    return active_mlp()


def active_mlp() -> str:
    """Return ``"fused"`` or ``"linear_gelu"`` for the current mode."""
    if _state.mlp_mode == "fused":
        return "fused"
    if _state.mlp_mode == "linear_gelu":
        return "linear_gelu"
    if _state.device.type == "cuda":
        return "fused"
    return "linear_gelu"


def uninstall() -> None:
    """Restore the PyTorch entry points this module replaced.

    SAM 3 modules that already imported a patched function keep that
    function. Call this only from tests that have not imported SAM 3, or
    before a fresh interpreter imports it.
    """
    if _state.hook is not None:
        try:
            sys.meta_path.remove(_state.hook)
        except ValueError:
            pass
        _state.hook = None
    for key, original in list(_state.originals.items()):
        owner, name = key
        setattr(owner, name, original)
    _state.originals.clear()
    _state.autocast_impl = None
    _state.installed = False
    _state.patched_modules.clear()


def build_image_model(**kwargs):
    """Build ``build_sam3_image_model`` on the installed device.

    ``compile`` defaults to False. ``device`` defaults to the shim device.
    Weight loading stays at SAM 3's own default.
    """
    if not _state.installed:
        raise RuntimeError("Call install() before build_image_model().")
    if "sam3" in sys.modules and "sam3.model_builder" not in _state.patched_modules:
        _patch_already_imported()
    import sam3

    kwargs.setdefault("device", _state.device.type)
    kwargs.setdefault("compile", False)
    return sam3.build_sam3_image_model(**kwargs)


def build_tracker(**kwargs):
    """Build ``build_tracker`` and move it onto the installed device.

    The tracker's constructor enters bfloat16 autocast for the rest of the
    process. That is upstream behavior; the patch only retargets the device.
    """
    if not _state.installed:
        raise RuntimeError("Call install() before build_tracker().")
    from sam3.model_builder import build_tracker as build_tracker_upstream

    model = build_tracker_upstream(**kwargs)
    if _state.device.type != "cpu":
        model.to(_state.device)
    # Inference. Training mode reads an attribute this constructor does not set.
    model.eval()
    return model


def _report() -> InstallReport:
    return InstallReport(
        device=_state.device.type,
        mlp=active_mlp(),
        triton_stub=_state.triton_stub,
        patches=tuple(_state.patched_modules),
    )


def _cuda_device(device) -> bool:
    if device is None:
        return False
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return device.split(":", 1)[0] == "cuda"
    return False


def _remember(owner, name: str) -> None:
    key = (owner, name)
    if key not in _state.originals:
        _state.originals[key] = getattr(owner, name)


def _patch_torch() -> None:
    for name in _FACTORY_NAMES:
        _remember(torch, name)
        original = getattr(torch, name)

        def _factory(*args, _original=original, **kwargs):
            device = kwargs.get("device")
            if _cuda_device(device) and _state.device.type != "cuda":
                kwargs["device"] = _state.device
                _state.factory_redirects += 1
            return _original(*args, **kwargs)

        setattr(torch, name, _factory)

    _remember(torch.Tensor, "cuda")
    original_cuda = torch.Tensor.cuda

    def _tensor_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        if _state.device.type == "cuda":
            return original_cuda(self, device=device, non_blocking=non_blocking, memory_format=memory_format)
        _state.cuda_calls += 1
        return self.to(device=_state.device, memory_format=memory_format)

    torch.Tensor.cuda = _tensor_cuda

    _remember(nn.Module, "cuda")
    original_module_cuda = nn.Module.cuda

    def _module_cuda(self, device=None):
        if _state.device.type == "cuda":
            return original_module_cuda(self, device=device)
        _state.cuda_calls += 1
        return self.to(_state.device)

    nn.Module.cuda = _module_cuda

    _remember(torch.Tensor, "pin_memory")
    original_pin = torch.Tensor.pin_memory

    def _pin_memory(self, *args, **kwargs):
        if _state.device.type == "cuda" and torch.cuda.is_available():
            return original_pin(self, *args, **kwargs)
        _state.pin_memory_calls += 1
        return self

    torch.Tensor.pin_memory = _pin_memory

    _remember(torch, "autocast")
    _remember(torch.amp, "autocast")
    original_autocast = torch.autocast
    _state.autocast_impl = original_autocast

    def _autocast(*args, **kwargs):
        args = list(args)
        if args:
            device_type = args[0]
        else:
            device_type = kwargs.get("device_type", "cuda")
        enabled = kwargs.get("enabled", True)
        if _cuda_device(device_type) and _state.device.type != "cuda":
            _state.autocast_rewrites += 1
            rewritten = _state.device.type
            if args:
                args[0] = rewritten
            else:
                kwargs["device_type"] = rewritten
            if not enabled:
                kwargs["enabled"] = False
        return original_autocast(*args, **kwargs)

    torch.autocast = _autocast
    torch.amp.autocast = _autocast


def _original_autocast():
    if _state.autocast_impl is None:
        raise RuntimeError("Call install() before using the SAM 3 autocast wrapper.")
    return _state.autocast_impl


def _under_autocast(fn):
    if getattr(fn, "_metal_kernels_autocast", False):
        return fn

    def wrapped(*args, **kwargs):
        if _state.device.type == "cuda":
            return fn(*args, **kwargs)
        autocast = _original_autocast()
        with autocast(_state.device.type, dtype=torch.bfloat16, enabled=True):
            return fn(*args, **kwargs)

    wrapped._metal_kernels_autocast = True
    return wrapped


def _linear_gelu(activation, linear, mat1):
    """bfloat16 linear plus ReLU or GELU, matching ``perflib.fused.addmm_act``."""
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")
    weight = linear.weight.detach().to(torch.bfloat16)
    bias = None if linear.bias is None else linear.bias.detach().to(torch.bfloat16)
    hidden = F.linear(mat1.to(torch.bfloat16), weight, bias)
    if activation in (F.relu, nn.ReLU) or (isinstance(activation, type) and issubclass(activation, nn.ReLU)):
        return F.relu(hidden)
    if activation in (F.gelu, nn.GELU) or (isinstance(activation, type) and issubclass(activation, nn.GELU)):
        return F.gelu(hidden)
    raise ValueError(f"Unexpected activation {activation}")


def _fused_module():
    return sys.modules.get("sam3.perflib.fused")


def _bind_mlp() -> None:
    fused = _fused_module()
    if fused is None:
        return
    choice = active_mlp()
    if choice == "fused":
        impl = fused._metal_kernels_original_addmm_act
    else:
        impl = _linear_gelu
    fused.addmm_act = impl
    vitdet = sys.modules.get("sam3.model.vitdet")
    if vitdet is not None:
        vitdet.addmm_act = impl


def _patch_fused(module) -> None:
    if not hasattr(module, "_metal_kernels_original_addmm_act"):
        module._metal_kernels_original_addmm_act = module.addmm_act
    _note("sam3.perflib.fused")
    _bind_mlp()


def _patch_vitdet(module) -> None:
    _note("sam3.model.vitdet")
    _bind_mlp()


def _patch_model_builder(module) -> None:
    if getattr(module._setup_device_and_mode, "_metal_kernels_device", False):
        return

    def _setup_device_and_mode(model, device, eval_mode):
        if _cuda_device(device):
            model = model.cuda()
        elif device is not None and not _cpu_device(device):
            model = model.to(device)
        if eval_mode:
            model.eval()
        return model

    _setup_device_and_mode._metal_kernels_device = True
    module._setup_device_and_mode = _setup_device_and_mode
    _note("sam3.model_builder")


def _cpu_device(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cpu"
    return str(device).split(":", 1)[0] == "cpu"


def _patch_processor(module) -> None:
    cls = module.Sam3Processor
    if not getattr(cls.__init__, "_metal_kernels_device", False):
        original_init = cls.__init__

        def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
            if _cuda_device(device) and _state.device.type != "cuda":
                device = _state.device.type
            original_init(
                self,
                model,
                resolution=resolution,
                device=device,
                confidence_threshold=confidence_threshold,
            )

        __init__._metal_kernels_device = True
        cls.__init__ = __init__
    for name in ("set_image", "set_image_batch", "set_text_prompt", "add_geometric_prompt"):
        current = getattr(cls, name)
        setattr(cls, name, _under_autocast(current))
    _note("sam3.model.sam3_image_processor")


def _patch_autocast_methods(module_name: str, class_name: str, method_names: tuple[str, ...]) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        return
    cls = getattr(module, class_name, None)
    if cls is None:
        return
    for name in method_names:
        current = getattr(cls, name, None)
        if current is None:
            continue
        setattr(cls, name, _under_autocast(current))
    _note(f"{module_name}.{class_name}")


def _note(name: str) -> None:
    if name not in _state.patched_modules:
        _state.patched_modules.append(name)


def _patch_already_imported() -> None:
    mapping = {
        "sam3.perflib.fused": _patch_fused,
        "sam3.model.vitdet": _patch_vitdet,
        "sam3.model_builder": _patch_model_builder,
        "sam3.model.sam3_image_processor": _patch_processor,
    }
    for name, patch in mapping.items():
        module = sys.modules.get(name)
        if module is not None:
            patch(module)
    for module_name, class_name, methods in _IMAGE_AUTOCAST_METHODS:
        if module_name == "sam3.model.sam3_image_processor":
            continue
        if module_name in sys.modules:
            _patch_autocast_methods(module_name, class_name, methods)


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, patch):
        self._inner = inner
        self._patch = patch

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        if create is None:
            return None
        return create(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        self._patch(module)


class _ImportHook(importlib.abc.MetaPathFinder):
    _TARGETS = {
        "sam3.perflib.fused": _patch_fused,
        "sam3.model.vitdet": _patch_vitdet,
        "sam3.model_builder": _patch_model_builder,
        "sam3.model.sam3_image_processor": _patch_processor,
        "sam3.model.vl_combiner": lambda module: _patch_autocast_methods(
            "sam3.model.vl_combiner", "SAM3VLBackbone", ("forward_image", "forward_text")
        ),
        "sam3.model.sam3_image": lambda module: _patch_autocast_methods(
            "sam3.model.sam3_image", "Sam3Image", ("forward_grounding",)
        ),
    }

    def find_spec(self, fullname, path, target=None):
        patch = self._TARGETS.get(fullname)
        if patch is None or _state.hook is None:
            return None
        try:
            sys.meta_path.remove(_state.hook)
        except ValueError:
            return None
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, _state.hook)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, patch)
        return spec


def _ensure_triton() -> None:
    try:
        module = importlib.import_module("triton")
    except Exception:
        module = None
    if module is None or sys.modules.get("triton") is None:
        for key in list(sys.modules):
            if key == "triton" or key.startswith("triton."):
                sys.modules.pop(key, None)
        _install_triton_stub()


def _install_triton_stub() -> None:
    language = types.ModuleType("triton.language")

    def _language_getattr(name):
        marker = type(name, (), {})
        setattr(language, name, marker)
        return marker

    language.__getattr__ = _language_getattr

    def _reject(*_args, **_kwargs):
        raise RuntimeError(_TRITON_STUB_ERROR)

    class _Kernel:
        def __getitem__(self, _grid):
            return _reject

        def __call__(self, *_args, **_kwargs):
            raise RuntimeError(_TRITON_STUB_ERROR)

    def jit(fn=None, **_kwargs):
        def decorate(_fn):
            return _Kernel()

        if fn is None:
            return decorate
        return decorate(fn)

    def autotune(*_args, **_kwargs):
        def decorate(fn):
            return fn

        return decorate

    class Config(dict):
        def __init__(self, values=None, **kwargs):
            super().__init__(values or {})
            self.update(kwargs)

    triton = types.ModuleType("triton")
    triton.jit = jit
    triton.autotune = autotune
    triton.Config = Config
    triton.language = language
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = language
    _state.triton_stub = True


__all__ = [
    "SAM3_COMMIT",
    "InstallReport",
    "active_mlp",
    "build_image_model",
    "build_tracker",
    "current_device",
    "default_device",
    "install",
    "set_mlp",
    "stats",
    "uninstall",
]
