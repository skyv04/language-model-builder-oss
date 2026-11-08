"""tinygrad WGSL execution through wgpu-native (Vulkan), including Linux/aarch64 wheels.

The stock tinygrad WebGPU runtime loads Dawn. This adapter instead drives the public
wgpu-py API, so ARM64 Linux does not need a local Chromium/Dawn build. Tensor
differentiation and WGSL generation remain tinygrad's implementation.

Speed comes from never paying WebGPU's fixed costs per kernel:
- dispatches are recorded into one open command encoder/compute pass and submitted
  together (``flush``), right before anything that must observe their results
  (buffer writes, reads, copies, ``synchronize``) or when a JIT graph batch ends;
- bind groups, bind-group layouts and integer uniforms are cached and reused;
- TinyJit replays go through ``WebGpuGraph``, which prepares every pipeline, bind
  group and launch size once and then records them with direct wgpu-native calls;
- freed buffers are recycled by tinygrad's LRU allocator, and real destruction is
  deferred until no unsubmitted command can still reference the buffer.
"""
from __future__ import annotations

import os
import re
import struct
import sys
import time
import types
from collections import OrderedDict, defaultdict
from pathlib import Path

import wgpu
from tinygrad.device import Compiled, Device, LRUAllocator, Program
from tinygrad.engine.jit import GraphRunner
from tinygrad.renderer.wgsl import WGSLRenderer

try:  # direct wgpu-native entry points for the hot recording loop
    from wgpu.backends.wgpu_native._ffi import ffi as _ffi, lib as _lib
except ImportError:  # pragma: no cover - other wgpu-py layouts fall back to the public API
    _ffi = _lib = None

DRIVER_MANIFEST = Path("/opt/mesa-adreno/share/vulkan/icd.d/freedreno_icd.aarch64.json")
PHYSICAL_ADAPTERS = ("integratedgpu", "discretegpu")
REQUESTED_LIMITS = (
    "max-storage-buffers-per-shader-stage",
    "max-storage-buffer-binding-size",
    "max-compute-workgroup-storage-size",
    "max-buffer-size",
)
BUFFER_USAGE = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
MAX_PENDING_COMMANDS = 4096  # submit even without a sync point so recordings stay small
MAX_BIND_GROUPS = 16384
# Transfers larger than SMALL_TRANSFER go through one persistent mappable buffer per direction, in
# STAGING_SIZE chunks.  wgpu's own write/read staging would otherwise stay mapped in this process
# afterwards (uploading 120 MB of weights left 117 MB resident; this way it is 4 MB).
SMALL_TRANSFER = 1 << 20
STAGING_SIZE = 4 << 20
_BINDING = re.compile(r"@binding\((\d+)\)\s*var<(storage|uniform)")


def configure_vulkan() -> None:
    if DRIVER_MANIFEST.is_file() and not (os.environ.get("VK_DRIVER_FILES") or os.environ.get("VK_ICD_FILENAMES")):
        os.environ["VK_DRIVER_FILES"] = str(DRIVER_MANIFEST)
    os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan")


def _round4(size: int) -> int:
    return (size + 3) & ~3


def _as_bytes(view: memoryview) -> memoryview:
    return view if view.format == "B" and view.ndim == 1 else view.cast("B")


class BufferView:
    """A byte range of a device buffer: tinygrad's memory planner packs intermediates into one arena."""

    __slots__ = ("buffer", "offset", "size")

    def __init__(self, buffer, offset: int, size: int):
        self.buffer, self.offset, self.size = buffer, offset, size


def _span(buffer) -> tuple[object, int]:
    """(underlying wgpu buffer, byte offset) of a buffer or view."""
    return (buffer.buffer, buffer.offset) if isinstance(buffer, BufferView) else (buffer, 0)


def _binding(buffer):
    if isinstance(buffer, BufferView):
        return {"buffer": buffer.buffer, "offset": buffer.offset, "size": _round4(buffer.size)}
    return buffer


def is_physical_adapter(info: dict) -> bool:
    return str(info.get("adapter_type", "")).lower() in PHYSICAL_ADAPTERS


def request_adapter():
    """The high-performance adapter and its info; raises for software (CPU) adapters."""
    configure_vulkan()
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = dict(adapter.info)
    if not is_physical_adapter(info):
        raise RuntimeError(f"Refusing a software WebGPU adapter: {info}")
    return adapter, info


def adapter_summary() -> dict:
    """Name/backend/type of the GPU adapter, without creating a device or loading tinygrad kernels."""
    _, info = request_adapter()
    return {
        "name": info.get("device") or info.get("description") or "GPU",
        "backend": info.get("backend_type", "Vulkan"),
        "adapter_type": info.get("adapter_type", ""),
    }


class NativeAllocator(LRUAllocator):
    def __init__(self, dev: "WebGpuDevice"):
        super().__init__(dev, supports_copy_from_disk=False)

    def _alloc(self, size, options):
        return self.dev.gpu.create_buffer(size=_round4(size), usage=BUFFER_USAGE)

    def _free(self, buffer, options):
        self.dev.release_buffer(buffer)

    def _copyin(self, dest, src: memoryview):
        self.dev.write(dest, src)

    def _copyout(self, dest: memoryview, src):
        self.dev.read(dest, src)

    def _transfer(self, dest, src, sz: int, src_dev, dest_dev):
        self.dev.copy(src, dest, sz)

    def _offset(self, buf, size: int, offset: int):
        base, start = _span(buf)
        if (start + offset) % self.dev.offset_alignment:
            raise RuntimeError(f"buffer view offset {start + offset} is not {self.dev.offset_alignment}-byte aligned")
        return BufferView(base, start + offset, size)


class NativeProgram(Program):
    def __init__(self, device: "WebGpuDevice", program):
        self.device = device
        self.name = program.name
        source = program.lib.decode("utf-8")
        kinds = dict((int(index), kind) for index, kind in _BINDING.findall(source))
        self.kinds = tuple(kinds.get(index, "uniform") for index in range(max(kinds, default=0) + 1))
        self.layout = device.layout(self.kinds)
        module = device.gpu.create_shader_module(code=source)
        self.pipeline = device.gpu.create_compute_pipeline(
            layout=device.gpu.create_pipeline_layout(bind_group_layouts=[self.layout]),
            compute={"module": module, "entry_point": self.name},
        )

    def resources(self, buffers, vals) -> list:
        """Buffers and integer uniforms in binding order (binding 0 is the INFINITY constant)."""
        buffers, vals = iter(buffers), iter(vals)
        resources = [self.device.infinity]
        for kind in self.kinds[1:]:
            resources.append(next(buffers) if kind == "storage" else self.device.int_uniform(next(vals)))
        return resources

    def __call__(self, *buffers, global_size=(1, 1, 1), local_size=None, vals=(), wait=False, **_kwargs):
        start = time.perf_counter()
        group = self.device.bind_group(self, buffers, vals)
        self.device.record([(self.pipeline, group, *(int(size) for size in global_size))])
        if wait:
            self.device.synchronize()
            return time.perf_counter() - start
        return None


class WebGpuGraph(GraphRunner):
    """Replays a captured TinyJit batch with pre-built pipelines, bind groups and launch sizes."""

    def __init__(self, linear, input_uops=()):
        super().__init__(linear, input_uops)
        self.dev: WebGpuDevice = Device[self.device]
        defaults = {name: 0 for name in self.vars}
        self.commands: list[tuple] = []
        self.arguments: list[list] = []
        self.replacements: list[list[tuple[int, int]]] = []
        for (_, ast, bufs, device_vars), program, replace in zip(self.calls, self.runtimes, self.uop_replace):
            order = list(ast.arg.globals)
            values = {**defaults, **device_vars}
            global_size, _ = ast.arg.launch_dims(values)
            arguments = [bufs[index]._buf for index in order]
            self.arguments.append(arguments)
            self.replacements.append([(order.index(position), slot) for position, slot in replace if position in order])
            group = self.dev.bind_group(program, arguments, ast.arg.vals(values))
            self.commands.append((program.pipeline, group, *(int(size) for size in global_size)))
        self.updatable = sorted(
            {j for j, replace in enumerate(self.replacements) if replace} | set(self.var_vals_replace) | set(self.launch_dims_replace)
        )

    def __call__(self, input_uops, var_vals, wait=False):
        start = time.perf_counter()
        if self.updatable:
            dims = {j: global_size for j, global_size, _ in self.updated_launch_dims(var_vals)}
            for j in self.updatable:
                _, ast, _, device_vars = self.calls[j]
                arguments = self.arguments[j]
                for position, slot in self.replacements[j]:
                    arguments[position] = input_uops[slot].buffer._buf
                values = {**var_vals, **device_vars}
                group = self.dev.bind_group(self.runtimes[j], arguments, ast.arg.vals(values))
                pipeline, _, *global_size = self.commands[j]
                global_size = dims.get(j, global_size)
                self.commands[j] = (pipeline, group, *(int(size) for size in global_size))
        self.dev.record(self.commands)
        # Submit now: the GPU runs this batch while Python records the next one.
        self.dev.flush()
        if wait:
            self.dev.synchronize()
            return time.perf_counter() - start
        return None


class WebGpuDevice(Compiled):
    def __init__(self, device: str):
        adapter, self.info = request_adapter()
        self.gpu = adapter.request_device_sync(
            required_limits={key: value for key, value in adapter.limits.items() if key.replace("_", "-") in REQUESTED_LIMITS}
        )
        self.queue = self.gpu.queue
        self.offset_alignment = int(adapter.limits.get("min-storage-buffer-offset-alignment", 256))
        self.infinity = self.gpu.create_buffer_with_data(data=struct.pack("<f", float("inf")), usage=wgpu.BufferUsage.UNIFORM)
        self.dispatches = 0
        self.submits = 0
        self._encoder = None
        self._pass = None
        self._pending = 0
        self._doomed: list = []
        self._layouts: dict[tuple, object] = {}
        self._uniforms: dict[int, object] = {}
        self._groups: OrderedDict[tuple, tuple] = OrderedDict()
        self._groups_by_buffer: defaultdict[int, set] = defaultdict(set)
        self._staging: dict[bool, object] = {}  # write? -> persistent mappable buffer
        super().__init__(device, NativeAllocator(self), [WGSLRenderer], NativeProgram, graph=WebGpuGraph, arch="float32")

    # ---- cached objects
    def layout(self, kinds: tuple[str, ...]):
        if kinds not in self._layouts:
            self._layouts[kinds] = self.gpu.create_bind_group_layout(entries=[
                {"binding": index, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": kind}}
                for index, kind in enumerate(kinds)
            ])
        return self._layouts[kinds]

    def int_uniform(self, value: int):
        if (buffer := self._uniforms.get(value)) is None:
            buffer = self._uniforms[value] = self.gpu.create_buffer_with_data(
                data=struct.pack("<i", value), usage=wgpu.BufferUsage.UNIFORM
            )
        return buffer

    def bind_group(self, program: NativeProgram, buffers, vals):
        key = (program.kinds, *map(id, buffers), *vals)
        entry = self._groups.get(key)
        if entry is None:
            resources = program.resources(buffers, vals)
            group = self.gpu.create_bind_group(
                layout=program.layout,
                entries=[{"binding": index, "resource": _binding(buffer)} for index, buffer in enumerate(resources)],
            )
            entry = self._groups[key] = (group, tuple(buffers))
            for buffer in buffers:
                self._groups_by_buffer[id(_span(buffer)[0])].add(key)
            if len(self._groups) > MAX_BIND_GROUPS:
                old_key, (_, old_buffers) = self._groups.popitem(last=False)
                for buffer in old_buffers:
                    self._groups_by_buffer[id(_span(buffer)[0])].discard(old_key)
        return entry[0]

    # ---- command recording
    def _compute_pass(self):
        if self._pass is None:
            if self._encoder is None:
                self._encoder = self.gpu.create_command_encoder()
            self._pass = self._encoder.begin_compute_pass()
        return self._pass

    def _command_encoder(self):
        if self._pass is not None:
            self._pass.end()
            self._pass = None
        if self._encoder is None:
            self._encoder = self.gpu.create_command_encoder()
        return self._encoder

    def record(self, commands) -> None:
        """Append (pipeline, bind group, x, y, z) dispatches to the open compute pass."""
        compute = self._compute_pass()
        if _lib is not None:
            handle, null = compute._internal, _ffi.NULL
            set_pipeline = _lib.wgpuComputePassEncoderSetPipeline
            set_group = _lib.wgpuComputePassEncoderSetBindGroup
            dispatch = _lib.wgpuComputePassEncoderDispatchWorkgroups
            for pipeline, group, x, y, z in commands:
                set_pipeline(handle, pipeline._internal)
                set_group(handle, 0, group._internal, 0, null)
                dispatch(handle, x, y, z)
        else:  # pragma: no cover
            for pipeline, group, x, y, z in commands:
                compute.set_pipeline(pipeline)
                compute.set_bind_group(0, group)
                compute.dispatch_workgroups(x, y, z)
        self.dispatches += len(commands)
        self._pending += len(commands)
        if self._pending >= MAX_PENDING_COMMANDS:
            self.flush()

    def flush(self) -> None:
        """Submit everything recorded so far, then destroy buffers that were waiting on it."""
        if self._encoder is not None:
            if self._pass is not None:
                self._pass.end()
                self._pass = None
            command_buffer = self._encoder.finish()
            self._encoder, self._pending = None, 0
            self.queue.submit([command_buffer])
            self.submits += 1
        if self._doomed:
            doomed, self._doomed = self._doomed, []
            for buffer in doomed:
                buffer.destroy()

    def synchronize(self) -> None:
        self.flush()
        if hasattr(self.gpu, "_poll_wait"):
            self.gpu._poll_wait()  # blocking wgpuDevicePoll: returns once all submitted work finished
        else:  # pragma: no cover
            self.queue.on_submitted_work_done_sync()

    # ---- memory
    def release_buffer(self, buffer) -> None:
        for key in self._groups_by_buffer.pop(id(buffer), ()):
            self._groups.pop(key, None)
        if self._encoder is not None:
            self._doomed.append(buffer)
        else:
            buffer.destroy()

    def _staging_buffer(self, write: bool):
        if write not in self._staging:
            usage = (wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC) if write else (wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST)
            self._staging[write] = self.gpu.create_buffer(size=STAGING_SIZE, usage=usage)
        return self._staging[write]

    def write(self, dest, src: memoryview) -> None:
        self.flush()  # queue writes would otherwise overtake recorded-but-unsubmitted kernels
        data = _as_bytes(src)
        base, offset = _span(dest)
        if data.nbytes <= SMALL_TRANSFER:
            self.queue.write_buffer(base, offset, data if data.nbytes % 4 == 0 else bytes(data) + bytes(4 - data.nbytes % 4))
            return
        staging = self._staging_buffer(write=True)
        for start in range(0, data.nbytes, STAGING_SIZE):
            chunk = data[start : start + STAGING_SIZE]
            size = _round4(chunk.nbytes)
            staging.map_async("WRITE", 0, size).sync_wait()  # waits until the previous chunk's copy is done
            try:
                staging.write_mapped(chunk if size == chunk.nbytes else bytes(chunk) + bytes(size - chunk.nbytes), 0)
            finally:
                staging.unmap()
            self._command_encoder().copy_buffer_to_buffer(staging, 0, base, offset + start, size)
            self.flush()

    def copy(self, src, dest, size: int) -> None:
        self._command_encoder().copy_buffer_to_buffer(*_span(src), *_span(dest), _round4(size))
        self._pending += 1

    def read(self, dest: memoryview, src) -> None:
        out = _as_bytes(dest)
        base, offset = _span(src)
        staging = self._staging_buffer(write=False)
        for start in range(0, out.nbytes, STAGING_SIZE):
            size = min(STAGING_SIZE, out.nbytes - start)
            self._command_encoder().copy_buffer_to_buffer(base, offset + start, staging, 0, _round4(size))
            self.flush()
            staging.map_async("READ_NOSYNC", 0, _round4(size)).sync_wait()
            try:
                out[start : start + size] = staging.read_mapped(0, _round4(size), copy=False)[:size]
            finally:
                staging.unmap()


VIEW_ALIGNMENT = 256  # WebGPU guarantees minStorageBufferOffsetAlignment <= 256; tinygrad's planner packs in 256-byte blocks


def _on_webgpu(device) -> bool:
    return device is not None and any(name.startswith("WEBGPU") for name in ((device,) if isinstance(device, str) else device))


def _enable_buffer_views() -> None:
    """Let tinygrad use 256-byte-aligned buffer views on WEBGPU, which enables its memory planner.

    tinygrad disables views (and so memory planning) for all of WEBGPU because arbitrary
    offsets violate WebGPU's binding-offset alignment.  This runtime binds aligned views as
    buffer ranges (``BufferView``), so only unaligned views are refused (tinygrad then copies,
    as before).  Without the planner every JIT intermediate is a separate device buffer.
    """
    from tinygrad.schedule import memory
    from tinygrad.uop import ops

    if getattr(ops.UOp.contiguous_view, "webgpu_views", False):
        return
    original_view, original_plan = ops.UOp.contiguous_view, memory._can_plan

    def contiguous_view(self):
        if not _on_webgpu(self.device):
            return original_view(self)
        # tinygrad 0.14's UOp.contiguous_view without its blanket WEBGPU refusal, plus the alignment check
        from tinygrad.schedule.rangeify import pm_mops
        from tinygrad.uop.symbolic import symbolic

        index = self.flatten().index(ops.UOp.range(self.numel(), 0))
        out = ops.graph_rewrite(index, pm_mops + symbolic + ops.pm_contiguous_view_offset, ctx=self, name="contiguous_view_offset")
        if out.op is not ops.Ops.INDEX or not (base := out.src[0]).tag:
            return None
        if (offset := out.src[1]).op is not ops.Ops.CONST or not isinstance(offset.val, int):
            return None
        if offset.val * base.dtype.itemsize % VIEW_ALIGNMENT:
            return None
        return base.rtag(None), offset.val

    def can_plan(buffer, held_buffers) -> bool:
        return buffer not in held_buffers if _on_webgpu(buffer.device) else original_plan(buffer, held_buffers)

    contiguous_view.webgpu_views = can_plan.webgpu_views = True
    ops.UOp.contiguous_view = contiguous_view
    memory._can_plan = can_plan


def register_runtime() -> None:
    name = "tinygrad.runtime.ops_webgpu"
    previous = sys.modules.get(name)
    if previous is not None and getattr(previous, "WebGpuDevice", None) is not WebGpuDevice:
        raise RuntimeError("Another WebGPU runtime is already active in this process.")
    if previous is None:
        module = types.ModuleType(name)
        module.WebGpuDevice = WebGpuDevice
        sys.modules[name] = module
    _enable_buffer_views()


def probe_training() -> dict:
    """One SGD step on the GPU, checked against the analytic answer."""
    import numpy as np
    from tinygrad import Tensor
    from tinygrad.helpers import Context

    register_runtime()
    with Context(DEV="WEBGPU"):
        device = Device["WEBGPU"]
        before = device.dispatches
        values = Tensor(np.array([1, 2, 3, 4], dtype=np.float32), device="WEBGPU")
        (gradient,) = values.square().mean().gradient(values)
        updated = (values - 0.1 * gradient).realize()
        gradient_values, updated_values = gradient.numpy(), updated.numpy()
        np.testing.assert_allclose(gradient_values, [0.5, 1, 1.5, 2], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(updated_values, [0.95, 1.9, 2.85, 3.8], rtol=1e-5, atol=1e-6)
        return {
            "adapter": device.info,
            "device": values.device,
            "gpu_dispatches": device.dispatches - before,
            "gradient": gradient_values.tolist(),
            "updated_weights": updated_values.tolist(),
        }


if __name__ == "__main__":
    import json

    print(json.dumps(probe_training(), indent=2))
