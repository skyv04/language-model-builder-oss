"""Compare CPU (torch) and Vulkan GPU (tinygrad) training speed and memory for a few model sizes.

Each measurement runs in its own subprocess, one after another (with a cool-down pause in between),
so peak memory is per run and a GPU fault cannot take the other runs down.  Runs whose estimated
memory exceeds ``--memory-limit-mb`` are skipped, and a running measurement aborts if its sampled
memory crosses the limit.  The GPU/CPU column compares against the fastest CPU thread count measured.

    .venv/bin/python tools/benchmark_devices.py --json results.json
    ... --configs tiny --devices gpu --custom 10000,256,512,8,8,4   # V,C,d,L,H,B
    ... --devices cpu --threads 6,4                                 # CPU at several thread counts
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MiB = 2**20
CONFIGS = {
    "tiny": dict(vocab_size=4096, context=128, width=128, blocks=4, heads=4, batch=16),
    "medium": dict(vocab_size=10000, context=256, width=256, blocks=6, heads=8, batch=8),
    "wide-vocab": dict(vocab_size=50261, context=256, width=256, blocks=4, heads=4, batch=4),
    "large": dict(vocab_size=10000, context=256, width=512, blocks=8, heads=8, batch=4),
}
DEVICES = ("cpu", "gpu")
LEARNING_RATE = 3e-4
WARMUP_STEPS = 3  # GPU: the first step compiles, the second captures the TinyJit, the third is the first replay
RESULT_MARK = "BENCHMARK-RESULT "


def _parameters(spec: dict) -> int:
    d = spec["width"]
    return spec["vocab_size"] * d + spec["context"] * d + spec["blocks"] * (12 * d * d + 13 * d) + 2 * d


def estimate_mb(spec: dict, device: str) -> float:
    """Rough peak memory of one run (process plus GPU buffers), calibrated on the default configs.

    Weights, gradients and two AdamW moments are 16 bytes per parameter; activations per token
    scale with blocks x (width, heads x context) plus a few vocabulary-sized rows for the logits.
    """
    tokens, d, layers = spec["batch"] * spec["context"], spec["width"], spec["blocks"]
    attention = 2 * spec["heads"] * spec["context"]
    if device == "gpu":
        vocab = -(-spec["vocab_size"] // 128) * 128
        return 450 + (16 * _parameters({**spec, "vocab_size": vocab}) + 4 * tokens * (layers * (8 * d + attention) + 3 * vocab)) / MiB
    return 350 + (16 * _parameters(spec) + 4 * tokens * (layers * (17 * d + attention) + 3 * spec["vocab_size"])) / MiB


def usable_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _pss_mb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/smaps_rollup") as handle:
            for line in handle:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def _children(pid: int) -> list[int]:
    found = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as handle:
                    if int(handle.read().rsplit(")", 1)[1].split()[1]) == pid:
                        found.append(int(entry))
            except (OSError, ValueError, IndexError):
                continue
    return found


class MemorySampler(threading.Thread):
    """Peak proportional set size of this process plus its children (tinygrad's compile workers).

    GPU buffers live in system RAM on integrated GPUs but outside the process's RSS, so the
    current tinygrad allocation is added for the limit check.
    """

    def __init__(self, limit_mb: float, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.limit_mb, self.interval = limit_mb, interval
        self.peak_mb = 0.0
        self.gpu_mb = lambda: 0.0
        self.stopped = threading.Event()

    def run(self) -> None:
        pid = os.getpid()
        while not self.stopped.wait(self.interval):
            total = _pss_mb(pid) + sum(_pss_mb(child) for child in _children(pid))
            self.peak_mb = max(self.peak_mb, total)
            if total + self.gpu_mb() > self.limit_mb:
                print(f"aborting: {total:.0f} MB in processes + {self.gpu_mb():.0f} MB GPU buffers > {self.limit_mb:.0f} MB",
                      file=sys.stderr, flush=True)
                os._exit(3)


def run_worker(name: str, spec: dict, device: str, seconds: float, threads: int, limit_mb: float) -> dict:
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT))
    from model_builder.model import GPT, ModelConfig, make_optimizer

    spec = dict(spec)
    batch = spec.pop("batch")
    result = {"config": name, "device": device, **spec, "batch": batch, "parameters": _parameters({**spec, "batch": batch})}
    torch.set_num_threads(threads)
    adapter = None
    if device == "gpu":
        from model_builder.gpu_model import gpu_available

        adapter = gpu_available()
        if adapter is None:
            return {**result, "skipped": "no physical GPU adapter"}
    sampler = MemorySampler(limit_mb)
    sampler.start()
    rng = np.random.default_rng(0)
    shape = (batch, spec["context"])
    batches = [(rng.integers(0, spec["vocab_size"], shape, dtype=np.int32),
                rng.integers(0, spec["vocab_size"], shape, dtype=np.int32)) for _ in range(4)]
    torch.manual_seed(0)
    start = time.perf_counter()
    model = GPT(ModelConfig(**spec))
    optimizer = make_optimizer(model, LEARNING_RATE)
    if device == "gpu":
        from tinygrad.helpers import GlobalCounters

        from model_builder.gpu_model import GpuTrainer

        trainer = GpuTrainer(model, optimizer, 0)
        sampler.gpu_mb = lambda: GlobalCounters.mem_used / MiB

        def step(tokens, targets) -> float:
            return trainer.train_step(tokens, targets, LEARNING_RATE)
    else:

        def step(tokens, targets) -> float:
            x, y = torch.from_numpy(tokens).long(), torch.from_numpy(targets).long()
            loss = model.loss(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            return float(loss)

    for index in range(WARMUP_STEPS):
        step(*batches[index % len(batches)])
        if index == 0:
            result["first_step_s"] = round(time.perf_counter() - start, 2)
    result["warmup_s"] = round(time.perf_counter() - start, 2)
    steps, began = 0, time.perf_counter()
    while steps < 3 or time.perf_counter() - began < seconds:
        loss = step(*batches[steps % len(batches)])
        steps += 1
    elapsed = time.perf_counter() - began
    sampler.stopped.set()
    result.update(
        steps=steps,
        step_ms=round(1000 * elapsed / steps, 1),
        tokens_per_second=round(steps * batch * spec["context"] / elapsed),
        final_loss=round(loss, 4),
        peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        peak_pss_with_workers_mb=round(sampler.peak_mb),
        threads=threads,
    )
    if device == "gpu":
        result.update(gpu_buffers_mb=round(sampler.gpu_mb()), adapter=adapter)
    return result


def _measure(name: str, spec: dict, device: str, threads: int, args) -> dict:
    estimate = estimate_mb(spec, device)
    base = {"config": name, "device": device, **spec, "threads": threads, "estimated_mb": round(estimate)}
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps([name, spec, device]),
               "--seconds", str(args.seconds), "--threads", str(threads), "--memory-limit-mb", str(args.memory_limit_mb)]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {**base, "error": f"timed out after {args.timeout} s"}
    for line in done.stdout.splitlines():
        if line.startswith(RESULT_MARK):
            return {**json.loads(line[len(RESULT_MARK):]), "estimated_mb": round(estimate)}
    tail = [line for line in done.stderr.strip().splitlines() if "Forcing backend" not in line][-3:]
    return {**base, "error": f"exit code {done.returncode}: " + " | ".join(tail)}


def _table(results: list[dict]) -> str:
    rows = [("config", "device", "threads", "tok/s", "step ms", "warmup s", "peak RSS MB", "+workers MB", "GPU buf MB",
             "GPU/CPU")]
    best_cpu: dict[str, int] = {}
    for r in results:
        if r["device"] == "cpu" and r.get("tokens_per_second"):
            best_cpu[r["config"]] = max(best_cpu.get(r["config"], 0), r["tokens_per_second"])
    for r in results:
        threads = str(r.get("threads", "")) if r["device"] == "cpu" else ""
        if "tokens_per_second" not in r:
            rows.append((r["config"], r["device"], threads, r.get("skipped") or r.get("error", "?"), "", "", "", "", "", ""))
            continue
        ratio = ""
        if r["device"] == "gpu" and best_cpu.get(r["config"]):
            ratio = f"{r['tokens_per_second'] / best_cpu[r['config']]:.2f}x"
        rows.append((r["config"], r["device"], threads, f"{r['tokens_per_second']:,}", f"{r['step_ms']:.1f}",
                     f"{r['warmup_s']:.1f}", str(r["peak_rss_mb"]), str(r["peak_pss_with_workers_mb"]),
                     str(r.get("gpu_buffers_mb", "")), ratio))
    widths = [max(len(row[i]) for row in rows if len(row[i]) < 40) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--configs", default=",".join(CONFIGS), help=f"comma-separated subset of {', '.join(CONFIGS)}")
    parser.add_argument("--custom", action="append", default=[], metavar="V,C,d,L,H,B",
                        help="extra config: vocabulary, context, width, blocks, heads, batch")
    parser.add_argument("--devices", default=",".join(DEVICES))
    parser.add_argument("--seconds", type=float, default=15.0, help="timed training per run, after warm-up")
    parser.add_argument("--threads", default=str(usable_cores()),
                        help="torch CPU threads; a comma-separated list measures the CPU at each (default: usable cores)")
    parser.add_argument("--cooldown", type=float, default=20.0, help="pause in seconds between runs (thermal throttling)")
    parser.add_argument("--memory-limit-mb", type=float, default=1536.0)
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per run")
    parser.add_argument("--json", type=Path, help="write all results to this file")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    thread_counts = [int(part) for part in args.threads.split(",") if part]
    if args.worker:
        name, spec, device = json.loads(args.worker)
        result = run_worker(name, spec, device, args.seconds, thread_counts[0], args.memory_limit_mb)
        print(RESULT_MARK + json.dumps(result), flush=True)
        return 0
    configs = {name: CONFIGS[name] for name in args.configs.split(",") if name}
    for text in args.custom:
        v, c, d, l, h, b = (int(part) for part in text.split(","))
        configs[f"V{v}-C{c}-d{d}-L{l}-H{h}-B{b}"] = dict(vocab_size=v, context=c, width=d, blocks=l, heads=h, batch=b)
    results, last_run = [], None
    for name, spec in configs.items():
        for device in args.devices.split(","):
            for threads in thread_counts if device == "cpu" else thread_counts[:1]:
                estimate = estimate_mb(spec, device)
                if estimate > args.memory_limit_mb:
                    results.append({"config": name, "device": device, **spec, "threads": threads, "estimated_mb": round(estimate),
                                    "skipped": f"estimated {estimate:.0f} MB exceeds the {args.memory_limit_mb:.0f} MB limit"})
                    continue
                if last_run is not None:
                    time.sleep(max(0.0, args.cooldown - (time.monotonic() - last_run)))
                print(f"measuring {name} on {device} ({threads} threads) ...", file=sys.stderr, flush=True)
                results.append(_measure(name, spec, device, threads, args))
                last_run = time.monotonic()
    print(_table(results))
    if args.json:
        report = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "machine": platform.machine(),
            "usable_cores": usable_cores(),
            "settings": {"seconds": args.seconds, "threads": thread_counts, "cooldown": args.cooldown,
                         "warmup_steps": WARMUP_STEPS, "learning_rate": LEARNING_RATE,
                         "memory_limit_mb": args.memory_limit_mb},
            "results": results,
        }
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
