"""Command line: `language-model-builder` opens the app; sub-commands help with scripting and setup."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version


def _doctor() -> int:
    from . import devices, paths, system
    from .app import find_chromium, local_typelib_dirs, webkit_python
    from .devices import in_proot

    def package(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "missing"

    print(f"System        {system.os_label()} · {platform.machine()}")
    print(f"Python        {sys.version.split()[0]} ({sys.executable})")
    extras = ("pywebview",) if system.is_macos() else ("tinygrad", "wgpu")
    for name in ("torch", "numpy", "tokenizers", "safetensors", "pyarrow", *extras):
        print(f"{name:<13} {package(name)}")
    print(f"Data folder   {paths.data_dir()}")
    free = shutil.disk_usage(paths.data_dir()).free
    print(f"Free disk     {free / 1e9:.1f} GB")
    cpu = devices.cpu_device()
    print(f"CPU           {cpu['name']} ({cpu['detail']})")
    if sys.stdout.isatty():
        print("GPU           looking…", end="\r", flush=True)
    gpus = devices.run_probe()
    found = ", ".join(f"{gpu['name']} via {gpu['label']}" for gpu in gpus)
    print(f"GPU           {found or 'none found (training uses the CPU)'}")
    if not system.is_macos():
        print(f"PRoot         {'yes' if in_proot() else 'no'}")
    webkit, chromium = webkit_python(), find_chromium()
    own = "pywebview (run: bash install.sh)" if system.is_macos() else "WebKitGTK (gir1.2-webkit2-4.1)"
    if webkit:
        window = "the Mac's own WebKit, through pywebview" if system.is_macos() else f"WebKitGTK (through {webkit})"
    elif chromium:
        side = "a Windows browser" if chromium.lower().endswith(".exe") else "a Chromium-family browser"
        window = f"{side} in app mode ({chromium}); install {own} for the app's own window"
    else:
        window = f"the default web browser; install {own} for the app's own window"
    print(f"Window        {window}")
    typelibs = local_typelib_dirs()
    if typelibs:
        print(f"Typelibs      {', '.join(typelibs)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="language-model-builder",
        description="Build a small language model from scratch, step by step "
                    "(unofficial open-source edition for Linux, WSL and macOS).",
    )
    commands = parser.add_subparsers(dest="command")
    app = commands.add_parser("app", help="open the app window (default)")
    app.add_argument("--window", choices=("auto", "webkit", "chromium", "browser"), default="auto",
                     help="which kind of window to use")
    serve = commands.add_parser("serve", help="run only the local engine (for use from a browser)")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--persist", action="store_true", help="keep running while no window is connected")
    commands.add_parser("url", help="print the address of the running engine, including its access token")
    probe = commands.add_parser("probe-gpu", help="look for a GPU that can train models")
    probe.add_argument("--json", action="store_true", help="print one JSON line")
    probe.add_argument("--train", action="store_true", help="also check a training step on the GPU against PyTorch")
    commands.add_parser("doctor", help="check this machine's setup")
    worker = commands.add_parser("worker")  # started by the server
    worker.add_argument("--project", required=True)
    worker.add_argument("--kind", required=True, choices=("pretrain", "sft", "dpo", "calibrate"))
    worker.add_argument("--device", default="cpu", choices=("cpu", "vulkan", "cuda", "mps"))
    worker.add_argument("--job", required=True)
    worker.add_argument("--devices", default="cpu")
    args = parser.parse_args(argv)
    command = args.command or "app"

    if command == "app":
        from .app import main as open_app

        return open_app(getattr(args, "window", "auto"))
    if command == "serve":
        from .server import serve

        def announce(info: dict) -> None:
            print(f"Open http://127.0.0.1:{info['port']}/?token={info['token']}", flush=True)

        return serve(args.port, persist=args.persist, ready=announce)
    if command == "url":
        from .app import app_url, ping, server_info

        info = server_info()
        if not info or not ping(info):
            print("The engine is not running. Start it with: language-model-builder serve", file=sys.stderr)
            return 1
        print(app_url(info))
        return 0
    if command == "probe-gpu":
        from .devices import probe_here

        result = probe_here()
        if args.train and result.get("vulkan"):
            try:
                from .gpu_runtime import probe_training

                result["vulkan_training"] = probe_training()
            except Exception as error:  # noqa: BLE001
                result["vulkan_training"] = {"ok": False, "error": str(error)[:300]}
        print(json.dumps(result) if args.json else json.dumps(result, indent=2), flush=True)
        return 0
    if command == "doctor":
        return _doctor()
    if command == "worker":
        from .worker import run

        return run(args.project, args.kind, args.device, args.job, [d for d in args.devices.split(",") if d])
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
