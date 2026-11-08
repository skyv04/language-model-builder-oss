"""Open the app: start (or reuse) the local engine server, then show the UI in a window.

Window options, in order:
- the app's own window: WebKitGTK on Linux (window.py, run by any Python that has PyGObject;
  the engine's own Python often does not) or WKWebView through pywebview on macOS
  (webview_window.py);
- a Chromium-family browser in --app mode (under WSL also Edge or Chrome on the Windows side);
- the default web browser.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from . import paths, system
from .devices import engine_env, in_proot

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
CHROMIUM_NAMES = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "brave-browser", "microsoft-edge")
MAC_BROWSERS = ("Google Chrome", "Chromium", "Microsoft Edge", "Brave Browser")
WINDOW_UNAVAILABLE = 3  # exit status of window.py and webview_window.py when they cannot open a window
HANDED_OFF = -1  # a browser passed the window to one it already had open (that launch reports the close)
QUICK_EXIT_SECONDS = 5.0


def server_info() -> dict | None:
    try:
        return json.loads((paths.runtime_dir() / "server.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def ping(info: dict, timeout: float = 2.0) -> bool:
    request = urllib.request.Request(f"http://127.0.0.1:{info['port']}/api/ping", headers={
        "X-LMB-Token": info["token"], "Host": f"127.0.0.1:{info['port']}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read()).get("ok") is True
    except (OSError, ValueError):
        return False


def ensure_server(timeout: float = 60.0) -> dict:
    """Return {port, token, pid} of a running engine server, starting one if needed."""
    info = server_info()
    if info and ping(info):
        return info
    log = open(paths.runtime_dir() / "server.log", "a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "model_builder", "serve"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True, env=engine_env(), cwd=str(PACKAGE_PARENT),
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = server_info()
        if info and info.get("pid") == process.pid and ping(info):
            return info
        if process.poll() is not None:
            info = server_info()  # another launcher may have won the race
            if info and ping(info):
                return info
            raise RuntimeError(f"The engine server exited early; see {paths.runtime_dir() / 'server.log'}")
        time.sleep(0.2)
    raise RuntimeError("The engine server did not start in time.")


def app_url(info: dict) -> str:
    return f"http://127.0.0.1:{info['port']}/?token={info['token']}"


def _gi_pythons() -> list[str]:
    if system.is_macos():
        return []  # no WebKitGTK; and /usr/bin/python3 there may offer to install developer tools
    candidates = [sys.executable, "/usr/bin/python3", shutil.which("python3") or "", "/usr/bin/python3.12",
                  "/usr/bin/python3.11", "/usr/bin/python3.10"]
    seen, result = set(), []
    for candidate in candidates:
        if candidate and candidate not in seen and os.path.exists(candidate):
            seen.add(candidate)
            result.append(candidate)
    return result


def local_typelib_dirs() -> list[str]:
    """WebKitGTK typelibs unpacked by install.sh when the system package is missing (no root needed)."""
    root = paths.data_dir() / "runtime-typelibs"
    return [str(path) for path in root.glob("usr/lib/*/girepository-1.0") if path.is_dir()]


def window_env() -> dict:
    env = engine_env()
    extra = local_typelib_dirs()
    if extra:
        env["GI_TYPELIB_PATH"] = os.pathsep.join([*extra, *filter(None, [env.get("GI_TYPELIB_PATH")])])
    env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_PARENT), *filter(None, [env.get("PYTHONPATH")])])
    if in_proot():
        env.setdefault("WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS", "1")
        env.setdefault("WEBKIT_DISABLE_COMPOSITING_MODE", "1")
        env.setdefault("LMB_WEBKIT_SOFTWARE", "1")
    elif system.is_wsl():
        # WSLg's GPU path can leave WebKit pages blank; this page is light enough to draw in software.
        env.setdefault("WEBKIT_DISABLE_COMPOSITING_MODE", "1")
        env.setdefault("WEBKIT_DISABLE_DMABUF_RENDERER", "1")
    return env


def run_webkit(url: str) -> int | None:
    """The app's own window; None when nothing here can open one."""
    env = window_env()
    if system.is_macos():
        code = subprocess.call([sys.executable, "-m", "model_builder.webview_window", url], env=env, cwd=str(PACKAGE_PARENT))
        return None if code == WINDOW_UNAVAILABLE else code
    for python in _gi_pythons():
        check = subprocess.run([python, "-c", "import gi"], env=env, capture_output=True)
        if check.returncode != 0:
            continue
        code = subprocess.call([python, "-m", "model_builder.window", url], env=env, cwd=str(PACKAGE_PARENT))
        if code != WINDOW_UNAVAILABLE:
            return code
    return None


def webkit_python() -> str | None:
    """The first Python that can open the app's own window (WebKitGTK, or pywebview on macOS), if any."""
    check = "import sys; from model_builder.window import _load_webkit; sys.exit(0 if _load_webkit() else 3)"
    if system.is_macos():
        check = "import webview"
    for python in [sys.executable] if system.is_macos() else _gi_pythons():
        try:
            completed = subprocess.run([python, "-c", check], env=window_env(), cwd=str(PACKAGE_PARENT),
                                       capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0:
            return python
    return None


def chromium_candidates() -> list[str]:
    """Chromium-family browsers that can show the app in a window of its own (--app mode), best first."""
    if system.is_macos():
        found = []
        for folder in (Path("/Applications"), Path.home() / "Applications"):
            for name in MAC_BROWSERS:
                path = folder / f"{name}.app" / "Contents" / "MacOS" / name
                if path.is_file():
                    found.append(str(path))
        return found
    found = []
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        found = [path for path in map(shutil.which, CHROMIUM_NAMES) if path]
    if system.is_wsl():
        found += system.windows_browsers()
    return list(dict.fromkeys(found))


def find_chromium() -> str | None:
    candidates = chromium_candidates()
    return candidates[0] if candidates else None


def _chromium_command(browser: str, url: str) -> list[str] | None:
    on_windows = system.is_wsl() and browser.lower().endswith(".exe")
    # A profile of its own makes the window a separate browser process that lives as long as the window.
    profile = system.windows_browser_profile() if on_windows else str(paths.data_dir() / "browser-profile")
    if not profile:
        return None
    command = [browser, f"--app={url}", f"--user-data-dir={profile}"]
    if not on_windows and not system.is_macos():
        command.append("--class=language-model-builder")  # matches the .desktop file's StartupWMClass
    command += ["--window-size=1400,900", "--no-first-run", "--no-default-browser-check"]
    if in_proot():
        # PRoot cannot run Chromium's sandbox; --test-type hides the "unsupported flag" warning bar.
        command += ["--no-sandbox", "--test-type", "--disable-gpu", "--disable-dev-shm-usage"]
    command += [flag for flag in os.environ.get("LMB_CHROMIUM_FLAGS", "").split() if flag]
    return command


def _wait_for_window(command: list[str]) -> int | None:
    """Run a browser window until it closes. None if the browser gave up at once (for example a
    snap package that cannot start here); HANDED_OFF if it passed the window to a running copy."""
    try:
        process = subprocess.Popen(command, env=engine_env(), stdin=subprocess.DEVNULL)
    except OSError:
        return None
    try:
        code = process.wait(timeout=QUICK_EXIT_SECONDS)
    except subprocess.TimeoutExpired:
        return process.wait()
    return HANDED_OFF if code == 0 else None


def run_chromium(url: str) -> int | None:
    for browser in chromium_candidates():
        command = _chromium_command(browser, url)
        code = _wait_for_window(command) if command else None
        if code is not None:
            return code
    return None


def run_browser(url: str) -> int:
    if not system.open_target(url, env=engine_env()):
        import webbrowser

        webbrowser.open(url)
    print(f"Opened Language Model Builder in your browser: {url}")
    return 0


def notify_closed(info: dict) -> None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{info['port']}/api/window-closed", data=b"{}", method="POST",
        headers={"X-LMB-Token": info["token"], "Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=3).close()
    except OSError:
        pass


def main(window: str = "auto") -> int:
    info = ensure_server()
    url = app_url(info)
    if window == "browser":
        return run_browser(url)
    for choice in ("webkit", "chromium") if window == "auto" else (window,):
        code = run_webkit(url) if choice == "webkit" else run_chromium(url)
        if code == HANDED_OFF:
            return 0
        if code is not None:
            notify_closed(info)
            return code
    if window != "auto":
        hint = ""
        if window == "webkit":
            hint = " (pywebview is missing; run: bash install.sh)" if system.is_macos() else \
                " (install gir1.2-webkit2-4.1, or run: bash install.sh)"
        print(f"The {window} window is not available here{hint}.", file=sys.stderr)
        return 1
    print("No app window toolkit found; opening the browser instead.", file=sys.stderr)
    return run_browser(url)
