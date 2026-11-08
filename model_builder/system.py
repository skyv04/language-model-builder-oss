"""What kind of system this is, and how to open links, files and folders on it.

Plain Linux, the Windows Subsystem for Linux (WSL) and macOS each open things differently.
Standard library only and Python 3.8-compatible, because window.py may run under the system Python.
"""

from __future__ import annotations

import functools
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

APP_ID = "language-model-builder"
# Chromium-family browsers that Windows installs for every user, relative to the C: drive.
WINDOWS_BROWSERS = (
    "Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "Program Files/Microsoft/Edge/Application/msedge.exe",
    "Program Files/Google/Chrome/Application/chrome.exe",
    "Program Files (x86)/Google/Chrome/Application/chrome.exe",
)


def is_macos() -> bool:
    return sys.platform == "darwin"


@functools.lru_cache(maxsize=None)
def is_wsl() -> bool:
    """True inside the Windows Subsystem for Linux (WSL 1 or 2)."""
    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        release = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in release or "wsl" in release


def os_label() -> str:
    """The operating system's name and version, e.g. "Ubuntu 24.04.1 LTS (WSL)" or "macOS 15.1"."""
    if is_macos():
        version = platform.mac_ver()[0]
        return f"macOS {version}" if version else "macOS"
    name = None
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                name = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    name = name or platform.system() or "Unknown system"
    return f"{name} (WSL)" if is_wsl() else name


def file_manager_name() -> str:
    """What this system calls the app that open_target() shows folders in."""
    if is_macos():
        return "Finder"
    return "File Explorer" if is_wsl() else "Files"


def sysctl(name: str) -> str:
    """A macOS sysctl value such as hw.memsize, or "" when there is none."""
    if not is_macos():
        return ""
    return _output(["/usr/sbin/sysctl", "-n", name], timeout=5) or ""


def _output(command: list[str], timeout: float = 10.0) -> str | None:
    try:
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                   errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip()
    return output if completed.returncode == 0 and output else None


def wsl_path(windows_path: str) -> str | None:
    """A Windows path (C:\\Users\\…) as WSL sees it (/mnt/c/Users/…)."""
    return _output(["wslpath", "-u", windows_path]) if shutil.which("wslpath") else None


def windows_path(path: str) -> str | None:
    """A WSL path as Windows programs see it (C:\\… or \\\\wsl.localhost\\…)."""
    return _output(["wslpath", "-w", str(path)]) if shutil.which("wslpath") else None


@functools.lru_cache(maxsize=None)
def windows_drive() -> Path | None:
    """Where WSL mounts the Windows C: drive (usually /mnt/c)."""
    if not is_wsl():
        return None
    for candidate in (wsl_path("C:\\"), "/mnt/c"):
        if candidate and os.path.isdir(candidate):
            return Path(candidate)
    return None


def _windows_program(relative: str, name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    drive = windows_drive()
    return str(drive / relative) if drive and (drive / relative).is_file() else None


@functools.lru_cache(maxsize=None)
def windows_env(name: str) -> str | None:
    """A Windows environment variable such as LOCALAPPDATA, read through cmd.exe (WSL only)."""
    if not is_wsl():
        return None
    cmd = _windows_program("Windows/System32/cmd.exe", "cmd.exe")
    if not cmd:
        return None
    drive = windows_drive()
    try:
        # /u makes cmd write UTF-16, so user names outside the console code page survive.
        completed = subprocess.run([cmd, "/d", "/u", "/c", "echo", f"%{name}%"], stdin=subprocess.DEVNULL,
                                   capture_output=True, timeout=10, cwd=str(drive) if drive else None)
    except (OSError, subprocess.TimeoutExpired):
        return None
    raw = completed.stdout
    text = raw.decode("utf-16-le", errors="replace") if b"\x00" in raw else raw.decode("utf-8", errors="replace")
    value = text.strip()
    return value if completed.returncode == 0 and value and value != f"%{name}%" else None


@functools.lru_cache(maxsize=None)
def windows_downloads() -> Path | None:
    """The Windows user's Downloads folder, seen from WSL."""
    profile = windows_env("USERPROFILE")
    converted = wsl_path(profile.rstrip("\\") + "\\Downloads") if profile else None
    return Path(converted) if converted and os.path.isdir(converted) else None


def windows_browsers() -> list[str]:
    """Edge or Chrome installed on the Windows side, best first (WSL only)."""
    drive = windows_drive()
    if drive is None:
        return []
    return [str(drive / relative) for relative in WINDOWS_BROWSERS if (drive / relative).is_file()]


def windows_browser_profile() -> str | None:
    """A Windows folder for the app window's own browser profile, so the window stays a separate process."""
    local = windows_env("LOCALAPPDATA")
    if not local or not re.match(r"^[A-Za-z]:\\", local):
        return None
    return local.rstrip("\\") + f"\\{APP_ID}\\browser-profile"


def opener_commands(target: str) -> list[list[str]]:
    """Commands that open a URL, file or folder with the user's default application, best first."""
    if is_macos():
        return [["/usr/bin/open" if os.path.exists("/usr/bin/open") else "open", target]]
    commands = []
    if is_wsl():
        wslview = shutil.which("wslview")
        if wslview:
            commands.append([wslview, target])
        explorer = _windows_program("Windows/explorer.exe", "explorer.exe")
        converted = target if "://" in target else windows_path(target)
        if explorer and converted:
            commands.append([explorer, converted])
    xdg_open, gio = shutil.which("xdg-open"), shutil.which("gio")
    if xdg_open:
        commands.append([xdg_open, target])
    if gio:
        commands.append([gio, "open", target])
    return commands


def open_target(target: str, env: dict | None = None) -> bool:
    """Open a URL, file or folder without waiting; False when this system has no way to."""
    for command in opener_commands(target):
        try:
            subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True, env=env)
            return True
        except OSError:
            continue
    return False
