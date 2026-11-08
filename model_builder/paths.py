"""Filesystem locations: the XDG base directories on Linux (including WSL), ~/Library on macOS."""

from __future__ import annotations

import os
from pathlib import Path

from . import system

APP_ID = "language-model-builder"


def _xdg(variable: str, fallback: str, macos: str | None = None) -> Path:
    """An XDG folder when its variable is set; otherwise the platform's usual place."""
    value = os.environ.get(variable)
    if value and os.path.isabs(value):
        return Path(value)
    return Path.home() / (macos if macos and system.is_macos() else fallback)


def data_dir() -> Path:
    override = os.environ.get("LMB_DATA_DIR")
    path = Path(override) if override else _xdg("XDG_DATA_HOME", ".local/share", "Library/Application Support") / APP_ID
    path.mkdir(parents=True, exist_ok=True)
    return path


def projects_dir() -> Path:
    path = data_dir() / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_dir() -> Path:
    path = data_dir() / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    override = os.environ.get("LMB_CACHE_DIR")
    path = Path(override) if override else _xdg("XDG_CACHE_HOME", ".cache", "Library/Caches") / APP_ID
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR")
    path = Path(base) / APP_ID if base and os.path.isdir(base) else cache_dir() / "run"
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def downloads_dir() -> Path:
    """The user's Downloads folder: on WSL the Windows one; elsewhere including localized names
    set in ~/.config/user-dirs.dirs."""
    override = os.environ.get("XDG_DOWNLOAD_DIR")
    if not override and system.is_wsl():
        windows = system.windows_downloads()
        override = str(windows) if windows else None
    if not override:
        config = _xdg("XDG_CONFIG_HOME", ".config") / "user-dirs.dirs"
        try:
            for line in config.read_text(encoding="utf-8").splitlines():
                if line.startswith("XDG_DOWNLOAD_DIR="):
                    override = line.split("=", 1)[1].strip().strip('"').replace("$HOME", str(Path.home()))
        except OSError:
            pass
    path = Path(override) if override and override != str(Path.home()) else Path.home() / "Downloads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ui_dir() -> Path:
    return Path(__file__).resolve().parent / "ui"
