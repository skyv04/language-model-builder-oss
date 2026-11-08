"""The app window on macOS: a native WebKit (WKWebView) window through pywebview.

Usage: python -m model_builder.webview_window URL
Exit status 3 means pywebview is not available, so the launcher can try another window.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

UNAVAILABLE = 3
TITLE = "Language Model Builder"


def _brand_application() -> None:
    """Show the app's name in the menu bar and its icon in the Dock, instead of Python's (best effort)."""
    try:
        from AppKit import NSApplication, NSImage
        from Foundation import NSBundle
    except ImportError:
        return
    try:
        info = NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = TITLE
        icon = Path(__file__).resolve().parent / "ui" / "img" / "icon.png"
        image = NSImage.alloc().initWithContentsOfFile_(str(icon)) if icon.exists() else None
        if image is not None:
            NSApplication.sharedApplication().setApplicationIconImage_(image)
    except Exception:  # noqa: BLE001 - purely cosmetic
        pass


def main(url: str) -> int:
    try:
        import webview
    except ImportError:
        return UNAVAILABLE
    _brand_application()
    try:
        # Links with target="_blank" (credits, dataset sources) open in the default browser.
        webview.create_window(TITLE, url, width=1400, height=900, min_size=(900, 600), text_select=True, zoomable=True)
        webview.start(private_mode=True, debug=bool(os.environ.get("LMB_DEBUG")))
    except Exception as error:  # noqa: BLE001
        print(f"Couldn't open the app window: {error}", file=sys.stderr)
        return UNAVAILABLE
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) if len(sys.argv) > 1 else 2)
