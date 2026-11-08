"""The app window on Linux: a WebKitGTK view of the local UI (run with any Python that has PyGObject).

Usage: python -m model_builder.window URL
Exit status 3 means WebKitGTK is not available, so the launcher can try another window.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from urllib.parse import urlsplit

from . import system

UNAVAILABLE = 3
TITLE = "Language Model Builder"


def _load_webkit():
    """Return (Gtk, Gdk, GLib, WebKit2) for WebKitGTK on GTK 3, or None (WebKit 6.0 on GTK 4 is not supported)."""
    try:
        import gi

        # Pin Gdk too: left unpinned, PyGObject may pick Gdk 4.0, which then clashes with Gtk 3.0.
        gi.require_version("Gtk", "3.0")
        gi.require_version("Gdk", "3.0")
    except (ImportError, ValueError):
        return None
    for version in ("4.1", "4.0"):
        try:
            gi.require_version("WebKit2", version)
        except ValueError:
            continue
        try:
            from gi.repository import Gdk, GLib, Gtk, WebKit2
        except ImportError:
            return None
        return Gtk, Gdk, GLib, WebKit2
    return None


def main(url: str) -> int:
    modules = _load_webkit()
    if modules is None:
        return UNAVAILABLE
    Gtk, Gdk, GLib, WebKit2 = modules
    if not Gtk.init_check(sys.argv)[0]:
        print("No graphical display is available.", file=sys.stderr)
        return UNAVAILABLE
    GLib.set_prgname("language-model-builder")
    GLib.set_application_name(TITLE)
    origin = urlsplit(url)
    icon = Path(__file__).resolve().parent / "ui" / "img" / "icon.png"

    window = Gtk.Window(title=TITLE)
    window.set_default_size(1400, 900)
    with warnings.catch_warnings():  # deprecated, but still what X11 docks match StartupWMClass against
        warnings.simplefilter("ignore", DeprecationWarning)
        window.set_wmclass("language-model-builder", "language-model-builder")
    if icon.exists():
        try:
            window.set_icon_from_file(str(icon))
        except GLib.Error:
            pass
    screen = Gdk.Screen.get_default()
    if screen is not None:
        monitor = screen.get_display().get_primary_monitor() or screen.get_display().get_monitor(0)
        if monitor is not None:
            area = monitor.get_workarea()
            window.set_default_size(min(1400, area.width - 40), min(900, area.height - 40))

    context = WebKit2.WebContext.new_ephemeral() if hasattr(WebKit2.WebContext, "new_ephemeral") else WebKit2.WebContext.get_default()
    view = WebKit2.WebView.new_with_context(context)
    settings = view.get_settings()
    settings.set_enable_developer_extras(bool(os.environ.get("LMB_DEBUG")))
    settings.set_javascript_can_access_clipboard(True)
    settings.set_enable_back_forward_navigation_gestures(False)
    if os.environ.get("LMB_WEBKIT_SOFTWARE") == "1":
        settings.set_hardware_acceleration_policy(WebKit2.HardwareAccelerationPolicy.NEVER)

    def open_externally(target: str) -> None:
        system.open_target(target)

    def decide_policy(_view, decision, kind):
        if kind in (WebKit2.PolicyDecisionType.NAVIGATION_ACTION, WebKit2.PolicyDecisionType.NEW_WINDOW_ACTION):
            target = decision.get_navigation_action().get_request().get_uri()
            parts = urlsplit(target)
            if parts.scheme in ("http", "https") and parts.netloc != origin.netloc:
                open_externally(target)
                decision.ignore()
                return True
            if kind == WebKit2.PolicyDecisionType.NEW_WINDOW_ACTION:
                decision.ignore()
                return True
        return False

    view.connect("decide-policy", decide_policy)
    view.connect("context-menu", lambda *_: not os.environ.get("LMB_DEBUG"))

    def on_key(_window, event) -> bool:
        """Browser-style shortcuts: F5 or Ctrl+R reloads, Ctrl +/-/0 zooms."""
        name = Gdk.keyval_name(event.keyval) or ""
        ctrl = bool(event.state & Gdk.ModifierType.CONTROL_MASK)
        if name == "F5" or (ctrl and name.lower() == "r"):
            view.reload()
        elif ctrl and name in ("plus", "equal", "KP_Add"):
            view.set_zoom_level(min(round(view.get_zoom_level() + 0.1, 1), 2.0))
        elif ctrl and name in ("minus", "KP_Subtract"):
            view.set_zoom_level(max(round(view.get_zoom_level() - 0.1, 1), 0.5))
        elif ctrl and name in ("0", "KP_0"):
            view.set_zoom_level(1.0)
        else:
            return False
        return True

    window.connect("key-press-event", on_key)
    window.add(view)
    window.connect("destroy", lambda *_: Gtk.main_quit())
    view.load_uri(url)
    window.show_all()
    Gtk.main()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) if len(sys.argv) > 1 else 2)
