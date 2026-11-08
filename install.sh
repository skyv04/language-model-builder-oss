#!/bin/bash
# Install Language Model Builder (open-source edition) for the current user. No root access is needed.
# Works on Linux (Debian, Ubuntu and others, including WSL on Windows and PRoot on Android) and macOS.
#
#   bash install.sh              install; pick GPU support automatically; add the app to your menu or ~/Applications
#   bash install.sh --gpu        Linux: also install the Vulkan GPU backend when no Vulkan driver was detected
#   bash install.sh --cpu-only   skip GPU backends
#   bash install.sh --no-menu    skip the menu entry (Linux) or app (macOS), the icons and the command
#   bash install.sh --uninstall  remove the menu entry or app, the icons and the command (your projects are kept)
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
cd "$root"
app_id=language-model-builder
app_name="Language Model Builder"
data_home=${XDG_DATA_HOME:-$HOME/.local/share}
bin_dir=$HOME/.local/bin
mac_app="$HOME/Applications/$app_name.app"
gpu=auto
menu=1
uninstall=0

for option in "$@"; do
    case "$option" in
        --gpu) gpu=yes ;;
        --cpu-only|--no-gpu) gpu=no ;;
        --no-menu) menu=0 ;;
        --uninstall) uninstall=1 ;;
        -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $option (see --help)" >&2; exit 2 ;;
    esac
done

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
fail() {  # a message, then optional hint lines
    printf '\033[1;31mError:\033[0m %s\n' "$1" >&2
    shift
    while [ "$#" -gt 0 ]; do printf '       %s\n' "$1" >&2; shift; done
    exit 1
}

os=$(uname -s)
case "$os" in
    Linux|Darwin) ;;
    *) fail "This installer supports Linux and macOS. On Windows, run it inside WSL (see README.md)." ;;
esac
wsl=0
if [ "$os" = Linux ] && { [ -n "${WSL_DISTRO_NAME:-}${WSL_INTEROP:-}" ] || grep -qiE 'microsoft|wsl' /proc/sys/kernel/osrelease 2>/dev/null; }; then
    wsl=1
fi

in_proot() {
    local tracer
    [ -n "${PROOT_TMP_DIR:-}${PROOT_LOADER:-}" ] && return 0
    tracer=$(awk '/^TracerPid:/ {print $2}' /proc/self/status 2>/dev/null || echo 0)
    [ "${tracer:-0}" != 0 ] && grep -q proot "/proc/$tracer/comm" 2>/dev/null
}

refresh_menus() {
    command -v update-desktop-database >/dev/null && update-desktop-database -q "$data_home/applications" 2>/dev/null || true
    command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t "$data_home/icons/hicolor" 2>/dev/null || true
}

ours() {  # is $1 a symlink to this copy's launcher? (plain readlink: older macOS has no readlink -f)
    [ -L "$1" ] && [ "$(readlink "$1")" = "$root/bin/$app_id" ]
}

if [ "$uninstall" -eq 1 ]; then
    if [ "$os" = Darwin ]; then
        if grep -q "io.github.skyv04.$app_id" "$mac_app/Contents/Info.plist" 2>/dev/null; then rm -rf "$mac_app"; fi
        data_dir="$HOME/Library/Application Support/$app_id"
    else
        rm -f "$data_home/applications/$app_id.desktop"
        find "$data_home/icons/hicolor" -path "*/apps/$app_id.*" -delete 2>/dev/null || true
        refresh_menus
        data_dir="$data_home/$app_id"
    fi
    if ours "$bin_dir/$app_id"; then rm -f "$bin_dir/$app_id"; fi
    say "Removed the app's menu entry or app bundle, its icons and the command."
    note "Your projects and datasets are still in ${LMB_DATA_DIR:-$data_dir};"
    note "delete that folder and $root/.venv to remove everything."
    exit 0
fi

# Preloaded host shims (common in PRoot on Android) can break pip, PyTorch and GPU drivers.
if in_proot; then unset LD_PRELOAD; fi
if [ "$wsl" -eq 1 ]; then
    case "$root" in
        /mnt/?/*)
            note "Note: this copy is on a Windows drive ($root)."
            note "Installing and training are much faster from WSL's own disk: clone the repository into ~ and run install.sh there." ;;
    esac
fi

# ---- Python ----
python_ok() {  # Python 3.10 or newer; at most 3.12 on Intel Macs, whose last PyTorch release (2.2) stops there
    "$1" - >/dev/null 2>&1 <<'PY'
import platform, sys
newest = (3, 12) if sys.platform == "darwin" and platform.machine() == "x86_64" else (3, 99)
sys.exit(0 if (3, 10) <= sys.version_info[:2] <= newest else 1)
PY
}
python_candidates() {  # one path per line, best first
    local name
    if [ -n "${LMB_PYTHON:-}" ]; then
        command -v "$LMB_PYTHON" || echo "$LMB_PYTHON"
        return 0
    fi
    for name in python3.12 python3.13 python3.11 python3.10 python3.14 python3; do
        command -v "$name" || true
        if [ "$os" = Darwin ]; then  # Homebrew and python.org, in case they are not on PATH yet
            echo "/opt/homebrew/bin/$name"
            echo "/usr/local/bin/$name"
            echo "/Library/Frameworks/Python.framework/Versions/${name#python}/bin/$name"
        fi
    done
}
python=""
while IFS= read -r candidate; do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    # Without the developer tools, Apple's /usr/bin/python3 only offers to install them (and is 3.9 anyway).
    if [ "$os" = Darwin ] && [ "$candidate" = /usr/bin/python3 ]; then continue; fi
    if python_ok "$candidate"; then python=$candidate; break; fi
done < <(python_candidates)
if [ -z "$python" ]; then
    if [ -n "${LMB_PYTHON:-}" ]; then
        fail "LMB_PYTHON=$LMB_PYTHON is not a usable Python 3.10 or newer (3.10 to 3.12 on Intel Macs)."
    elif [ "$os" = Darwin ] && [ "$(uname -m)" = x86_64 ]; then
        fail "Python 3.10, 3.11 or 3.12 is required (PyTorch has no newer builds for Intel Macs)." \
            "Install it with Homebrew (https://brew.sh):  brew install python@3.12"
    elif [ "$os" = Darwin ]; then
        fail "Python 3.10 or newer is required." "Install it with Homebrew (https://brew.sh):  brew install python@3.12"
    else
        fail "Python 3.10 or newer is required." "Debian/Ubuntu:  sudo apt install python3 python3-venv" \
            "Fedora:  sudo dnf install python3     Arch:  sudo pacman -S python"
    fi
fi
if [ "$os" = Linux ]; then
    python=$(readlink -f "$python")  # a resolved path keeps the venv on the interpreter's own standard library
fi
say "Using $("$python" --version 2>&1) ($python)"

venv_ok() { [ -x .venv/bin/python ] && python_ok .venv/bin/python && .venv/bin/python -m pip --version >/dev/null 2>&1; }
if ! venv_ok; then
    say "Creating a private Python environment in $root/.venv"
    rm -rf .venv
    venv_options=""
    # On Linux the system's PyGObject (for the WebKitGTK window) stays importable from the environment.
    if [ "$os" = Linux ]; then venv_options=--system-site-packages; fi
    if ! "$python" -m venv $venv_options .venv; then
        rm -rf .venv
        version=$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
        fail "Couldn't create a Python environment." "Debian/Ubuntu:  sudo apt install python3-venv  (or python$version-venv)"
    fi
fi
pip=(.venv/bin/python -m pip install --quiet --disable-pip-version-check)
machine=$(.venv/bin/python -c 'import platform; print(platform.machine())')

# ---- PyTorch: Metal on Apple silicon, CUDA when an NVIDIA GPU is present, otherwise the smaller CPU build ----
cuda=0
if [ "$os" = Linux ]; then
    for smi in nvidia-smi /usr/lib/wsl/lib/nvidia-smi; do  # WSL keeps the Windows driver's tools in /usr/lib/wsl/lib
        if command -v "$smi" >/dev/null 2>&1 && "$smi" -L >/dev/null 2>&1; then cuda=1; break; fi
    done
fi
torch_failed() {
    fail "Couldn't install PyTorch for $("$python" --version 2>&1) on $os $machine." \
        "PyTorch may not have builds for this Python yet: install Python 3.12 and run" \
        "  rm -rf .venv && LMB_PYTHON=python3.12 bash install.sh"
}
if ! .venv/bin/python -c 'import torch' 2>/dev/null; then
    if [ "$os" = Darwin ] && [ "$machine" = x86_64 ]; then
        say "Installing PyTorch 2.2, the last release for Intel Macs"
        "${pip[@]}" 'torch>=2.2,<2.3' 'numpy>=1.26,<2'
    elif [ "$os" = Darwin ]; then
        say "Installing PyTorch (with Metal support for the Mac's GPU)"
        "${pip[@]}" 'torch>=2.4,<3'
    elif [ "$cuda" -eq 1 ] && [ "$gpu" != no ]; then
        say "Installing PyTorch with CUDA support (this download is large)"
        "${pip[@]}" 'torch>=2.4,<3'
    else
        say "Installing PyTorch (CPU build)"
        "${pip[@]}" --index-url https://download.pytorch.org/whl/cpu 'torch>=2.4,<3'
    fi || torch_failed
fi

# ---- Vulkan GPU backend (tinygrad + wgpu) for Linux GPUs without CUDA ----
vulkan_gpu() {
    # Distributions ship driver manifests for every GPU family, so also require a GPU we can open.
    local file node manifest=0
    for file in /usr/share/vulkan/icd.d/*.json /etc/vulkan/icd.d/*.json /opt/*/share/vulkan/icd.d/*.json; do
        [ -e "$file" ] || continue
        case "$(basename "$file")" in lvp_icd*|*lavapipe*|*swiftshader*) continue ;; esac
        manifest=1
    done
    [ "$manifest" -eq 1 ] || return 1
    for node in /dev/dri/renderD* /dev/kgsl-3d0 /dev/mali0; do
        [ -r "$node" ] && [ -w "$node" ] && return 0
    done
    return 1
}
extra=""
if [ "$os" = Darwin ]; then
    if [ "$gpu" = yes ]; then note "Macs train on their GPU through PyTorch's Metal backend; nothing extra is needed."; fi
elif [ "$gpu" = yes ] || { [ "$gpu" = auto ] && [ "$cuda" -eq 0 ] && vulkan_gpu; }; then
    extra="[gpu]"
fi

say "Installing Language Model Builder${extra:+ with the Vulkan GPU backend}"
"${pip[@]}" -e ".$extra"
if [ -n "$extra" ]; then
    .venv/bin/python tools/patch_tinygrad.py
fi

say "Looking for GPUs"
.venv/bin/python - <<'PY'
from model_builder import devices
gpus = devices.run_probe()
cpu = devices.cpu_device()
print(f"    CPU: {cpu['name']} ({cpu['detail']})")
for gpu in gpus:
    print(f"    GPU: {gpu['name']} via {gpu['label']} ({gpu['detail']})")
if not gpus:
    print("    No usable GPU found; models train on the CPU.")
PY

# ---- App window: the app's own window when possible, otherwise a browser's app mode or the browser ----
webkit_python() {  # the same check the app makes before it opens its window
    .venv/bin/python -c 'import sys; from model_builder.app import webkit_python; sys.exit(0 if webkit_python() else 1)' >/dev/null 2>&1
}
has_library() {  # not grep -q: under pipefail its early exit makes ldconfig fail with SIGPIPE
    { ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null; } | grep -F "$1" >/dev/null
}
if [ "$os" = Darwin ]; then
    if webkit_python; then
        note "The app opens in its own window, drawn by the Mac's WebKit (through pywebview)."
    else
        note "pywebview could not be loaded, so the app opens in Chrome's app mode or your browser."
    fi
else
    if ! webkit_python; then
        # The WebKitGTK libraries are often installed (as a dependency of other apps) without their
        # GObject-introspection data; that data can be unpacked here without root.
        if command -v apt-get >/dev/null && command -v dpkg-deb >/dev/null && has_library libwebkit2gtk-4.1.so.0; then
            say "Unpacking WebKitGTK introspection data for the app window"
            data_dir=$(.venv/bin/python -c 'from model_builder import paths; print(paths.data_dir())')
            work=$(mktemp -d "${TMPDIR:-/tmp}/lmb.XXXXXX")
            if (cd "$work" && apt-get download gir1.2-webkit2-4.1 gir1.2-javascriptcoregtk-4.1 gir1.2-soup-3.0 >/dev/null 2>&1); then
                mkdir -p "$data_dir/runtime-typelibs"
                for package in "$work"/*.deb; do dpkg-deb -x "$package" "$data_dir/runtime-typelibs"; done
            fi
            rm -rf "$work"
        fi
    fi
    if webkit_python; then
        note "The app opens in its own WebKitGTK window."
    elif [ "$wsl" -eq 1 ]; then
        note "The app opens in a window of its own from Microsoft Edge (or Chrome) on Windows."
        note "For a Linux window through WSLg instead: sudo apt install python3-gi gir1.2-webkit2-4.1"
    else
        note "WebKitGTK for Python was not found, so the app opens in a browser's app mode or your browser."
        note "For the app's own window install it, e.g.  Debian/Ubuntu: sudo apt install python3-gi gir1.2-webkit2-4.1"
        note "Fedora: sudo dnf install python3-gobject webkit2gtk4.1   Arch: sudo pacman -S python-gobject webkit2gtk-4.1"
    fi
fi

# ---- Menu entry (Linux) or app bundle (macOS), and the command ----
install_desktop_entry() {
    local icon target
    for icon in assets/icons/hicolor/*/apps/$app_id.*; do
        target="$data_home/icons/hicolor/${icon#assets/icons/hicolor/}"
        mkdir -p "$(dirname "$target")"
        cp "$icon" "$target"
    done
    mkdir -p "$data_home/applications"
    cat > "$data_home/applications/$app_id.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=$app_name
GenericName=Language Model Trainer
Comment=Build a small language model from scratch, step by step
Exec="$root/bin/$app_id"
Icon=$app_id
Terminal=false
Categories=Education;ComputerScience;ArtificialIntelligence;
Keywords=AI;LLM;GPT;transformer;machine learning;neural network;training;
StartupWMClass=$app_id
StartupNotify=true
DESKTOP
    refresh_menus
}
install_mac_app() {
    # A small app bundle in ~/Applications, so the app shows up in Launchpad, Spotlight and the Dock.
    # It is only a launcher (LSUIElement): the window it opens has its own Dock icon.
    local version
    version=$(.venv/bin/python -c 'from model_builder import __version__; print(__version__)')
    rm -rf "$mac_app"
    mkdir -p "$mac_app/Contents/MacOS" "$mac_app/Contents/Resources"
    cat > "$mac_app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$app_name</string>
    <key>CFBundleDisplayName</key><string>$app_name</string>
    <key>CFBundleIdentifier</key><string>io.github.skyv04.$app_id</string>
    <key>CFBundleExecutable</key><string>$app_id</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$version</string>
    <key>CFBundleVersion</key><string>$version</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>LSUIElement</key><true/>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
    cat > "$mac_app/Contents/MacOS/$app_id" <<LAUNCHER
#!/bin/bash
# Opens Language Model Builder from $root (written by its install.sh).
mkdir -p "\$HOME/Library/Logs"
exec "$root/bin/$app_id" "\$@" >>"\$HOME/Library/Logs/$app_id.log" 2>&1
LAUNCHER
    chmod +x "$mac_app/Contents/MacOS/$app_id"
    cp "assets/icons/$app_id.icns" "$mac_app/Contents/Resources/AppIcon.icns"
    touch "$mac_app"  # so Finder and the Dock notice the new icon
}
if [ "$menu" -eq 1 ]; then
    if [ "$os" = Darwin ]; then install_mac_app; else install_desktop_entry; fi
    mkdir -p "$bin_dir"
    launch="$root/bin/$app_id"
    if [ ! -e "$bin_dir/$app_id" ] || [ -L "$bin_dir/$app_id" ]; then
        ln -sfn "$root/bin/$app_id" "$bin_dir/$app_id"
        case ":$PATH:" in
            *":$bin_dir:"*) launch="$app_id" ;;
            *)
                profile="~/.profile"
                case "${SHELL:-}" in *zsh) profile="~/.zprofile" ;; esac
                note "To type just '$app_id' in new terminals, add $bin_dir to your PATH:"
                note "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> $profile" ;;
        esac
    else
        note "Kept the existing $bin_dir/$app_id, which is not this app's link."
    fi
    if [ "$os" = Darwin ]; then
        say "Done. Open “$app_name” from Launchpad, Spotlight or ~/Applications, or run: $launch"
    elif [ "$wsl" -eq 1 ] || [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        say "Done. Start it with: $launch"
    else
        say "Done. Open “$app_name” from your application menu, or run: $launch"
    fi
else
    launch="$root/bin/$app_id"
    say "Done. Start it with: $launch"
fi
note "No desktop? Run '$launch serve' and open the address it prints in a browser on this device."
note "Check this machine's setup any time with: $launch doctor"
