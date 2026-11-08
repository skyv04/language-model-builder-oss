"""Build the macOS app icon (assets/icons/language-model-builder.icns) from the PNG icons.

An .icns file is a short header followed by entries that may simply hold PNG files, one per
size, so no image library (and no Mac) is needed.  macOS scales the 64 px entry down for the
smallest sizes.

Usage: python tools/build_icns.py [output.icns]
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ICONS = ROOT / "assets" / "icons"
OUTPUT = ICONS / "language-model-builder.icns"
# Entry type and the PNG size it holds (types ending in 1 to 4 are the @2x Retina variants).
ENTRIES = (("icp6", 64), ("ic07", 128), ("ic08", 256), ("ic09", 512),
           ("ic12", 64), ("ic13", 256), ("ic14", 512))


def png(size: int) -> bytes:
    data = (ICONS / "hicolor" / f"{size}x{size}" / "apps" / "language-model-builder.png").read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError(f"the {size} px icon is not a PNG file")
    width, height = struct.unpack(">II", data[16:24])
    if (width, height) != (size, size):
        raise ValueError(f"the {size} px icon is {width}x{height}")
    return data


def build() -> bytes:
    body = b"".join(kind.encode("ascii") + struct.pack(">I", 8 + len(data)) + data
                    for kind, data in ((kind, png(size)) for kind, size in ENTRIES))
    return b"icns" + struct.pack(">I", 8 + len(body)) + body


def main(argv: list[str]) -> int:
    output = Path(argv[1]) if len(argv) > 1 else OUTPUT
    output.write_bytes(build())
    print(f"Wrote {output} ({output.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
