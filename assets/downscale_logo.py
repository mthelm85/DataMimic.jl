#!/usr/bin/env python3
"""Regenerate docs/src/assets/logo.png from assets/logo-source.png.

Pure stdlib (zlib only), so it runs anywhere without an imaging library.

    python assets/downscale_logo.py [width]

The one thing that matters here: alpha is premultiplied before averaging.
The logo's background is transparent *black*, (0, 0, 0, 0), so averaging
straight RGB channels would drag every edge pixel toward black and leave a
dark fringe around the artwork. Weighting each sample by its own alpha and
dividing by the summed alpha keeps edge colour true.
"""

import io
import os
import struct
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "logo-source.png")
DST = os.path.join(ROOT, "docs", "src", "assets", "logo.png")


def decode_rgba(path):
    """Minimal 8-bit RGBA PNG decoder: concatenate IDAT, inflate, unfilter."""
    d = io.open(path, "rb").read()
    if d[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    w, h = struct.unpack(">II", d[16:24])
    if (d[24], d[25]) != (8, 6):
        raise ValueError("expected 8-bit RGBA (bit depth 8, colour type 6)")

    i, idat = 8, b""
    while i < len(d):
        ln = struct.unpack(">I", d[i:i + 4])[0]
        if d[i + 4:i + 8] == b"IDAT":
            idat += d[i + 8:i + 8 + ln]
        i += 12 + ln
    raw = zlib.decompress(idat)

    bpp, stride = 4, w * 4
    prev, rows, pos = bytearray(stride), [], 0
    for _ in range(h):
        f = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + stride])
        pos += stride
        if f:
            for x in range(stride):
                a = line[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                if f == 1:
                    line[x] = (line[x] + a) & 255
                elif f == 2:
                    line[x] = (line[x] + b) & 255
                elif f == 3:
                    line[x] = (line[x] + (a + b) // 2) & 255
                elif f == 4:
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                    line[x] = (line[x] + pr) & 255
                else:
                    raise ValueError(f"unknown filter type {f}")
        prev = line
        rows.append(bytes(line))
    return w, h, rows


def box_downscale(w, h, rows, nw):
    """Area-average downscale, premultiplying alpha (see module docstring)."""
    nh = round(h * nw / w)
    out = []
    for ny in range(nh):
        y0, y1 = ny * h // nh, max(ny * h // nh + 1, (ny + 1) * h // nh)
        line = bytearray(nw * 4)
        for nx in range(nw):
            x0, x1 = nx * w // nw, max(nx * w // nw + 1, (nx + 1) * w // nw)
            sr = sg = sb = sa = n = 0
            for y in range(y0, y1):
                row = rows[y]
                for x in range(x0, x1):
                    o = x * 4
                    a = row[o + 3]
                    sr += row[o] * a
                    sg += row[o + 1] * a
                    sb += row[o + 2] * a
                    sa += a
                    n += 1
            o = nx * 4
            if sa:
                line[o] = min(255, round(sr / sa))
                line[o + 1] = min(255, round(sg / sa))
                line[o + 2] = min(255, round(sb / sa))
                line[o + 3] = round(sa / n)
        out.append(bytes(line))
    return nw, nh, out


def encode_rgba(path, w, h, rows):
    """Write 8-bit RGBA PNG, Sub-filtering each row (cheap, good on flat art)."""
    raw = bytearray()
    for r in rows:
        raw.append(1)
        sub = bytearray(len(r))
        for x in range(len(r)):
            sub[x] = (r[x] - (r[x - 4] if x >= 4 else 0)) & 255
        raw += sub

    def chunk(typ, data):
        return (struct.pack(">I", len(data)) + typ + data +
                struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    io.open(path, "wb").write(png)


if __name__ == "__main__":
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    w, h, rows = decode_rgba(SRC)
    print(f"source {w}x{h}, {os.path.getsize(SRC) / 1024:.0f} KB")
    nw, nh, nr = box_downscale(w, h, rows, width)
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    encode_rgba(DST, nw, nh, nr)
    print(f"wrote  {nw}x{nh}, {os.path.getsize(DST) / 1024:.0f} KB -> {DST}")
