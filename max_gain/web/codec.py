"""Compact encodings shared by the page build: Google polyline for ints/coords, Douglas-Peucker thinning."""
import numpy as np


def encode_ints(values):
    """Polyline-encode a flat sequence of ints as deltas (zigzag, 5-bit chunks)."""
    out, prev = [], 0
    for v in values:
        d, prev = int(v) - prev, int(v)
        d = ~(d << 1) if d < 0 else d << 1
        while d >= 0x20:
            out.append(chr((0x20 | (d & 0x1F)) + 63))
            d >>= 5
        out.append(chr(d + 63))
    return "".join(out)


def encode_pairs(pairs, scale):
    """Encode [(a, b), ...] scaled to ints as interleaved delta streams, as in Google polylines."""
    a = np.round(np.asarray(pairs, float) * scale).astype(np.int64)
    s, pa, pb = [], 0, 0
    for x, y in a:
        s.append(_enc(int(x) - pa))
        s.append(_enc(int(y) - pb))
        pa, pb = int(x), int(y)
    return "".join(s)


def _enc(d):
    d = ~(d << 1) if d < 0 else d << 1
    out = []
    while d >= 0x20:
        out.append(chr((0x20 | (d & 0x1F)) + 63))
        d >>= 5
    out.append(chr(d + 63))
    return "".join(out)


def simplify(xyz, tol):
    """Iterative Douglas-Peucker on 3D points; returns the kept indices."""
    keep = np.zeros(len(xyz), bool)
    keep[[0, -1]] = True
    stack = [(0, len(xyz) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = xyz[i], xyz[j]
        ab = b - a
        t = np.clip(((xyz[i + 1:j] - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        d = np.linalg.norm(xyz[i + 1:j] - (a + t[:, None] * ab), axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return np.flatnonzero(keep)
