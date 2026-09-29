#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless smoke test for PromptDonate.

- fx.animate on a SYNTHETIC stand-in image (no diffusion, no GPU needed).
- 3 canvas sizes; asserts RGBA, real transparency at corners, text drawn.
- Pillow transparent GIF export round-trip.
- Offscreen GUI import + MainWindow construction (torch stubbed so the
  check-engine thread reports gracefully).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image


def make_standin(size=512):
    """Synthetic stand-in for an SD image: gold radial orb on dark bg."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size / 2)
    glow = np.clip(1 - d, 0, 1) ** 1.8
    arr = np.zeros((size, size, 3), np.uint8)
    arr[..., 0] = (20 + glow * 235).astype(np.uint8)
    arr[..., 1] = (12 + glow * 165).astype(np.uint8)
    arr[..., 2] = (8 + glow * 60).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def main():
    import fx

    img = make_standin()
    sizes = [(640, 360), (480, 480), (320, 180)]
    for w, h in sizes:
        frames = fx.animate(img, "آرش", "۱۰۰٬۰۰۰ تومان", w=w, h=h,
                            fps=15, duration=2.0, seed=7)
        assert len(frames) == 30, f"{w}x{h}: {len(frames)} frames"
        f0 = frames[0]
        assert f0.mode == "RGBA" and f0.size == (w, h)
        a = np.array(f0.getchannel("A"), dtype=np.float32)
        # corners must stay transparent (alert overlay); check 16x16 patches
        # by mean alpha since particles may legitimately fly near edges
        for (ys, xs) in [(slice(0, 16), slice(0, 16)),
                         (slice(0, 16), slice(w - 16, w)),
                         (slice(h - 16, h), slice(0, 16)),
                         (slice(h - 16, h), slice(w - 16, w))]:
            assert a[ys, xs].mean() < 48, f"{w}x{h}: corner patch not transparent"
        # something drawn in the middle
        mid = frames[len(frames) // 2]
        ma = np.array(mid.getchannel("A"))
        assert (ma > 128).sum() > w * h * 0.05, f"{w}x{h}: frame looks empty"
        print(f"fx OK: {w}x{h}, {len(frames)} frames, transparency preserved")

    # text presence: compare frame with/without text
    a1 = fx.animate(img, "آرش", "۱۰۰٬۰۰۰ تومان", w=320, h=180,
                    fps=15, duration=2.0, seed=7)
    a2 = fx.animate(img, "", "", w=320, h=180, fps=15, duration=2.0, seed=7)
    d1 = np.array(a1[-1].convert("RGB"), dtype=np.int32)
    d2 = np.array(a2[-1].convert("RGB"), dtype=np.int32)
    diff = np.abs(d1 - d2).sum()
    assert diff > 20000, "text does not seem to be drawn"
    print(f"fx OK: donation text visibly drawn (diff={diff})")

    # GIF export round-trip
    out = "/tmp/prompt_donate_smoke.gif"
    size = fx.export_gif(a1, out, fps=15)
    assert os.path.getsize(out) == size and size > 10000
    chk = Image.open(out)
    assert getattr(chk, "n_frames", 1) > 1
    print(f"export OK: {out} ({size} bytes), {chk.n_frames} frames")

    # sd_engine imports WITHOUT torch (lazy) and raises NoTorchError cleanly
    import sd_engine
    assert sd_engine.MODEL_ID == "runwayml/stable-diffusion-v1-5"
    try:
        sd_engine.pick_device()
        print("sd_engine: torch present (unexpected here)")
    except sd_engine.NoTorchError:
        print("sd_engine OK: lazy import, NoTorchError without torch")

    # GUI offscreen
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import app as gui
    win = gui.MainWindow()
    assert win.preset_combo.count() == 8, "expected 8 presets"
    assert win.size_combo.count() == 2
    assert len(gui.PRESETS) == 8
    print(f"GUI OK (offscreen): {win.preset_combo.count()} presets, "
          f"window '{win.windowTitle()}'")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
