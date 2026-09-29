#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PromptDonate fx — IMAGE-PROCESSING animation: any input image becomes a
transparent donation-alert GIF.

Pipeline per frame (all Pillow + numpy, no ffmpeg):
  input image -> rounded-rect mask + gold border + outer glow
  motion      -> punch-zoom in, slow drift, screen-shake burst, flash frame
  effects     -> RGB-split glitch, coin/bill particles, light rays,
                 golden vignette, sparkles
  text        -> donor name + amount (B Nazanin), jelly elastic pop-in,
                 amount counts up

The coin/bill/jelly renderers follow the same procedural style as
donate-studio/donate_pro.py, re-implemented here so this repo is
self-contained.
"""
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from PIL import features as _pil_features
    _HAS_RAQM = _pil_features.check("raqm")
except Exception:
    _HAS_RAQM = False
try:
    import arabic_reshaper  # noqa: F401
    from bidi.algorithm import get_display  # noqa: F401
    _HAS_RESHAPER = True
except Exception:
    _HAS_RESHAPER = False


# ------------------------------------------------------------------ helpers --
def clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)


def ease_out(t):
    t = clamp01(t)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = clamp01(t)
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2


def ease_out_back(t):
    t = clamp01(t)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def _jelly(t):
    """Elastic pop-in scale: 0 before trigger, overshoots, settles at 1."""
    if t <= 0:
        return 0.0
    tt = clamp01(t / 0.30)
    return ease_out(tt) * (1 + 0.30 * math.exp(-3.2 * t)
                           * math.cos(2 * math.pi * 3.0 * t))


class _Cancelled(Exception):
    pass


_cb = None


def _frame_iter(F):
    for f in range(F):
        if _cb is not None:
            try:
                keep = _cb(f, F)
            except Exception:
                keep = True
            if keep is False:
                raise _Cancelled()
        yield f


def shake_xy(f, F, amp):
    """Decaying random-ish screen shake (deterministic per frame)."""
    if amp <= 0:
        return 0, 0
    r = random.Random(9000 + f)
    decay = 1 - f / max(1, F)
    return int(r.uniform(-amp, amp) * decay), int(r.uniform(-amp, amp) * decay)


# ------------------------------------------------------------------- glow ----
def radial_glow(size, color, max_alpha):
    """Square RGBA tile with radial falloff glow of `color`."""
    s = max(2, int(size))
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    d = np.sqrt((xx - s / 2) ** 2 + (yy - s / 2) ** 2) / (s / 2)
    a = np.clip(1 - d, 0, 1) ** 2 * max_alpha
    arr = np.zeros((s, s, 4), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = color
    arr[..., 3] = a.astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _rays(base, cx, cy, R, angle, n, color, alpha):
    if alpha <= 1:
        return
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    hw = math.pi * 0.05
    for i in range(n):
        a0 = angle + i * 2 * math.pi / n
        d.polygon([(cx, cy),
                   (cx + R * math.cos(a0 - hw), cy + R * math.sin(a0 - hw)),
                   (cx + R * math.cos(a0 + hw), cy + R * math.sin(a0 + hw))],
                  fill=color + (int(alpha),))
    ov = ov.filter(ImageFilter.GaussianBlur(8))
    base.alpha_composite(ov)


# --------------------------------------------------------------- particles --
def _coin_sprite(r):
    """Pre-rendered golden coin sprite (RGBA), diameter 2r."""
    s = max(4, int(r * 2))
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = s / 2
    d.ellipse([1, 1, s - 1, s - 1], fill=(176, 118, 20, 255))      # rim
    d.ellipse([3, 3, s - 3, s - 3], fill=(255, 205, 90, 255))      # face
    d.ellipse([s * 0.28, s * 0.28, s * 0.72, s * 0.72],
              fill=(255, 226, 140, 255))                           # inner
    d.ellipse([s * 0.16, s * 0.12, s * 0.38, s * 0.30],
              fill=(255, 255, 255, 200))                           # shine
    return im


def _bill_sprite(w):
    """Pre-rendered dollar-bill sprite (RGBA)."""
    h = int(w * 0.52)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w, h], radius=h // 6, fill=(34, 120, 70, 255))
    d.rounded_rectangle([3, 3, w - 3, h - 3], radius=h // 8,
                        outline=(220, 240, 210, 255), width=2)
    d.ellipse([w * 0.36, h * 0.18, w * 0.64, h * 0.82],
              fill=(235, 245, 230, 255))
    try:
        fnt = ImageFont.truetype(_font_path(), max(8, h // 3))
    except Exception:
        fnt = ImageFont.load_default()
    d.text((w / 2, h / 2), "$", font=fnt, fill=(30, 110, 60, 255), anchor="mm")
    return im


def _sparkle_sprite(s, color=(255, 230, 160, 255)):
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = s / 2
    d.polygon([(c, 0), (c + s * 0.12, c - s * 0.12), (s, c),
               (c + s * 0.12, c + s * 0.12), (c, s),
               (c - s * 0.12, c + s * 0.12), (0, c),
               (c - s * 0.12, c - s * 0.12)], fill=color)
    return im


# ------------------------------------------------------------ Persian text --
def _font_path(explicit=None):
    if explicit and os.path.isfile(explicit):
        return explicit
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        p = os.path.join(meipass, "fonts", "BNazanin.ttf")
        if os.path.isfile(p):
            return p
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fonts", "BNazanin.ttf")
    if os.path.isfile(here):
        return here
    return None


def _truetype(px):
    try:
        fp = _font_path()
        if fp:
            return ImageFont.truetype(fp, max(8, int(px)))
    except Exception:
        pass
    return ImageFont.load_default()


_SEP_FIX = str.maketrans({"٬": "،"})  # B Nazanin lacks U+066C


def _clean(text):
    return text.translate(_SEP_FIX) if text else text


def _draw_rtl(d, xy, text, font, fill, sw=0, sfill=None):
    text = _clean(text)
    kw = {"stroke_width": sw, "stroke_fill": sfill} if sw else {}
    if _HAS_RAQM:
        d.text(xy, text, font=font, fill=fill, anchor="mm",
               direction="rtl", **kw)
    elif _HAS_RESHAPER:
        d.text(xy, get_display(arabic_reshaper.reshape(text)), font=font,
               fill=fill, anchor="mm", **kw)
    else:
        d.text(xy, text, font=font, fill=fill, anchor="mm", **kw)


def _text_size(text, font, sw=0):
    text = _clean(text)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8), (0, 0, 0, 0)))
    kw = {"stroke_width": sw} if sw else {}
    if _HAS_RAQM:
        bb = probe.textbbox((0, 0), text, font=font, direction="rtl", **kw)
    elif _HAS_RESHAPER:
        bb = probe.textbbox((0, 0), get_display(arabic_reshaper.reshape(text)),
                            font=font, **kw)
    else:
        bb = probe.textbbox((0, 0), text, font=font, **kw)
    return bb[2] - bb[0], bb[3] - bb[1]


def _jelly_text(base, cx, cy, text, px, scale, color, stroke, glow=True):
    if scale <= 0.01 or not text:
        return
    font = _truetype(px)
    tw, th = _text_size(text, font, sw=2)
    pad = int(px * 0.6) + 12
    tile = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    td = ImageDraw.Draw(tile)
    _draw_rtl(td, (tile.width / 2, tile.height / 2), text, font,
              color + (255,), sw=2, sfill=stroke + (255,))
    if glow:
        g = radial_glow(max(tile.width, tile.height), (255, 190, 80), 90)
        base.alpha_composite(g, (int(cx - g.width / 2), int(cy - g.height / 2)))
    nw, nh = max(1, int(tile.width * scale)), max(1, int(tile.height * scale))
    tile = tile.resize((nw, nh), Image.LANCZOS)
    base.alpha_composite(tile, (int(cx - nw / 2), int(cy - nh / 2)))


_FA_D = "۰۱۲۳۴۵۶۷۸۹"
_DIGIT_MAP = {}
for _c in "0123456789":
    _DIGIT_MAP[_c] = _c
for _i, _c in enumerate("۰۱۲۳۴۵۶۷۸۹"):
    _DIGIT_MAP[_c] = str(_i)
for _i, _c in enumerate("٠١٢٣٤٥٦٧٨٩"):
    _DIGIT_MAP[_c] = str(_i)


def _parse_amount(text):
    digit_chars = set(_DIGIT_MAP) | set("٬, ")
    n, i = len(text), 0
    while i < n and text[i] not in digit_chars:
        i += 1
    j = i
    while j < n and text[j] in digit_chars:
        j += 1
    val = "".join(_DIGIT_MAP.get(c, "") for c in text[i:j])
    val = "".join(c for c in val if c.isdigit())
    if not val:
        return None, text.strip()
    return int(val), text[j:].strip()


def _fa_num(n):
    s = str(int(round(n)))
    grp = []
    while len(s) > 3:
        grp.append(s[-3:])
        s = s[:-3]
    grp.append(s)
    s = "،".join(reversed(grp))
    return "".join(_FA_D[int(c)] if c.isdigit() else c for c in s)


# ------------------------------------------------------------------ mosaic --
def _rounded_tile(img, size, radius, border, border_color):
    """img -> RGBA square tile with rounded mask + border."""
    tile = img.convert("RGBA").resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size, size],
                                           radius=radius, fill=255)
    tile.putalpha(Image.fromarray(
        (np.array(mask, dtype=np.float32) / 255 *
         np.array(tile.getchannel("A"), dtype=np.float32)).astype(np.uint8)))
    if border > 0:
        ov = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ImageDraw.Draw(ov).rounded_rectangle(
            [border // 2, border // 2, size - border // 2, size - border // 2],
            radius=radius, outline=border_color + (255,), width=border)
        tile = Image.alpha_composite(tile, ov)
    return tile


def _glitch_shift(layer, dx):
    """RGB-split: shift R +dx, B -dx on an RGBA layer."""
    r, g, b, a = layer.split()
    r = r.transform(layer.size, Image.AFFINE, (1, 0, -dx, 0, 1, 0))
    b = b.transform(layer.size, Image.AFFINE, (1, 0, dx, 0, 1, 0))
    return Image.merge("RGBA", (r, g, b, a))


def _golden_vignette(w, h, strength=46):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xx - w / 2) / (w / 2)
    dy = (yy - h / 2) / (h / 2)
    d = np.sqrt(dx * dx + dy * dy) / math.sqrt(2)
    edge = np.clip((d - 0.55) / 0.45, 0, 1) ** 1.6
    # fade back to 0 at the extreme corners so the overlay keeps
    # transparent corners (no visible rectangle edge in OBS)
    edge = edge * np.clip((0.985 - d) / 0.085, 0, 1)
    arr = np.zeros((h, w, 4), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = 255, 175, 70
    arr[..., 3] = (edge * strength).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


# ---------------------------------------------------------------- animate --
def animate(image, name="", amount="", w=1280, h=720, fps=30, duration=4.0,
            seed=1234, on_frame=None):
    """Turn `image` (PIL) into donation-alert RGBA frames (transparent bg).

    Returns list[PIL.Image RGBA]. Cancel via on_frame(f, F) -> False.
    """
    global _cb
    _cb = on_frame
    try:
        return _animate_inner(image, name, amount, w, h, fps, duration, seed)
    finally:
        _cb = None


def _animate_inner(image, name, amount, w, h, fps, duration, seed):
    F = max(8, int(round(fps * duration)))
    rng = random.Random(seed)
    sc = h / 720.0  # global scale factor

    cx = w / 2
    tile_size = int(min(w, h) * 0.60)
    tile_cy = h * 0.40
    tile = _rounded_tile(image, tile_size, int(tile_size * 0.10),
                         max(3, int(6 * sc)), (255, 205, 110))
    coin_spr = _coin_sprite(max(6, int(26 * sc)))
    bill_spr = _bill_sprite(max(20, int(90 * sc)))
    spark_spr = _sparkle_sprite(max(8, int(18 * sc)))
    glow_big = radial_glow(int(tile_size * 1.9), (255, 185, 80), 110)
    vignette = _golden_vignette(w, h)

    # ---- particle definitions (analytic, deterministic) ----
    coins = []
    for _ in range(46):
        coins.append({
            "spawn": rng.uniform(0.04, 0.50),
            "x0": cx + rng.uniform(-w * 0.30, w * 0.30),
            "vx": rng.uniform(-60, 60) * sc,
            "vy": -rng.uniform(420, 780) * sc,
            "r": rng.uniform(0.55, 1.15),
            "spin": rng.uniform(0, math.pi * 2),
            "spinv": rng.uniform(4, 11),
            "life": rng.uniform(1.1, 1.7),
        })
    bills = []
    for _ in range(22):
        bills.append({
            "spawn": rng.uniform(0.0, 0.55),
            "x0": rng.uniform(0, w),
            "vy": rng.uniform(150, 300) * sc,
            "sway": rng.uniform(20, 70) * sc,
            "ph": rng.uniform(0, math.pi * 2),
            "rot": rng.uniform(0, 360),
            "rotv": rng.uniform(-90, 90),
            "s": rng.uniform(0.6, 1.2),
            "life": 3.2,
        })
    sparks = []
    for _ in range(40):
        sparks.append({
            "x": rng.uniform(0, w), "y": rng.uniform(0, h),
            "ph": rng.uniform(0, math.pi * 2),
            "spd": rng.uniform(1.5, 4.0),
            "s": rng.uniform(0.5, 1.3),
        })

    val, unit = _parse_amount(amount) if amount else (None, "")

    name_px = int(54 * sc)
    amt_px = int(96 * sc)
    name_y = h * 0.78
    amt_y = h * 0.885

    frames = []
    for f in _frame_iter(F):
        t = f / max(1, F - 1)
        base = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        # ---- light rays ----
        ray_a = 70 * clamp01((t - 0.15) / 0.25)
        _rays(base, cx, tile_cy, max(w, h) * 0.75, t * 1.4, 10,
              (255, 200, 110), ray_a)

        # ---- glow behind image ----
        gp = 90 + 45 * math.sin(2 * math.pi * 1.2 * t)
        ga = glow_big.copy()
        ga.putalpha(ga.getchannel("A").point(lambda v: int(v * gp / 135)))
        base.alpha_composite(ga, (int(cx - ga.width / 2),
                                  int(tile_cy - ga.height / 2)))

        # ---- image tile: punch-zoom + drift ----
        punch = 0.55 + 0.53 * ease_out_back(clamp01(t / 0.16))
        settle = 1.0 + 0.018 * math.sin(2 * math.pi * 2 * t)
        s = punch if t < 0.30 else settle
        dx = 10 * sc * math.sin(2 * math.pi * 0.5 * t)
        dy = 7 * sc * math.sin(2 * math.pi * 0.35 * t + 1)
        tw = max(1, int(tile.width * s))
        tile_s = tile.resize((tw, tw), Image.LANCZOS)
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        layer.alpha_composite(tile_s, (int(cx - tw / 2 + dx),
                                       int(tile_cy - tw / 2 + dy)))
        # RGB-split glitch: 3 frames at t≈0.25
        gf0 = int(0.25 * F)
        if gf0 <= f < gf0 + 3:
            layer = _glitch_shift(layer, int(7 * sc))
        base.alpha_composite(layer)

        # ---- coins ----
        for c in coins:
            dt = t * duration - c["spawn"] * duration
            if dt < 0 or dt > c["life"]:
                continue
            x = c["x0"] + c["vx"] * dt
            y = (h + 30 * sc) + c["vy"] * dt + 0.5 * 900 * sc * dt * dt
            if y > h + 60 * sc:
                continue
            sx = abs(math.cos(c["spin"] + c["spinv"] * dt))
            cw = max(2, int(coin_spr.width * c["r"] * max(0.25, sx)))
            ch = max(2, int(coin_spr.height * c["r"]))
            spr = coin_spr.resize((cw, ch), Image.LANCZOS)
            base.alpha_composite(spr, (int(x - cw / 2), int(y - ch / 2)))

        # ---- bills ----
        for b in bills:
            dt = t * duration - b["spawn"] * duration
            if dt < 0 or dt > b["life"]:
                continue
            y = -60 * sc + b["vy"] * dt
            if y > h + 60 * sc:
                continue
            x = b["x0"] + b["sway"] * math.sin(b["ph"] + dt * 2.2)
            bw = max(4, int(bill_spr.width * b["s"]))
            bh = max(4, int(bill_spr.height * b["s"]))
            spr = bill_spr.resize((bw, bh), Image.LANCZOS).rotate(
                b["rot"] + b["rotv"] * dt, expand=True, resample=Image.BICUBIC)
            base.alpha_composite(spr, (int(x - spr.width / 2),
                                       int(y - spr.height / 2)))

        # ---- sparkles ----
        for sp in sparks:
            tw2 = 0.5 + 0.5 * math.sin(sp["ph"] + t * duration * sp["spd"])
            if tw2 < 0.55:
                continue
            ss = max(2, int(spark_spr.width * sp["s"] * tw2))
            spr = spark_spr.resize((ss, ss), Image.LANCZOS)
            base.alpha_composite(spr, (int(sp["x"] - ss / 2),
                                       int(sp["y"] - ss / 2)))

        # ---- golden vignette ----
        base.alpha_composite(vignette)

        # ---- text: name + amount (jelly pop + count-up) ----
        _jelly_text(base, cx, name_y, name, name_px,
                    _jelly((t - 0.55) / 0.12),
                    (255, 255, 255), (40, 25, 8))
        if val is not None:
            shown = _fa_num(val * ease_out((t - 0.62) / 0.35))
            amt_text = f"{shown} {unit}".strip()
        else:
            amt_text = amount
        _jelly_text(base, cx, amt_y, amt_text, amt_px,
                    _jelly((t - 0.62) / 0.12),
                    (255, 205, 90), (60, 32, 6))

        # ---- flash + screen shake at text pop ----
        fl = (0.0 if t < 0.55
              else 150 * max(0.0, 1 - (t - 0.55) / 0.12))
        if fl > 2:
            flash = Image.new("RGBA", (w, h), (255, 225, 160, int(fl)))
            base.alpha_composite(flash)
        shake_t = (t - 0.55) / 0.18
        sx, sy = (0, 0) if not (0 <= shake_t <= 1) else shake_xy(
            f, F, int(11 * sc * (1 - shake_t)))
        fin = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        fin.alpha_composite(base, (sx, sy))
        frames.append(fin)

    return frames


# -------------------------------------------------------------- export -----
def export_gif(frames, path, fps=30):
    """Export RGBA frames to a transparent GIF (Pillow only, no ffmpeg).

    Shared 255-color palette sampled across the animation; palette index
    255 reserved for transparency. Returns output file size in bytes.
    """
    if not frames:
        raise ValueError("no frames to export")
    import math as _math
    w, h = frames[0].size
    step = max(1, len(frames) // 8)
    sample = frames[::step][:8] or frames[:1]
    cols = _math.ceil(_math.sqrt(len(sample)))
    rows = _math.ceil(len(sample) / cols)
    mosaic = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
    for i, fr in enumerate(sample):
        bg = Image.new("RGBA", fr.size, (0, 0, 0, 255))
        mosaic.paste(Image.alpha_composite(bg, fr).convert("RGB"),
                     ((i % cols) * w, (i // cols) * h))
    palette = mosaic.quantize(colors=255, method=Image.FASTOCTREE)
    pal_frames = []
    for fr in frames:
        transp = fr.getchannel("A").point(lambda v: 255 if v < 128 else 0)
        bg = Image.new("RGBA", fr.size, (0, 0, 0, 255))
        comp = Image.alpha_composite(bg, fr).convert("RGB")
        q = comp.quantize(palette=palette, dither=Image.FLOYDSTEINBERG)
        q.paste(255, transp)
        pal_frames.append(q)
    duration_ms = int(round(1000.0 / max(1, fps)))
    pal_frames[0].save(path, save_all=True, append_images=pal_frames[1:],
                       duration=duration_ms, loop=0, transparency=255,
                       disposal=1)
    return os.path.getsize(path)


__all__ = ["animate", "export_gif"]
