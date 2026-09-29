#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PromptDonate — on-device Stable Diffusion txt2img engine.

Mirrors the proven approach from ai-animator/ai_engine.py:
- torch is imported ONLY inside functions, so the GUI starts without it.
- Device auto-pick: CUDA > DirectML (Windows AMD/Intel) > MPS > CPU.
- Model (runwayml/stable-diffusion-v1-5) downloads ONCE into an app-local
  folder (sd_models/) next to the app/exe — no installs on the user's PC.
- HF mirror fallback for networks where the main CDN is blocked.

Public API:
    generate(prompt_en, negative="", seed=-1, steps=25, guidance=7.5,
             progress_cb=None) -> PIL.Image (RGB, 512x512)
    check_engine() -> [(name_fa, ok, detail)]
    pick_device()  -> (kind, label_fa)
"""
import os
import sys
import threading

MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMG_SIZE = 512
STEPS = 25
GUIDANCE = 7.5

DEFAULT_NEGATIVE = (
    "blurry, low quality, watermark, text, logo, signature, deformed, "
    "distorted, extra limbs, ugly, oversaturated"
)


class NoTorchError(RuntimeError):
    pass


def app_data_dir():
    """Writable app-local folder for the model (no installs)."""
    base = os.path.dirname(os.path.abspath(__file__))
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    d = os.path.join(base, "sd_models")
    os.makedirs(d, exist_ok=True)
    return d


def _point_hf_cache():
    d = app_data_dir()
    os.environ.setdefault("HF_HUB_CACHE", d)
    os.environ.setdefault("TRANSFORMERS_CACHE", d)
    os.environ.setdefault("HF_HOME", d)
    return d


def pick_device():
    """Return (kind, label_fa). Raises NoTorchError when torch is missing."""
    try:
        import torch
    except ImportError:
        raise NoTorchError("torch داخل برنامه پیدا نشد")
    if torch.cuda.is_available():
        return "cuda", "کارت گرافیک انویدیا (CUDA)"
    try:
        import torch_directml  # noqa: F401  (Windows: AMD/Intel GPUs)
        return "dml", "کارت گرافیک (DirectML)"
    except ImportError:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", "گرافیک مک (MPS)"
    return "cpu", "پردازنده (کندتر، ولی کار می‌کند)"


def check_engine():
    """Self-test: import every AI component. Returns [(name_fa, ok, detail)]."""
    out = []

    def _ver(mod):
        return getattr(mod, "__version__", "?")

    try:
        import torch
        out.append(("torch", True, _ver(torch)))
    except Exception as e:  # noqa: BLE001
        out.append(("torch", False, str(e)[:160]))
        return out
    try:
        import torch_directml  # noqa: F401
        try:
            from importlib.metadata import version as _pv
            _dml_ver = _pv("torch-directml")
        except Exception:
            _dml_ver = _ver(sys.modules["torch_directml"])
        out.append(("torch-directml", True, _dml_ver))
    except Exception:  # noqa: BLE001
        out.append(("torch-directml", False, "موجود نیست (فقط ویندوز/اختیاری)"))
    try:
        from diffusers import StableDiffusionPipeline  # noqa: F401
        import diffusers
        out.append(("diffusers", True, _ver(diffusers)))
    except Exception as e:  # noqa: BLE001
        out.append(("diffusers", False, str(e)[:200]))
    try:
        import transformers
        out.append(("transformers", True, _ver(transformers)))
    except Exception as e:  # noqa: BLE001
        out.append(("transformers", False, str(e)[:160]))
    try:
        kind, label = pick_device()
        out.append(("دستگاه محاسبه", True, f"{label} [{kind}]"))
    except Exception as e:  # noqa: BLE001
        out.append(("دستگاه محاسبه", False, str(e)[:160]))
    return out


class SDEngine:
    """Lazy Stable Diffusion txt2img. progress_cb(stage_fa, percent)."""

    def __init__(self, progress_cb=None, log_cb=None):
        self.progress_cb = progress_cb or (lambda s, p: None)
        self.log_cb = log_cb or (lambda t: None)
        self._pipe = None
        self._lock = threading.Lock()
        self.device_kind = None
        self.device_label = None

    # ------------------------------------------------------------ internals --
    def _progress(self, stage, pct):
        try:
            self.progress_cb(stage, pct)
        except Exception:
            pass

    def _load_pipe(self, dtype):
        from diffusers import StableDiffusionPipeline

        last_err = None
        for endpoint in (None, "https://hf-mirror.com"):
            if endpoint:
                os.environ["HF_ENDPOINT"] = endpoint
                try:
                    import huggingface_hub.constants as _c
                    _c.HF_HUB_ENDPOINT = endpoint
                except Exception:
                    pass
                self.log_cb("سرور اصلی جواب نداد؛ تلاش دوباره با آینه‌ی "
                            "hf-mirror.com ...")
            try:
                return StableDiffusionPipeline.from_pretrained(
                    MODEL_ID,
                    torch_dtype=dtype,
                    cache_dir=_point_hf_cache(),
                    safety_checker=None,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                where = "سرور اصلی" if endpoint is None else "آینه"
                self.log_cb(f"❌ دانلود از {where} ناموفق بود: {str(e)[:220]}")
        raise RuntimeError(
            "دانلود مدل ناموفق بود. اینترنت را بررسی کنید؛ اگر CDN سایت "
            "HuggingFace در شبکه‌ی شما مسدود است، با VPN یک بار مدل را "
            "دانلود کنید (فقط بار اول لازم است). "
            f"جزئیات: {str(last_err)[:300]}")

    def _pipe_get(self):
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            try:
                import torch
            except ImportError:
                raise NoTorchError(
                    "موتور AI داخل این نسخه نیست (torch پیدا نشد)")
            kind, label = pick_device()
            self.device_kind, self.device_label = kind, label
            self.log_cb(f"دستگاه محاسبه: {label}")

            use_fp16 = kind in ("cuda", "mps")
            dtype = torch.float16 if use_fp16 else torch.float32
            self.log_cb(f"دانلود/بارگذاری مدل {MODEL_ID} ... (فقط بار اول)")
            pipe = self._load_pipe(dtype)
            if kind == "cuda":
                pipe = pipe.to("cuda")
            elif kind == "mps":
                pipe = pipe.to("mps")
            elif kind == "dml":
                import torch_directml
                pipe = pipe.to(torch_directml.device())
            try:
                pipe.enable_attention_slicing()
                pipe.enable_vae_slicing()  # حافظه‌ی کمتر روی گرافیک‌های ضعیف
            except Exception:
                pass
            self._pipe = pipe
            self.log_cb("مدل آماده شد ✅")
            return pipe

    # ------------------------------------------------------------------ API --
    def generate(self, prompt_en, negative="", seed=-1,
                 steps=STEPS, guidance=GUIDANCE):
        """Run txt2img. Returns PIL RGB 512x512."""
        try:
            import torch
        except ImportError:
            raise NoTorchError("موتور AI داخل این نسخه نیست (torch پیدا نشد)")
        pipe = self._pipe_get()
        if seed is None or seed < 0:
            seed = torch.seed() % (2 ** 31)
        gen = torch.Generator().manual_seed(int(seed))
        neg = negative.strip() or DEFAULT_NEGATIVE

        def _cb(*args):
            # old-style: (step, timestep, latents) / new-style:
            # (pipe, step, timestep, callback_kwargs)
            try:
                if len(args) == 4:
                    step = int(args[1])
                    self._progress("تولید تصویر با AI", int(100 * step / steps))
                    return args[3]
                step = int(args[0])
                self._progress("تولید تصویر با AI", int(100 * step / steps))
            except Exception:
                pass

        self._progress("تولید تصویر با AI", 2)
        out = pipe(
            prompt=prompt_en,
            negative_prompt=neg,
            height=IMG_SIZE,
            width=IMG_SIZE,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=gen,
            callback=_cb,
            callback_steps=1,
        )
        self._progress("تولید تصویر با AI", 100)
        img = out.images[0].convert("RGB")
        img.info["seed"] = int(seed)
        return img
