# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PromptDonate (Windows).

Bundles the full on-device AI stack (torch, diffusers, torch-directml, ...)
so the app runs with zero installs on the user's PC. The Stable Diffusion
weights are NOT bundled - they download once into sd_models/ on first use.
Expected exe bundle size: ~1-1.5GB (torch is heavy; this is normal).
"""
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = [("fonts", "fonts")]
binaries = []
hiddenimports = []

for pkg in ("torch", "torchvision", "diffusers", "transformers", "tokenizers",
            "huggingface_hub", "safetensors", "accelerate",
            "PIL", "numpy", "requests", "urllib3", "certifi",
            "charset_normalizer", "idna", "tqdm", "packaging",
            "filelock", "fsspec", "regex"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Package *metadata* (dist-info): huggingface_hub/transformers call
# importlib.metadata at runtime and crash without it.
for pkg in ("requests", "urllib3", "certifi", "transformers", "tokenizers",
            "diffusers", "huggingface_hub", "safetensors", "accelerate",
            "filelock", "fsspec", "tqdm", "packaging", "regex", "numpy",
            "Pillow", "torch", "torchvision", "torch_directml"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Windows-only: AMD/Intel GPU backend
try:
    d, b, h = collect_all("torch_directml")
    datas += d
    binaries += b
    hiddenimports += h
except Exception:
    pass

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "pandas", "notebook", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PromptDonate",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="PromptDonate",
)
