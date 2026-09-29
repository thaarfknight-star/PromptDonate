#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PromptDonate — پرامپت بده، گیف دونیت تحویل بگیر.

Persian RTL PySide6 GUI, dark streamer theme with gold accents.
Pipeline: prompt -> on-device Stable Diffusion txt2img -> image-processing
animation (fx.animate) -> transparent GIF export (Pillow only, no ffmpeg).

torch/diffusers are imported lazily inside sd_engine only, so the GUI
always starts even on machines without the AI stack.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QSpinBox,
    QProgressBar, QFileDialog, QMessageBox, QGroupBox,
)

import fx
import sd_engine

OUT_DIR = os.path.expanduser("~/workspace/your_files/prompt-donate")

PRESETS = [
    ("انفجار سکه طلایی",
     "golden coins explosion, treasure burst, dramatic cinematic lighting, "
     "dark background, flying gold coins with motion blur, high detail",
     ""),
    ("باران اسکناس",
     "dollar bills raining from the sky, money rain, green and gold tones, "
     "dark moody background, cinematic lighting, high detail",
     ""),
    ("نئون سایبرپانک",
     "neon cyberpunk city at night, purple and cyan glow, futuristic "
     "skyscrapers, rain reflections, dramatic, ultra detailed",
     ""),
    ("حماسه آتشین",
     "epic phoenix made of fire rising from flames, orange fire feathers, "
     "burning embers, dark background, cinematic, dramatic",
     ""),
    ("کهکشان بنفش",
     "purple galaxy nebula, bright stars, cosmic dust clouds, deep space, "
     "vibrant colors, ultra detailed",
     ""),
    ("شمش طلا",
     "gold bars stacked in a treasure vault, golden light rays, luxurious, "
     "sparkling gold, cinematic lighting, high detail",
     ""),
    ("اژدهای طلایی",
     "majestic golden dragon, fantasy art, glowing eyes, scales shining, "
     "dark epic background, cinematic, ultra detailed",
     ""),
    ("سفارشی", "", ""),  # uses the prompt box as-is
]

SIZES = [("۱۲۸۰×۷۲۰ (تمام‌صفحه)", 1280, 720),
         ("۷۲۰×۷۲۰ (مربع)", 720, 720)]

STYLESHEET = """
QWidget { background: #16181f; color: #e8eaf0; font-size: 14px; }
QGroupBox { border: 1px solid #4a3f28; border-radius: 8px; margin-top: 14px;
            font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top right;
                  right: 12px; padding: 0 6px; color: #e8c26a; }
QPushButton { background: #2d2a20; border: 1px solid #6b5a33; border-radius: 8px;
              padding: 8px 14px; }
QPushButton:hover { background: #3d382a; }
QPushButton:disabled { color: #6b7280; background: #20222b; }
QPushButton#primary { background: #7a5c1e; border-color: #a8842f;
                      font-weight: bold; }
QPushButton#primary:hover { background: #8f6d24; }
QLineEdit, QTextEdit, QComboBox, QSpinBox { background: #20222b;
    border: 1px solid #4a3f28; border-radius: 6px; padding: 6px; }
QProgressBar { border: 1px solid #4a3f28; border-radius: 6px;
               text-align: center; background: #20222b; }
QProgressBar::chunk { background: #7a5c1e; border-radius: 4px; }
QLabel#preview { border: 1px dashed #6b5a33; border-radius: 8px; }
QLabel#status { color: #9aa3b2; }
"""


def pil_to_pixmap(im):
    im = im.convert("RGBA")
    raw = im.tobytes("raw", "RGBA")
    qimg = QImage(raw, im.width, im.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class GenWorker(QThread):
    """Stable Diffusion txt2img in the background."""
    progress = Signal(int)
    status = Signal(str)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, prompt, negative, seed, **kw):
        super().__init__()
        self.prompt, self.negative, self.seed = prompt, negative, seed
        self._engine = None

    def run(self):
        try:
            eng = SDEngineThreaded(self)
            img = eng.generate(self.prompt, self.negative, self.seed)
            self.done.emit(img)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class SDEngineThreaded:
    """Thin wrapper routing sd_engine callbacks into Qt signals."""

    def __init__(self, worker):
        self._w = worker
        self._eng = sd_engine.SDEngine(
            progress_cb=lambda s, p: self._w.progress.emit(int(p)),
            log_cb=lambda t: self._w.status.emit(t))

    def generate(self, *a, **kw):
        return self._eng.generate(*a, **kw)


class FxWorker(QThread):
    """fx.animate (+ optional GIF export) in the background."""
    progress = Signal(int)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, image, name, amount, w, h, fps, duration, seed,
                 out_path=None):
        super().__init__()
        self.image, self.name, self.amount = image, name, amount
        self.w, self.h, self.fps, self.duration, self.seed = w, h, fps, duration, seed
        self.out_path = out_path
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            def cb(i, n):
                self.progress.emit(int(100 * i / max(1, n)))
                return not self._cancel

            frames = fx.animate(self.image, self.name, self.amount,
                                w=self.w, h=self.h, fps=self.fps,
                                duration=self.duration, seed=self.seed,
                                on_frame=cb)
            if self._cancel:
                return
            if self.out_path:
                size = fx.export_gif(frames, self.out_path, fps=self.fps)
                self.done.emit(("file", self.out_path, size))
            else:
                self.done.emit(("frames", frames))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class EngineCheckWorker(QThread):
    done = Signal(object)

    def run(self):
        try:
            self.done.emit(sd_engine.check_engine())
        except Exception as e:  # noqa: BLE001
            self.done.emit([("خطا", False, str(e)[:160])])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("استودیو پرامپت دونیت — PromptDonate")
        self.setLayoutDirection(Qt.RightToLeft)
        self.resize(1180, 760)
        self.gen_image = None
        self.frames = []
        self.worker = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick)
        self._play_i = 0
        self._build_ui()
        self._check_engine()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        # ---- right: controls ----
        side = QWidget()
        form = QVBoxLayout(side)
        form.setSpacing(8)

        g1 = QGroupBox("🎨 پرامپت و سبک")
        v1 = QVBoxLayout(g1)
        self.preset_combo = QComboBox()
        for fa, _, _ in PRESETS:
            self.preset_combo.addItem(fa)
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        v1.addWidget(QLabel("سبک آماده:"))
        v1.addWidget(self.preset_combo)
        v1.addWidget(QLabel("پرامپت (انگلیسی):"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setLayoutDirection(Qt.LeftToRight)
        self.prompt_edit.setAlignment(Qt.AlignLeft)
        self.prompt_edit.setFixedHeight(110)
        v1.addWidget(self.prompt_edit)
        form.addWidget(g1)

        g2 = QGroupBox("💰 مشخصات دونیت")
        v2 = QVBoxLayout(g2)
        self.name_edit = QLineEdit("آرش")
        self.amount_edit = QLineEdit("۱۰۰٬۰۰۰ تومان")
        v2.addWidget(QLabel("نام دونیت‌کننده:"))
        v2.addWidget(self.name_edit)
        v2.addWidget(QLabel("مبلغ:"))
        v2.addWidget(self.amount_edit)
        row = QHBoxLayout()
        self.size_combo = QComboBox()
        for fa, _, _ in SIZES:
            self.size_combo.addItem(fa)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2 ** 31 - 1)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText("تصادفی")
        row.addWidget(QLabel("سایز:"))
        row.addWidget(self.size_combo)
        row.addWidget(QLabel("seed:"))
        row.addWidget(self.seed_spin)
        v2.addLayout(row)
        form.addWidget(g2)

        g3 = QGroupBox("⚙️ عملیات")
        v3 = QVBoxLayout(g3)
        self.btn_gen = QPushButton("🎨 تولید تصویر")
        self.btn_gen.setObjectName("primary")
        self.btn_gen.clicked.connect(self.start_generate)
        self.btn_play = QPushButton("▶ پخش پیش‌نمایش")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        self.btn_export = QPushButton("💾 خروجی گیف شفاف")
        self.btn_export.clicked.connect(self.start_export)
        self.btn_export.setEnabled(False)
        v3.addWidget(self.btn_gen)
        v3.addWidget(self.btn_play)
        v3.addWidget(self.btn_export)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        v3.addWidget(self.progress)
        form.addWidget(g3)

        self.status = QLabel("آماده")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        form.addWidget(self.status)
        form.addStretch(1)
        main.addWidget(side, 1)

        # ---- left: preview ----
        self.preview = QLabel("اول یک سبک انتخاب کن و «تولید تصویر» را بزن")
        self.preview.setObjectName("preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 400)
        main.addWidget(self.preview, 2)

        self._on_preset(0)

    # -------------------------------------------------------------- engine --
    def _check_engine(self):
        self.status.setText("بررسی موتور AI ...")
        self._chk = EngineCheckWorker()
        self._chk.done.connect(self._on_engine_checked)
        self._chk.start()

    def _on_engine_checked(self, rows):
        try:
            self._chk.quit()
            self._chk.wait(2000)
        except Exception:
            pass
        dev = [r for r in rows if r[0] == "دستگاه محاسبه"]
        if dev and dev[0][1]:
            self.status.setText(f"موتور AI آماده — {dev[0][2]}")
        else:
            bad = "; ".join(f"{n}: {d}" for n, ok, d in rows if not ok)
            self.status.setText(f"موتور AI در دسترس نیست — {bad}")

    def _on_preset(self, idx):
        fa, prompt, _neg = PRESETS[idx]
        if fa != "سفارشی":
            self.prompt_edit.setPlainText(prompt)

    def _set_working(self, working, label=""):
        for b in (self.btn_gen, self.btn_play, self.btn_export,
                  self.preset_combo):
            b.setEnabled(not working)
        if working:
            self.status.setText(label)
            self.progress.setValue(0)

    # ------------------------------------------------------------ generate --
    def start_generate(self):
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "پرامپت خالی", "اول یک پرامپت بنویس یا سبک انتخاب کن.")
            return
        self._stop_play()
        self._set_working(True, "تولید تصویر با AI ...")
        w = GenWorker(prompt, "", int(self.seed_spin.value()))
        w.progress.connect(self.progress.setValue)
        w.status.connect(self.status.setText)
        w.done.connect(self._on_gen_done)
        w.error.connect(self._on_error)
        self.worker = w
        w.start()

    def _on_gen_done(self, img):
        self._set_working(False)
        self.gen_image = img
        seed = img.info.get("seed")
        self.status.setText(f"تصویر ساخته شد ✅" + (f" (seed={seed})" if seed is not None else ""))
        self.preview.setPixmap(pil_to_pixmap(img).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.btn_play.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.worker = None

    # --------------------------------------------------------------- play --
    def _fx_params(self):
        _, w, h = SIZES[self.size_combo.currentIndex()]
        seed = int(self.seed_spin.value())
        if seed < 0:
            seed = int(time.time()) % (2 ** 31)
        return dict(image=self.gen_image, name=self.name_edit.text().strip(),
                    amount=self.amount_edit.text().strip(), w=w, h=h,
                    fps=30, duration=4.0, seed=seed)

    def toggle_play(self):
        if self._play_timer.isActive():
            self._stop_play()
            return
        if self.gen_image is None:
            return
        self._set_working(True, "ساخت انیمیشن ...")
        w = FxWorker(**self._fx_params())
        w.progress.connect(self.progress.setValue)
        w.done.connect(self._on_fx_done)
        w.error.connect(self._on_error)
        self.worker = w
        w.start()

    def _on_fx_done(self, payload):
        self._set_working(False)
        kind = payload[0]
        if kind == "frames":
            self.frames = payload[1]
            self._play_i = 0
            self._play_timer.start(33)
            self.btn_play.setText("⏸ توقف")
            self.btn_play.setEnabled(True)
        self.worker = None

    def _stop_play(self):
        self._play_timer.stop()
        self.btn_play.setText("▶ پخش پیش‌نمایش")
        if self.worker is not None:
            try:
                self.worker.cancel()
            except Exception:
                pass

    def _tick(self):
        if not self.frames:
            return
        fr = self.frames[self._play_i % len(self.frames)]
        self._play_i += 1
        self.preview.setPixmap(pil_to_pixmap(fr).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # -------------------------------------------------------------- export --
    def start_export(self):
        if self.gen_image is None:
            return
        os.makedirs(OUT_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        default = os.path.join(OUT_DIR, f"prompt_donate_{ts}.gif")
        path, _ = QFileDialog.getSaveFileName(self, "ذخیره گیف شفاف",
                                              default, "GIF (*.gif)")
        if not path:
            return
        self._stop_play()
        self._set_working(True, "رندر و خروجی گیف ...")
        w = FxWorker(out_path=path, **self._fx_params())
        w.progress.connect(self.progress.setValue)
        w.done.connect(self._on_export_done)
        w.error.connect(self._on_error)
        self.worker = w
        w.start()

    def _on_export_done(self, payload):
        self._set_working(False)
        _, path, size = payload
        mb = size / (1024 * 1024)
        self.status.setText(f"گیف ذخیره شد ✅ ({mb:.1f} مگابایت)")
        QMessageBox.information(self, "تمام شد", f"گیف شفاف ذخیره شد:\n{path}")
        self.worker = None

    def _on_error(self, msg):
        self._set_working(False)
        self.status.setText("خطا ❌")
        QMessageBox.critical(self, "خطا", msg[:600])
        self.worker = None

    def closeEvent(self, ev):
        self._stop_play()
        super().closeEvent(ev)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PromptDonate")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
