"""
Interactive Traffic Sign Classifier
Draw a 19x19 sign using the mouse, click Classify to get a prediction.

Left-click / drag  = draw (black)
Right-click / drag = erase (white)
"""
import os
import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from PIL import Image

from cnn import (
    extract_features, get_fixed_kernels, get_layer1_kernels,
    FCNetwork, SIGN_NAMES, THRESHOLD,
    HIDDEN_SIZE, NUM_CLASSES, LEARNING_RATE, DROPOUT_RATE,
)

# ── Config ────────────────────────────────────────────────────────────────────
CELL      = 25          # px per grid cell
GRID      = 19          # 19×19
CANVAS_PX = CELL * GRID # 475
REF_PX    = 190         # reference image display size
BASE_DIR  = "./base_images"

# ── Load model once at startup ────────────────────────────────────────────────
_kernels_l0 = get_fixed_kernels()
_kernels_l1 = get_layer1_kernels(len(_kernels_l0))
_n_features = len(_kernels_l0) * 4   # 10 channels × 4 quadrants = 40
_nn         = FCNetwork(input_size=_n_features, hidden_size=HIDDEN_SIZE,
                        output_size=NUM_CLASSES, lr=LEARNING_RATE,
                        dropout_rate=DROPOUT_RATE)
_data     = np.load("parameter.npz")
_nn.W1    = _data["W1"]
_nn.b1    = _data["b1"]
_nn.W2    = _data["W2"]
_nn.b2    = _data["b2"]
_feat_max = _data["feat_max"]


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity between two binary arrays (ignores white background)."""
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _load_base(sign_id: int) -> np.ndarray:
    """Load and binarize base_images/{sign_id}_0.png → 19×19 float32."""
    path = os.path.join(BASE_DIR, f"{sign_id}_0.png")
    img  = Image.open(path).convert("L").resize((GRID, GRID), Image.NEAREST)
    arr  = np.array(img, dtype=np.float32)
    return (arr < THRESHOLD).astype(np.float32)



class DrawApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Traffic Sign Classifier")
        root.resizable(False, False)

        # internal 19×19 grid: 0 = white, 1 = black
        self.grid = np.zeros((GRID, GRID), dtype=np.float32)

        self._build_ui()
        self._draw_grid()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        BG      = "#1e1e2e"
        PANEL   = "#2a2a3e"
        ACCENT  = "#7c3aed"
        FG      = "#e2e8f0"
        SUBTLE  = "#94a3b8"

        self.root.configure(bg=BG)

        title_font  = tkfont.Font(family="Helvetica", size=13, weight="bold")
        label_font  = tkfont.Font(family="Helvetica", size=10)
        result_font = tkfont.Font(family="Helvetica", size=14, weight="bold")
        btn_font    = tkfont.Font(family="Helvetica", size=11, weight="bold")
        hint_font   = tkfont.Font(family="Helvetica", size=8)

        # ── Title bar ────────────────────────────────────────────────────────
        tk.Label(self.root, text="Traffic Sign Classifier",
                 font=title_font, bg=BG, fg=FG,
                 pady=10).grid(row=0, column=0, columnspan=2)

        # ── Canvas ───────────────────────────────────────────────────────────
        canvas_frame = tk.Frame(self.root, bg=ACCENT, padx=2, pady=2)
        canvas_frame.grid(row=1, column=0, padx=(15, 8), pady=(0, 10))

        self.canvas = tk.Canvas(canvas_frame,
                                width=CANVAS_PX, height=CANVAS_PX,
                                bg="white", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()

        self.canvas.bind("<Button-1>",        self._on_press_draw)
        self.canvas.bind("<B1-Motion>",       self._on_drag_draw)
        self.canvas.bind("<Button-3>",        self._on_press_erase)
        self.canvas.bind("<B3-Motion>",       self._on_drag_erase)

        tk.Label(self.root,
                 text="Left-click: draw   Right-click: erase",
                 font=hint_font, bg=BG, fg=SUBTLE
                 ).grid(row=2, column=0)

        # ── Result panel ─────────────────────────────────────────────────────
        panel = tk.Frame(self.root, bg=PANEL, padx=16, pady=16)
        panel.grid(row=1, column=1, padx=(8, 15), pady=(0, 10), sticky="n")

        tk.Label(panel, text="Prediction", font=label_font,
                 bg=PANEL, fg=SUBTLE).pack(anchor="w")

        self.lbl_pred = tk.Label(panel, text="—",
                                 font=result_font, bg=PANEL, fg=FG,
                                 width=18, anchor="w")
        self.lbl_pred.pack(anchor="w", pady=(2, 12))

        tk.Label(panel, text="Confidence", font=label_font,
                 bg=PANEL, fg=SUBTLE).pack(anchor="w")
        self.lbl_conf = tk.Label(panel, text="—",
                                 font=result_font, bg=PANEL, fg="#34d399",
                                 anchor="w")
        self.lbl_conf.pack(anchor="w", pady=(2, 12))

        tk.Label(panel, text="Similarity to original sign",
                 font=label_font, bg=PANEL, fg=SUBTLE).pack(anchor="w")
        self.lbl_sim = tk.Label(panel, text="—",
                                font=result_font, bg=PANEL, fg="#60a5fa",
                                anchor="w")
        self.lbl_sim.pack(anchor="w", pady=(2, 16))

        tk.Label(panel, text="Reference sign",
                 font=label_font, bg=PANEL, fg=SUBTLE).pack(anchor="w")

        self.ref_canvas = tk.Canvas(panel, width=REF_PX, height=REF_PX,
                                    bg="#2a2a3e", highlightthickness=1,
                                    highlightbackground="#475569")
        self.ref_canvas.pack(pady=(4, 0))

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(0, 15))

        self.btn_classify = tk.Button(
            btn_frame, text="Classify", font=btn_font,
            bg=ACCENT, fg="white", activebackground="#6d28d9",
            activeforeground="white", relief="flat",
            padx=28, pady=8,
            command=self.classify)
        self.btn_classify.pack(side="left", padx=12)

        tk.Button(
            btn_frame, text="Clear", font=btn_font,
            bg="#475569", fg="white", activebackground="#334155",
            activeforeground="white", relief="flat",
            padx=28, pady=8,
            command=self.clear).pack(side="left", padx=12)

    # ── Grid drawing helpers ──────────────────────────────────────────────────

    def _draw_grid(self):
        """Redraw the entire canvas from self.grid."""
        self.canvas.delete("all")
        for r in range(GRID):
            for c in range(GRID):
                x0, y0 = c * CELL, r * CELL
                x1, y1 = x0 + CELL, y0 + CELL
                fill = "black" if self.grid[r, c] else "white"
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill=fill, outline="#d1d5db",
                                             width=1)

    def _set_cell(self, event, value: float):
        c = event.x // CELL
        r = event.y // CELL
        if 0 <= r < GRID and 0 <= c < GRID:
            if self.grid[r, c] != value:
                self.grid[r, c] = value
                x0, y0 = c * CELL, r * CELL
                fill = "black" if value else "white"
                self.canvas.create_rectangle(x0, y0,
                                             x0 + CELL, y0 + CELL,
                                             fill=fill, outline="#d1d5db",
                                             width=1)

    def _on_press_draw(self,  e): self._set_cell(e, 1.0)
    def _on_drag_draw(self,   e): self._set_cell(e, 1.0)
    def _on_press_erase(self, e): self._set_cell(e, 0.0)
    def _on_drag_erase(self,  e): self._set_cell(e, 0.0)

    # ── Classify ──────────────────────────────────────────────────────────────

    def classify(self):
        features = extract_features(self.grid, _kernels_l0, _kernels_l1)
        features = features / _feat_max
        probs    = _nn.forward(features, training=False)
        pred     = int(np.argmax(probs))
        conf     = probs[pred] * 100.0

        base_arr  = _load_base(pred)
        sim       = _jaccard(self.grid, base_arr) * 100.0

        self.lbl_pred.config(text=SIGN_NAMES[pred])
        self.lbl_conf.config(text=f"{conf:.1f}%")
        self.lbl_sim.config(text=f"{sim:.1f}%")

        self._draw_ref(pred)

    def _draw_ref(self, sign_id: int):
        """Draw the base sign image on ref_canvas using rectangles."""
        base = _load_base(sign_id)  # 19×19 binary
        cell = REF_PX // GRID       # 10px per cell
        self.ref_canvas.delete("all")
        for r in range(GRID):
            for c in range(GRID):
                x0, y0 = c * cell, r * cell
                fill = "black" if base[r, c] else "white"
                self.ref_canvas.create_rectangle(
                    x0, y0, x0 + cell, y0 + cell,
                    fill=fill, outline=fill)

    # ── Clear ─────────────────────────────────────────────────────────────────

    def clear(self):
        self.grid[:] = 0
        self._draw_grid()
        self.lbl_pred.config(text="—")
        self.lbl_conf.config(text="—")
        self.lbl_sim.config(text="—")
        self.ref_canvas.delete("all")


def main():
    root = tk.Tk()
    DrawApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
