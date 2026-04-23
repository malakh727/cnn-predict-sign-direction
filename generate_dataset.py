"""
Traffic Sign Dataset Generator - v3
Generates 150 augmented training images per sign = 1500 total.
Only uses NumPy and PIL as required by the assignment.

Expected input:  ./base_images/<sign_number>_0.png
Output:          ./dataset/<sign_number>_<version>.png  (0–149)
"""

import os
import numpy as np
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR   = "./base_images"
OUTPUT_DIR = "./dataset"
IMG_SIZE   = (19, 19)
NUM_SIGNS  = 10
THRESHOLD  = 128
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_binarize(path):
    img = Image.open(path).convert("L").resize(IMG_SIZE, Image.NEAREST)
    arr = np.array(img, dtype=np.float32)
    return (arr < THRESHOLD).astype(np.float32)


def save(arr, sign, version):
    img = Image.fromarray(((1 - arr) * 255).astype(np.uint8), mode="L")
    img.save(os.path.join(OUTPUT_DIR, f"{sign}_{version}.png"))


# ── Augmentation Functions ────────────────────────────────────────────────────

def rotate(arr, angle):
    img = Image.fromarray(((1 - arr) * 255).astype(np.uint8), mode="L")
    rotated = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)
    out = np.array(rotated, dtype=np.float32)
    return (out < THRESHOLD).astype(np.float32)


def add_noise(arr, prob=0.04):
    noisy = arr.copy()
    mask = np.random.rand(*arr.shape) < prob
    noisy[mask] = 1.0 - noisy[mask]
    return noisy


def shift(arr, dx, dy):
    out = np.zeros_like(arr)
    x0  = max(dx, 0);  x1  = min(IMG_SIZE[0] + dx, IMG_SIZE[0])
    y0  = max(dy, 0);  y1  = min(IMG_SIZE[1] + dy, IMG_SIZE[1])
    sx0 = max(-dx, 0); sx1 = min(IMG_SIZE[0] - dx, IMG_SIZE[0])
    sy0 = max(-dy, 0); sy1 = min(IMG_SIZE[1] - dy, IMG_SIZE[1])
    out[y0:y1, x0:x1] = arr[sy0:sy1, sx0:sx1]
    return out


def scale(arr, factor):
    new_w = max(1, int(IMG_SIZE[0] * factor))
    new_h = max(1, int(IMG_SIZE[1] * factor))
    img = Image.fromarray(((1 - arr) * 255).astype(np.uint8), mode="L")
    scaled = img.resize((new_w, new_h), Image.NEAREST)
    canvas = Image.new("L", IMG_SIZE, 255)
    paste_x = (IMG_SIZE[0] - new_w) // 2
    paste_y = (IMG_SIZE[1] - new_h) // 2
    canvas.paste(scaled, (paste_x, paste_y))
    out = np.array(canvas, dtype=np.float32)
    return (out < THRESHOLD).astype(np.float32)


def thicken(arr):
    out = arr.astype(np.uint8).copy()
    a   = arr.astype(np.uint8)
    out[1:, :]  |= a[:-1, :]
    out[:-1, :] |= a[1:, :]
    out[:, 1:]  |= a[:, :-1]
    out[:, :-1] |= a[:, 1:]
    return out.astype(np.float32)


def blur(arr):
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    padded = np.pad(arr, 1, mode='edge')
    out = np.zeros_like(arr)
    for i in range(IMG_SIZE[0]):
        for j in range(IMG_SIZE[1]):
            out[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
    return (out > 0.4).astype(np.float32)


def salt_pepper(arr, prob=0.05):
    noisy = arr.copy()
    rnd = np.random.rand(*arr.shape)
    noisy[rnd < prob / 2] = 1.0
    noisy[rnd > 1 - prob / 2] = 0.0
    return noisy


def elastic(arr, dx, dy):
    """Combine shift + slight rotation to simulate hand-drawn variation."""
    return shift(rotate(arr, dx * 2), dx, dy)


# ── Augmentation Pipeline (150 per sign) ─────────────────────────────────────
#
#  Group 1  v0       : original
#  Group 2  v1–16    : rotations (fine-grained)
#  Group 3  v17–32   : shifts (all directions including diagonals)
#  Group 4  v33–40   : scales
#  Group 5  v41–52   : noise variants (random flip noise)
#  Group 6  v53–60   : salt & pepper noise variants
#  Group 7  v61–63   : thicken / blur / thicken+blur
#  Group 8  v64–65   : double thicken, blur+thicken
#  Group 9  v66–79   : rotate + noise combos
#  Group 10 v80–91   : shift + noise combos
#  Group 11 v92–99   : scale + noise combos
#  Group 12 v100–115 : rotate + shift combos
#  Group 13 v116–125 : thicken/blur + rotate combos
#  Group 14 v126–137 : elastic (shift+rotate) combos
#  Group 15 v138–149 : triple combos (rotate + shift + noise)

def augment(base, sign):
    version = 0
    np.random.seed(sign * 100)

    def s(arr):
        nonlocal version
        save(arr, sign, version)
        version += 1

    # Group 1: Original (1) → v0
    s(base)

    # Group 2: Rotations fine-grained (16) → v1–16
    for angle in [-20, -17, -15, -12, -10, -7, -5, -3, 3, 5, 7, 10, 12, 15, 17, 20]:
        s(rotate(base, angle))

    # Group 3: Shifts all directions (16) → v17–32
    for dx, dy in [(-3,0),(3,0),(0,-3),(0,3),
                   (-2,0),(2,0),(0,-2),(0,2),
                   (-1,-1),(1,1),(-1,1),(1,-1),
                   (-2,-1),(2,1),(-1,-2),(1,2)]:
        s(shift(base, dx, dy))

    # Group 4: Scales (8) → v33–40
    for f in [0.75, 0.80, 0.85, 0.88, 0.92, 0.95, 1.10, 1.20]:
        s(scale(base, f))

    # Group 5: Noise variants random flip (12) → v41–52
    for prob in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]:
        s(add_noise(base, prob=prob))

    # Group 6: Salt & pepper noise (8) → v53–60
    for prob in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        s(salt_pepper(base, prob=prob))

    # Group 7: Thicken / blur combos (3) → v61–63
    s(thicken(base))
    s(blur(base))
    s(blur(thicken(base)))

    # Group 8: Double thicken, blur+thicken (2) → v64–65
    s(thicken(thicken(base)))
    s(thicken(blur(base)))

    # Group 9: Rotate + noise (14) → v66–79
    for angle, prob in [(-15,0.03),(-12,0.04),(-10,0.03),(-7,0.04),(-5,0.03),(-3,0.02),
                        (3,0.02),(5,0.03),(7,0.04),(10,0.03),(12,0.04),(15,0.03),
                        (-8,0.05),(8,0.05)]:
        s(add_noise(rotate(base, angle), prob=prob))

    # Group 10: Shift + noise (12) → v80–91
    for dx, dy, prob in [(-2,0,0.03),(2,0,0.03),(0,-2,0.03),(0,2,0.03),
                         (-1,-1,0.04),(1,1,0.04),(-1,1,0.04),(1,-1,0.04),
                         (-2,-1,0.03),(2,1,0.03),(-1,-2,0.03),(1,2,0.03)]:
        s(add_noise(shift(base, dx, dy), prob=prob))

    # Group 11: Scale + noise (8) → v92–99
    for f, prob in [(0.80,0.03),(0.85,0.03),(0.90,0.03),(0.95,0.03),
                    (1.05,0.03),(1.10,0.03),(1.15,0.03),(1.20,0.03)]:
        s(add_noise(scale(base, f), prob=prob))

    # Group 12: Rotate + shift combos (16) → v100–115
    for angle, dx, dy in [(-10,-1,0),(-10,1,0),(10,-1,0),(10,1,0),
                          (-5,0,-1),(-5,0,1),(5,0,-1),(5,0,1),
                          (-15,-1,-1),(-15,1,1),(15,-1,-1),(15,1,1),
                          (-7,-1,1),(7,1,-1),(-3,2,0),(3,-2,0)]:
        s(shift(rotate(base, angle), dx, dy))

    # Group 13: Thicken/blur + rotate (10) → v116–125
    for angle in [-15, -10, -5, 0, 5]:
        s(rotate(thicken(base), angle))
        s(rotate(blur(base), angle))

    # Group 14: Elastic combos (12) → v126–137
    for dx, dy in [(-2,-1),(-1,-2),(2,1),(1,2),
                   (-2,1),(1,-2),(2,-1),(-1,2),
                   (-3,0),(3,0),(0,-3),(0,3)]:
        s(elastic(base, dx, dy))

    # Group 15: Triple combos rotate+shift+noise (12) → v138–149
    for angle, dx, dy, prob in [(-10,-1,0,0.03),(-10,1,0,0.03),
                                 (10,-1,0,0.03),(10,1,0,0.03),
                                 (-5,0,-1,0.04),(-5,0,1,0.04),
                                 (5,0,-1,0.04),(5,0,1,0.04),
                                 (-15,1,1,0.03),(15,-1,-1,0.03),
                                 (-7,1,-1,0.03),(7,-1,1,0.03)]:
        s(add_noise(shift(rotate(base, angle), dx, dy), prob=prob))

    print(f"  Sign {sign}: {version} images generated.")
    return version


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    total = 0
    for sign in range(NUM_SIGNS):
        path = os.path.join(BASE_DIR, f"{sign}_0.png")
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found.")
            continue
        base  = load_and_binarize(path)
        count = augment(base, sign)
        total += count

    print(f"\nDone! Total images in ./dataset: {total}")
    print("Each image is named <sign_number>_<version_number>.png")


if __name__ == "__main__":
    main()
