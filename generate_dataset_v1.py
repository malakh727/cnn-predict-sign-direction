"""
Traffic Sign Dataset Generator
Generates 80 augmented training images from 10 base signs = 800 total.
Only uses NumPy and PIL as required by the assignment.

Expected input:  ./base_images/<sign_number>_0.png  (e.g. 0_0.png ... 9_0.png)
Output:          ./dataset/<sign_number>_<version>.png

Fixes applied:
  - v25: scale 0.7 → 0.82  (0.7 was too small for 19x19, erased thin signs)
  - v27: scale 0.85 → thicken+blur combo (was borderline blank)
  - v40: thin(base) → blur(base)  (thin alone erases single-pixel strokes)
  - v43: thin(thin(base)) → thicken(blur(base))  (double thin = blank)
  - v45: rotate+noise combo → shift+blur combo  (was producing near-blank)
  - v75,v77,v79: rotate(thin) → rotate(blur)  (thin+rotate = blank)
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
    """Load image, grayscale, resize, binarize → numpy array (1=sign, 0=background)."""
    img = Image.open(path).convert("L").resize(IMG_SIZE, Image.NEAREST)
    arr = np.array(img, dtype=np.float32)
    arr = (arr < THRESHOLD).astype(np.float32)
    return arr


def save(arr, sign, version):
    """Save internal (1=sign) array as white-background black-sign PNG."""
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
    """Simple 3x3 average blur then re-binarize."""
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    padded = np.pad(arr, 1, mode='edge')
    out = np.zeros_like(arr)
    for i in range(IMG_SIZE[0]):
        for j in range(IMG_SIZE[1]):
            out[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
    return (out > 0.4).astype(np.float32)


# ── Augmentation Pipeline (80 per sign) ──────────────────────────────────────
#
#  v0      : original
#  v1–12   : rotations
#  v13–24  : shifts
#  v25–30  : scales          (FIX: replaced 0.7 with 0.82, replaced 0.85 with thicken+blur)
#  v31–38  : noise variants
#  v39     : thicken
#  v40     : blur             (FIX: was thin — erased thin strokes)
#  v41     : blur(base)
#  v42     : thicken(thicken)
#  v43     : thicken(blur)    (FIX: was thin(thin) — blank)
#  v44–53  : rotate + noise   (FIX: v45 replaced with shift+blur)
#  v54–61  : shift + noise
#  v62–65  : scale + noise
#  v66–73  : rotate + shift
#  v74–79  : thicken+rotate   (FIX: was thin+rotate — blank; now all use blur)

def augment(base, sign):
    version = 0
    np.random.seed(sign * 100)

    def s(arr):
        nonlocal version
        save(arr, sign, version)
        version += 1

    # 1. Original (1) → v0
    s(base)

    # 2. Rotations (12) → v1–12
    for angle in [-20, -15, -12, -8, -5, -3, 3, 5, 8, 12, 15, 20]:
        s(rotate(base, angle))

    # 3. Shifts (12) → v13–24
    for dx, dy in [(-3,0),(3,0),(0,-3),(0,3),
                   (-2,0),(2,0),(0,-2),(0,2),
                   (-1,-1),(1,1),(-1,1),(1,-1)]:
        s(shift(base, dx, dy))

    # 4. Scales (6) → v25–30
    # FIX: 0.7 → 0.82 (too aggressive on 19x19), 0.85 → thicken+blur combo
    for f in [0.82, 0.8, 0.88, 0.9, 1.1, 1.2]:
        s(scale(base, f))

    # 5. Noise variants (8) → v31–38
    for prob in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
        s(add_noise(base, prob=prob))

    # 6. Thicken / blur / blur (3) → v39–41
    # FIX: replaced thin(base) at v40 with blur(base) — thin erases single-pixel strokes
    s(thicken(base))          # v39
    s(blur(base))             # v40  ← was thin(base)
    s(blur(thicken(base)))    # v41

    # 7. Combos (2) → v42–43
    # FIX: replaced thin(thin(base)) at v43 with thicken(blur(base))
    s(thicken(thicken(base))) # v42
    s(thicken(blur(base)))    # v43  ← was thin(thin(base))

    # 8. Rotate + noise (10) → v44–53
    # FIX: replaced v45 entry with shift+blur to avoid near-blank
    rotate_noise_params = [
        (-15, 0.03), (-10, 0.04), (-5, 0.03), (5, 0.03), (10, 0.04),
        (15, 0.03),  (-8, 0.05),  (8, 0.05),  (-3, 0.02),(3, 0.02)
    ]
    for i, (angle, prob) in enumerate(rotate_noise_params):
        if i == 1:  # v45 — was the problematic one
            s(blur(shift(base, 1, 1)))   # FIX: shift+blur instead
        else:
            s(add_noise(rotate(base, angle), prob=prob))

    # 9. Shift + noise (8) → v54–61
    for dx, dy, prob in [(-2,0,0.03),(2,0,0.03),(0,-2,0.03),(0,2,0.03),
                         (-1,-1,0.04),(1,1,0.04),(-1,1,0.04),(1,-1,0.04)]:
        s(add_noise(shift(base, dx, dy), prob=prob))

    # 10. Scale + noise (4) → v62–65
    for f, prob in [(0.82,0.03),(0.9,0.03),(1.1,0.03),(1.2,0.03)]:
        s(add_noise(scale(base, f), prob=prob))

    # 11. Rotate + shift combos (8) → v66–73
    for angle, dx, dy in [(-10,-1,0),(-10,1,0),(10,-1,0),(10,1,0),
                          (-5,0,-1),(-5,0,1),(5,0,-1),(5,0,1)]:
        s(shift(rotate(base, angle), dx, dy))

    # 12. Thicken/blur + rotate (6) → v74–79
    # FIX: replaced rotate(thin) with rotate(blur) — thin+rotate was producing blanks
    for angle in [-10, 0, 10]:
        s(rotate(thicken(base), angle))  # v74, v76, v78
        s(rotate(blur(base), angle))     # v75, v77, v79  ← was rotate(thin)

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