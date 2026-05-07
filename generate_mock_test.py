"""
Realistic Dr. Test Data Generator
Simulates the professor's test images: mild, realistic variations only.
  - ~7 random noise dots
  - small shifts (1-4 px)
  - slight rotations (±3° to ±8°)
  - combinations of the above

Output: ../TestingImages/<sign>_<version>.png  (20 per sign = 200 total)
"""

import os
import numpy as np
from PIL import Image

BASE_DIR   = "./base_images"
OUTPUT_DIR = "../TestingImages"
IMG_SIZE   = (19, 19)
THRESHOLD  = 128
TOTAL_PIXELS = IMG_SIZE[0] * IMG_SIZE[1]   # 361

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_binarize(path):
    img = Image.open(path).convert("L").resize(IMG_SIZE, Image.NEAREST)
    return (np.array(img, dtype=np.float32) < THRESHOLD).astype(np.float32)


def save(arr, sign, version):
    img = Image.fromarray(((1 - arr) * 255).astype(np.uint8), mode="L")
    img.save(os.path.join(OUTPUT_DIR, f"{sign}_{version}.png"))


def add_dots(arr, n_dots=7):
    """Flip exactly n_dots random pixels."""
    noisy = arr.copy()
    positions = np.random.choice(TOTAL_PIXELS, size=n_dots, replace=False)
    flat = noisy.flatten()
    flat[positions] = 1.0 - flat[positions]
    return flat.reshape(IMG_SIZE)


def shift(arr, dx, dy):
    out = np.zeros_like(arr)
    x0  = max(dx, 0);  x1  = min(IMG_SIZE[0] + dx, IMG_SIZE[0])
    y0  = max(dy, 0);  y1  = min(IMG_SIZE[1] + dy, IMG_SIZE[1])
    sx0 = max(-dx, 0); sx1 = min(IMG_SIZE[0] - dx, IMG_SIZE[0])
    sy0 = max(-dy, 0); sy1 = min(IMG_SIZE[1] - dy, IMG_SIZE[1])
    out[y0:y1, x0:x1] = arr[sy0:sy1, sx0:sx1]
    return out


def rotate(arr, angle):
    img = Image.fromarray(((1 - arr) * 255).astype(np.uint8), mode="L")
    rotated = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)
    return (np.array(rotated, dtype=np.float32) < THRESHOLD).astype(np.float32)


def generate(base, sign):
    np.random.seed(sign * 31 + 7)
    version = 0

    def s(arr):
        nonlocal version
        save(arr, sign, version)
        version += 1

    # v0: clean original — exactly as provided
    s(base)

    # v1–4: tiny rotations only (±3°, ±5°, ±7°, ±8°)
    for angle in [-8, -5, 5, 8]:
        s(rotate(base, angle))

    # v5–8: small shifts only (1–2 px, cardinal directions)
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        s(shift(base, dx, dy))

    # v9–12: light noise only (~7 dots)
    for n in [5, 7, 7, 9]:
        s(add_dots(base, n_dots=n))

    # v13–16: small shift + tiny rotation
    for angle, dx, dy in [(-5, -1, 0), (5, 1, 0), (-3, 0, -1), (3, 0, 1)]:
        s(shift(rotate(base, angle), dx, dy))

    # v17–19: shift + a few dots (most realistic: slightly off-center + minor scan noise)
    for dx, dy, n in [(-2, 1, 6), (1, -2, 7), (2, 2, 5)]:
        s(add_dots(shift(base, dx, dy), n_dots=n))

    return version


def main():
    total = 0
    for sign in range(10):
        path = os.path.join(BASE_DIR, f"{sign}_0.png")
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found.")
            continue
        base  = load_and_binarize(path)
        count = generate(base, sign)
        total += count
        print(f"  Sign {sign}: {count} images -> {OUTPUT_DIR}/")

    print(f"\nDone! {total} test images in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
