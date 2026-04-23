"""
Traffic Sign Testing Dataset Generator
Generates 20 NEW augmented images per sign (versions 81-100) = 200 total.
Augmentations are intentionally different from the training set.
Only uses NumPy and PIL as required by the assignment.

Expected input:  ./base_images/<sign_number>_0.png
Output:          ./testing_dataset/<sign_number>_<version>.png  (81–100)
"""

import os
import numpy as np
from PIL import Image

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR   = "./base_images"
OUTPUT_DIR = "./testing_dataset"
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


def salt_pepper_noise(arr, prob=0.05):
    """Adds salt & pepper noise — different pattern from gaussian-style noise in training."""
    noisy = arr.copy()
    rnd = np.random.rand(*arr.shape)
    noisy[rnd < prob / 2] = 1.0        # salt
    noisy[rnd > 1 - prob / 2] = 0.0   # pepper
    return noisy


def rotate_and_scale(arr, angle, factor):
    return scale(rotate(arr, angle), factor)


def shift_and_blur(arr, dx, dy):
    return blur(shift(arr, dx, dy))


def thicken_and_shift(arr, dx, dy):
    return shift(thicken(arr), dx, dy)


def rotate_and_shift(arr, angle, dx, dy):
    return shift(rotate(arr, angle), dx, dy)


# ── Testing Augmentation Pipeline (versions 81–100) ──────────────────────────
#
#  These combinations are intentionally different from training (v0–v79):
#  - Uses different rotation angles not in training set
#  - Uses salt & pepper noise instead of random flip noise
#  - Uses diagonal shifts not used in training
#  - Uses rotate+scale combos (not in training)
#  - Uses different scale factors
#  - Uses thicken+shift combos
#  - Uses higher noise probabilities
#
def augment_test(base, sign):
    version = 81
    np.random.seed(sign * 777 + 42)   # different seed from training

    def s(arr):
        nonlocal version
        save(arr, sign, version)
        version += 1

    # v81 — rotate +7° (not in training)
    s(rotate(base, 7))

    # v82 — rotate -7°
    s(rotate(base, -7))

    # v83 — rotate +17° (not in training)
    s(rotate(base, 17))

    # v84 — rotate -17°
    s(rotate(base, -17))

    # v85 — scale 0.95 (not in training)
    s(scale(base, 0.95))

    # v86 — scale 1.05 (not in training)
    s(scale(base, 1.05))

    # v87 — salt & pepper noise low
    s(salt_pepper_noise(base, prob=0.04))

    # v88 — salt & pepper noise medium
    s(salt_pepper_noise(base, prob=0.07))

    # v89 — salt & pepper noise high
    s(salt_pepper_noise(base, prob=0.10))

    # v90 — rotate+scale combo (7°, 0.92)
    s(rotate_and_scale(base, 7, 0.92))

    # v91 — rotate+scale combo (-7°, 1.08)
    s(rotate_and_scale(base, -7, 1.08))

    # v92 — diagonal shift (-2, -2) — not in training
    s(shift(base, -2, -2))

    # v93 — diagonal shift (2, 2)
    s(shift(base, 2, 2))

    # v94 — shift+blur (-1, -2)
    s(shift_and_blur(base, -1, -2))

    # v95 — shift+blur (1, 2)
    s(shift_and_blur(base, 1, 2))

    # v96 — thicken + diagonal shift
    s(thicken_and_shift(base, -1, -1))

    # v97 — thicken + diagonal shift other way
    s(thicken_and_shift(base, 1, -1))

    # v98 — rotate + shift diagonal combo
    s(rotate_and_shift(base, 10, -1, -1))

    # v99 — rotate + shift diagonal combo
    s(rotate_and_shift(base, -10, 1, 1))

    # v100 — thicken + noise (different combo from training)
    s(add_noise(thicken(base), prob=0.05))

    print(f"  Sign {sign}: versions 81–100 generated.")
    return version - 81   # returns 20


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    total = 0
    for sign in range(NUM_SIGNS):
        path = os.path.join(BASE_DIR, f"{sign}_0.png")
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found.")
            continue
        base  = load_and_binarize(path)
        count = augment_test(base, sign)
        total += count

    print(f"\nDone! Total testing images in ./testing_dataset: {total}")
    print("Versions 81–100 per sign, named <sign_number>_<version>.png")


if __name__ == "__main__":
    main()
