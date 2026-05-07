"""
Splits ./dataset/ into ./train/ (80%) and ./test/ (20%) stratified per class.
120 train + 30 test per sign = 1200 train / 300 test total.
"""

import os
import shutil
import numpy as np

SOURCE_DIR = "./dataset"
TRAIN_DIR  = "./train"
TEST_DIR   = "./test"
NUM_SIGNS  = 10
SEED       = 42

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR,  exist_ok=True)

np.random.seed(SEED)

total_train = 0
total_test  = 0

for sign in range(NUM_SIGNS):
    files = sorted(
        f for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith('.png') and f.split('_')[0] == str(sign)
    )

    if not files:
        print(f"  [SKIP] No files found for sign {sign}")
        continue

    indices   = np.random.permutation(len(files))
    n_test    = max(1, int(len(files) * 0.2))
    test_idx  = set(indices[:n_test])

    n_tr = n_te = 0
    for i, fname in enumerate(files):
        src = os.path.join(SOURCE_DIR, fname)
        if i in test_idx:
            shutil.copy2(src, os.path.join(TEST_DIR, fname))
            n_te += 1
        else:
            shutil.copy2(src, os.path.join(TRAIN_DIR, fname))
            n_tr += 1

    print(f"  Sign {sign}: {n_tr} train  |  {n_te} test")
    total_train += n_tr
    total_test  += n_te

print(f"\nDone!  Train: {total_train}  |  Test: {total_test}")
print(f"Dirs:  {TRAIN_DIR}/  and  {TEST_DIR}/")
