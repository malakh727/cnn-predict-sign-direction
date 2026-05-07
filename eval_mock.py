"""Evaluate saved model weights against the mock test set."""

import numpy as np
from cnn import (load_dataset, extract_features, FCNetwork,
                 get_fixed_kernels, get_layer1_kernels,
                 HIDDEN_SIZE, NUM_CLASSES, LEARNING_RATE, DROPOUT_RATE,
                 SIGN_NAMES)

MOCK_DIR = "./mock_test"

kernels_l0 = get_fixed_kernels()
kernels_l1 = get_layer1_kernels(len(kernels_l0))

feat_max = np.load("feat_max.npy")

nn = FCNetwork(input_size=32, hidden_size=HIDDEN_SIZE,
               output_size=NUM_CLASSES, lr=LEARNING_RATE,
               dropout_rate=DROPOUT_RATE)
nn.load("model_weights.npz")

X_raw, y = load_dataset(MOCK_DIR)
X = np.array([extract_features(img, kernels_l0, kernels_l1)
              for img in X_raw], dtype=np.float32)
X = X / feat_max

correct = 0
errors  = []
for i in range(len(X)):
    pred = nn.predict(X[i])
    if pred == y[i]:
        correct += 1
    else:
        errors.append((y[i], pred))

acc = correct / len(X) * 100
print(f"\n  Mock Test Accuracy: {correct}/{len(X)} = {acc:.1f}%")

if errors:
    print(f"\n  Misclassified ({len(errors)}):")
    by_true = {}
    for true, pred in errors:
        by_true.setdefault(true, []).append(pred)
    for true in sorted(by_true):
        preds = [SIGN_NAMES[p] for p in by_true[true]]
        print(f"    {SIGN_NAMES[true]:<22s} -> {preds}")
