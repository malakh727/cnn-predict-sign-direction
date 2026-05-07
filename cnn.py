"""
  - 10 fixed kernels in layer 0 (horizontal, vertical, diagonals, edges, L-curves)
  - 40 composite features in layer 1 (all 4 quadrants per channel)
  - Dropout regularization in FC layer (rate=0.2) to reduce overfitting
  - Hidden layer: 128 neurons
  - Epochs: 700
  - Learning rate: 0.005
  - Trained on 1520 images (152 per sign)

Only uses NumPy and PIL as required by the assignment.
"""
import os
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
current_directory = os.getcwd()
images_directory  = os.path.join(current_directory, "dataset")

TRAIN_DIR     = images_directory
TEST_DIR      = "../TestingImages" # Change to "../TestingImages" if using the provided test set
IMG_SIZE      = (19, 19)
NUM_CLASSES   = 10
THRESHOLD     = 128

# FC NN hyperparameters
HIDDEN_SIZE   = 128
LEARNING_RATE = 0.005
EPOCHS        = 700
DROPOUT_RATE  = 0.2      # drop 20% of hidden neurons during training
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 1.  IMAGE LOADING & BINARIZATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_binarize(path):
    img = Image.open(path).convert("L").resize(IMG_SIZE, Image.NEAREST)
    arr = np.array(img, dtype=np.float32)
    return (arr < THRESHOLD).astype(np.float32)


def load_dataset(directory):
    X, y = [], []
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith('.png'):
            continue
        try:
            label = int(fname.split('_')[0])
        except ValueError:
            continue
        arr = load_and_binarize(os.path.join(directory, fname))
        X.append(arr)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FIXED KERNELS — LAYER 0  (10 kernels)
#     Original 4 from lecture + 4 edge detectors + 2 L-curve detectors
# ══════════════════════════════════════════════════════════════════════════════

def get_fixed_kernels():
    # K0: horizontal line
    K0 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    # K1: vertical line
    K1 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.float32)

    # K2: slope=-1 diagonal
    K2 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=np.float32)

    # K3: slope=+1 diagonal
    K3 = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ], dtype=np.float32)

    # K4: top-edge detector
    K4 = np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    # K5: bottom-edge detector
    K5 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ], dtype=np.float32)

    # K6: left-edge detector
    K6 = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ], dtype=np.float32)

    # K7: right-edge detector
    K7 = np.array([
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ], dtype=np.float32)

    # K8: bottom-right L-curve (curved arrow bending right then down)
    K8 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ], dtype=np.float32)

    # K9: bottom-left L-curve (curved arrow bending left then down)
    K9 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ], dtype=np.float32)

    return np.stack([K0, K1, K2, K3, K4, K5, K6, K7, K8, K9], axis=0)  # (10, 5, 5)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CONVOLUTION & POOLING
# ══════════════════════════════════════════════════════════════════════════════

def conv2d(image, kernel, stride=1, padding=0):
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
    H, W = image.shape
    F    = kernel.shape[0]
    H_out = (H - F) // stride + 1
    W_out = (W - F) // stride + 1
    shape   = (H_out, W_out, F, F)
    strides = (image.strides[0] * stride, image.strides[1] * stride,
               image.strides[0], image.strides[1])
    patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    return (patches * kernel).sum(axis=(2, 3)).astype(np.float32)


def relu(x):
    return np.maximum(0, x)


def maxpool2d(fm, pool_size=2, stride=2):
    H, W  = fm.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    shape   = (H_out, W_out, pool_size, pool_size)
    strides = (fm.strides[0] * stride, fm.strides[1] * stride,
               fm.strides[0], fm.strides[1])
    patches = np.lib.stride_tricks.as_strided(fm, shape=shape, strides=strides)
    return patches.max(axis=(2, 3)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LAYER 0 — Extract Basic Features
#     Input : 19x19x1
#     Conv  : F=5x5x10, S=2  ->  8x8x10
#     Pool  : F=2, S=2        ->  4x4x10
# ══════════════════════════════════════════════════════════════════════════════

def layer0(image, kernels):
    maps = []
    for k in range(len(kernels)):
        fm = conv2d(image, kernels[k], stride=2, padding=0)
        fm = relu(fm)
        fm = maxpool2d(fm, pool_size=2, stride=2)
        maps.append(fm)
    return np.stack(maps, axis=0)   # (n_kernels, 4, 4)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LAYER 1 — Extract Composite Features
#     Input : 4x4x10
#     Conv  : F=10x40x2x2, S=2  ->  2x2x40
#     Pool  : F=2, S=2           ->  1x1x40
#
#     All 4 quadrants detected per channel = 10 x 4 = 40 features
# ══════════════════════════════════════════════════════════════════════════════

def get_layer1_kernels(n_channels):
    n_features = n_channels * 4
    F = np.zeros((n_channels, n_features, 2, 2), dtype=np.float32)
    for i in range(n_channels):
        F[i, i*4 + 0] = np.array([[1, 0], [0, 0]], dtype=np.float32)  # top-left
        F[i, i*4 + 1] = np.array([[0, 1], [0, 0]], dtype=np.float32)  # top-right
        F[i, i*4 + 2] = np.array([[0, 0], [1, 0]], dtype=np.float32)  # bottom-left
        F[i, i*4 + 3] = np.array([[0, 0], [0, 1]], dtype=np.float32)  # bottom-right
    return F  # (n_channels, n_features, 2, 2)


def layer1(basic_maps, kernels_l1):
    n_channels  = kernels_l1.shape[0]
    n_features  = kernels_l1.shape[1]
    composite = np.zeros((n_features, 2, 2), dtype=np.float32)
    for i in range(n_channels):
        for j in range(n_features):
            fm = conv2d(basic_maps[i], kernels_l1[i, j], stride=2, padding=0)
            if fm.shape == (2, 2):
                composite[j] += relu(fm)
            else:
                tmp = np.zeros((2, 2), dtype=np.float32)
                tmp[:fm.shape[0], :fm.shape[1]] = relu(fm)
                composite[j] += tmp
    features = np.zeros(n_features, dtype=np.float32)
    for j in range(n_features):
        features[j] = np.max(composite[j])
    return features  # (n_features,)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FEATURE EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(image, kernels_l0, kernels_l1):
    basic     = layer0(image, kernels_l0)    # (n_kernels, 4, 4)
    composite = layer1(basic, kernels_l1)    # (n_features,)
    return composite


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FULLY CONNECTED NEURAL NETWORK  with Dropout
#     40 -> 128 -> 10
#     Dropout applied to hidden layer during training only
# ══════════════════════════════════════════════════════════════════════════════

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def cross_entropy_loss(probs, label):
    return -np.log(probs[label] + 1e-9)


class FCNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr, dropout_rate=0.3):
        self.lr           = lr
        self.dropout_rate = dropout_rate
        np.random.seed(42)
        self.W1 = np.random.randn(hidden_size, input_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(output_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x, training=False):
        self.x   = x
        self.z1  = self.W1 @ x + self.b1
        self.a1  = np.maximum(0, self.z1)

        if training:
            # Inverted dropout: scale kept neurons so inference needs no adjustment
            self.mask = (np.random.rand(*self.a1.shape) > self.dropout_rate).astype(np.float32)
            self.a1   = self.a1 * self.mask / (1.0 - self.dropout_rate)
        else:
            self.mask = np.ones_like(self.a1)

        self.z2  = self.W2 @ self.a1 + self.b2
        self.out = softmax(self.z2)
        return self.out

    def backward(self, label):
        dz2 = self.out.copy()
        dz2[label] -= 1.0

        dW2 = np.outer(dz2, self.a1)
        db2 = dz2

        da1 = self.W2.T @ dz2
        da1 = da1 * self.mask / (1.0 - self.dropout_rate)
        dz1 = da1 * (self.z1 > 0).astype(np.float32)

        dW1 = np.outer(dz1, self.x)
        db1 = dz1

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, x):
        return np.argmax(self.forward(x, training=False))

    def save(self, path="model_weights.npz"):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"  Model saved to {path}")

    def load(self, path="model_weights.npz"):
        data    = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"  Model loaded from {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(X, y, nn):
    n       = len(X)
    indices = np.arange(n)

    for epoch in range(1, EPOCHS + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        correct    = 0

        for i in indices:
            probs       = nn.forward(X[i], training=True)
            total_loss += cross_entropy_loss(probs, y[i])
            correct    += (np.argmax(probs) == y[i])
            nn.backward(y[i])

        if epoch % 100 == 0 or epoch == 1:
            acc = correct / n * 100
            print(f"  Epoch {epoch:4d}/{EPOCHS}  "
                  f"Loss: {total_loss/n:.4f}  "
                  f"Train Acc: {acc:.1f}%")

    return nn


# ══════════════════════════════════════════════════════════════════════════════
# 9.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

SIGN_NAMES = {
    0: "All directions",
    1: "Closed Road",
    2: "To the Right",
    3: "To the Left",
    4: "Keep Straight",
    5: "Turn Right",
    6: "Turn Left",
    7: "U-Turn",
    8: "Pass to the right",
    9: "Pass to the left",
}


def evaluate(X, y, nn, split_name="Test"):
    correct = 0
    errors  = []
    for i in range(len(X)):
        pred = nn.predict(X[i])
        if pred == y[i]:
            correct += 1
        else:
            errors.append((y[i], pred))

    acc = correct / len(X) * 100
    print(f"\n  {split_name} Accuracy: {correct}/{len(X)} = {acc:.1f}%")

    if errors:
        print(f"  Misclassified samples ({len(errors)}):")
        for true, pred in errors[:10]:
            print(f"    True: {SIGN_NAMES[true]:22s}  Predicted: {SIGN_NAMES[pred]}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more.")

    return acc


def predict_single(image_path, nn, kernels_l0, kernels_l1, feat_max):
    """Predict a single image — useful for demo."""
    arr      = load_and_binarize(image_path)
    features = extract_features(arr, kernels_l0, kernels_l1)
    features = features / feat_max
    pred     = nn.predict(features)
    print(f"\n  Image  : {image_path}")
    print(f"  Result : Sign {pred} — {SIGN_NAMES[pred]}")
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Traffic Sign CNN  (v3 — Dropout + 40 features + 1520 imgs, 700 epochs)")
    print("=" * 60)

    kernels_l0 = get_fixed_kernels()                        # (10, 5, 5)
    kernels_l1 = get_layer1_kernels(len(kernels_l0))       # (10, 40, 2, 2)

    print("\n[1] Loading training data...")
    X_raw, y_train = load_dataset(TRAIN_DIR)
    print(f"    {len(X_raw)} images loaded.")

    print("\n[2] Extracting features (CNN layers 0 & 1)...")
    X_train = np.array([extract_features(img, kernels_l0, kernels_l1)
                        for img in X_raw], dtype=np.float32)
    print(f"    Feature vector size: {X_train.shape[1]}")

    feat_max = X_train.max(axis=0) + 1e-8
    X_train  = X_train / feat_max

    feature_size = X_train.shape[1]
    print(f"\n[3] Training FC Neural Network "
          f"({feature_size} -> {HIDDEN_SIZE} -> {NUM_CLASSES}, "
          f"{EPOCHS} epochs, dropout={DROPOUT_RATE})...")
    nn = FCNetwork(input_size=feature_size, hidden_size=HIDDEN_SIZE,
                   output_size=NUM_CLASSES, lr=LEARNING_RATE,
                   dropout_rate=DROPOUT_RATE)
    train(X_train, y_train, nn)

    print("\n[4] Evaluating on training set...")
    evaluate(X_train, y_train, nn, split_name="Train")

    if os.path.exists(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0:
        print("\n[5] Loading & evaluating on testing set...")
        X_test_raw, y_test = load_dataset(TEST_DIR)
        X_test = np.array([extract_features(img, kernels_l0, kernels_l1)
                           for img in X_test_raw], dtype=np.float32)
        X_test = X_test / feat_max
        evaluate(X_test, y_test, nn, split_name="Test")
    else:
        print("\n[5] No TestingImages folder found, skipping.")

    print("\n[6] Saving model...")
    nn.save("model_weights.npz")
    np.save("feat_max.npy", feat_max)

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
