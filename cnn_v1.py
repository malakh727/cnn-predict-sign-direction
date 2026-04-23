"""
Traffic Sign CNN - Matches professor's architecture from lecture (slide 36)
Architecture:
  Input     : 19x19x1
  Layer 0   : Conv (F=5x5x4, S=2, P=0) -> ReLU -> MaxPool (F=2,S=2) -> 4x4x4
  Layer 1   : Conv (F=4x10x2x2, S=2, P=0) -> ReLU -> MaxPool (F=2,S=2) -> 1x1x10
  Layer 2   : Fully Connected NN -> 10 outputs (one per sign)

Fixed kernels (slide 39):
  K0: horizontal line detector
  K1: vertical line detector
  K2: slope=-1 diagonal detector
  K3: slope=+1 diagonal detector

Only uses NumPy and PIL as required by the assignment.
"""

import os
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR    = "./dataset"
TEST_DIR     = "./testing_dataset"
BASE_DIR     = "./base_images"
IMG_SIZE     = (19, 19)
NUM_CLASSES  = 10
THRESHOLD    = 128

# FC NN hyperparameters
HIDDEN_SIZE  = 64
LEARNING_RATE= 0.01
EPOCHS       = 500
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 1.  IMAGE LOADING & BINARIZATION
# ══════════════════════════════════════════════════════════════════════════════

def load_and_binarize(path):
    """Load PNG, convert to grayscale, resize to 19x19, binarize (1=sign pixel)."""
    img = Image.open(path).convert("L").resize(IMG_SIZE, Image.NEAREST)
    arr = np.array(img, dtype=np.float32)
    # white background=255 -> 0, black sign=0 -> 1
    arr = (arr < THRESHOLD).astype(np.float32)
    return arr  # shape: (19, 19)


def load_dataset(directory):
    """
    Load all images from directory.
    Filename format: <label>_<version>.png
    Returns X (N, 19, 19), y (N,)
    """
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
# 2.  FIXED KERNELS  (slide 39)
# ══════════════════════════════════════════════════════════════════════════════

def get_fixed_kernels():
    """
    4 fixed 5x5 kernels as shown in lecture slide 39.
    K0 : horizontal line  (middle row = 1)
    K1 : vertical line    (middle col = 1)
    K2 : slope=-1 diagonal (top-left to bottom-right)
    K3 : slope=+1 diagonal (bottom-left to top-right)
    """
    K0 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    K1 = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.float32)

    K2 = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=np.float32)

    K3 = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ], dtype=np.float32)

    return np.stack([K0, K1, K2, K3], axis=0)  # shape: (4, 5, 5)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CONVOLUTION & POOLING  (NumPy only)
# ══════════════════════════════════════════════════════════════════════════════

def conv2d(image, kernel, stride=1, padding=0):
    """
    2D convolution of a single-channel image with a single kernel.
    image  : (H, W)
    kernel : (F, F)
    Returns feature map (H_out, W_out)
    """
    H, W = image.shape
    F    = kernel.shape[0]

    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
        H += 2 * padding
        W += 2 * padding

    H_out = (H - F) // stride + 1
    W_out = (W - F) // stride + 1
    out   = np.zeros((H_out, W_out), dtype=np.float32)

    for i in range(H_out):
        for j in range(W_out):
            patch    = image[i*stride : i*stride+F, j*stride : j*stride+F]
            out[i,j] = np.sum(patch * kernel)

    return out


def relu(x):
    return np.maximum(0, x)


def maxpool2d(feature_map, pool_size=2, stride=2):
    """
    Max pooling on a single feature map (H, W).
    """
    H, W  = feature_map.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out   = np.zeros((H_out, W_out), dtype=np.float32)

    for i in range(H_out):
        for j in range(W_out):
            patch    = feature_map[i*stride : i*stride+pool_size,
                                   j*stride : j*stride+pool_size]
            out[i,j] = np.max(patch)

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LAYER 0 — Extract Basic Features  (slide 39)
#     Input : 19x19x1
#     Conv  : F=5x5x4, S=2, P=0  ->  8x8x4
#     Pool  : F=2, S=2            ->  4x4x4
# ══════════════════════════════════════════════════════════════════════════════

def layer0(image, kernels):
    """
    image   : (19, 19)
    kernels : (4, 5, 5)
    Returns : (4, 4, 4)  — 4 feature maps of size 4x4
    """
    maps = []
    for k in range(4):
        fm   = conv2d(image, kernels[k], stride=2, padding=0)  # -> 8x8
        fm   = relu(fm)
        fm   = maxpool2d(fm, pool_size=2, stride=2)             # -> 4x4
        maps.append(fm)
    return np.stack(maps, axis=0)   # (4, 4, 4)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LAYER 1 — Extract Composite Features  (slide 40-41)
#     Input : 4x4x4
#     Conv  : F=4x10x2x2, S=2, P=0  ->  2x2x10
#     Pool  : F=2, S=2               ->  1x1x10
#
#     The layer-1 kernels are FIXED 2x2 detectors.
#     Each of the 10 output channels is produced by combining all 4 input maps.
#     F[i,j] is the 2x2 filter applied to input channel i, contributing to output j.
# ══════════════════════════════════════════════════════════════════════════════

def get_layer1_kernels():
    """
    Returns fixed 2x2 kernels for layer 1.
    Shape: (4, 10, 2, 2) — F[input_channel, output_channel, row, col]

    These kernels detect spatial positions of basic features:
      output 0  : horizontal line top
      output 1  : horizontal line bottom
      output 2  : vertical line right
      output 3  : vertical line left
      output 4  : slope=+1 full line
      output 5  : slope=-1 full line
      output 6  : slope=-1 half-line (1)
      output 7  : slope=-1 half-line (2)
      output 8  : slope=+1 half-line (1)
      output 9  : slope=+1 half-line (2)
    """
    F = np.zeros((4, 10, 2, 2), dtype=np.float32)

    # Output 0: horizontal line present in top half (from horizontal feature map)
    F[0, 0] = np.array([[1, 1], [0, 0]])
    # Output 1: horizontal line present in bottom half
    F[0, 1] = np.array([[0, 0], [1, 1]])
    # Output 2: vertical line present on right side
    F[1, 2] = np.array([[0, 1], [0, 1]])
    # Output 3: vertical line present on left side
    F[1, 3] = np.array([[1, 0], [1, 0]])
    # Output 4: slope=+1 full diagonal (from slope+1 map)
    F[3, 4] = np.array([[1, 0], [0, 1]])
    # Output 5: slope=-1 full diagonal (from slope-1 map)
    F[2, 5] = np.array([[0, 1], [1, 0]])
    # Output 6: slope=-1 upper-left region
    F[2, 6] = np.array([[1, 0], [0, 0]])
    # Output 7: slope=-1 lower-right region
    F[2, 7] = np.array([[0, 0], [0, 1]])
    # Output 8: slope=+1 upper-right region
    F[3, 8] = np.array([[0, 1], [0, 0]])
    # Output 9: slope=+1 lower-left region
    F[3, 9] = np.array([[0, 0], [1, 0]])

    return F  # (4, 10, 2, 2)


def layer1(basic_maps, kernels_l1):
    """
    basic_maps  : (4, 4, 4) from layer0
    kernels_l1  : (4, 10, 2, 2)
    Returns     : (10,)  — 10 composite feature scalars after pool
    """
    # Step 1: for each output channel j, sum conv results from all 4 input channels
    composite = np.zeros((10, 2, 2), dtype=np.float32)

    for i in range(4):       # input channels
        for j in range(10):  # output channels
            fm = conv2d(basic_maps[i], kernels_l1[i, j], stride=2, padding=0)  # -> 2x2 or 1x1
            # If input is 4x4 and kernel is 2x2 with stride 2 -> output is 2x2
            if fm.shape == (2, 2):
                composite[j] += relu(fm)
            else:
                # pad to 2x2 if needed
                tmp = np.zeros((2, 2), dtype=np.float32)
                tmp[:fm.shape[0], :fm.shape[1]] = relu(fm)
                composite[j] += tmp

    # Step 2: MaxPool each 2x2 composite map -> scalar
    features = np.zeros(10, dtype=np.float32)
    for j in range(10):
        features[j] = np.max(composite[j])  # 2x2 -> scalar (maxpool F=2,S=2)

    return features  # (10,)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FEATURE EXTRACTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(image, kernels_l0, kernels_l1):
    """Full pipeline: 19x19 image -> 10-dim feature vector."""
    basic      = layer0(image, kernels_l0)    # (4, 4, 4)
    composite  = layer1(basic, kernels_l1)    # (10,)
    return composite


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FULLY CONNECTED NEURAL NETWORK  (from scratch, NumPy only)
#     Architecture: 10 -> HIDDEN_SIZE -> 10
#     Activation  : ReLU (hidden), Softmax (output)
#     Loss        : Cross-entropy
#     Optimizer   : SGD with backpropagation
# ══════════════════════════════════════════════════════════════════════════════

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def cross_entropy_loss(probs, label):
    return -np.log(probs[label] + 1e-9)


class FCNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.lr = lr
        np.random.seed(42)
        scale = np.sqrt(2.0 / input_size)
        self.W1 = np.random.randn(hidden_size, input_size).astype(np.float32) * scale
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(output_size, hidden_size).astype(np.float32) * (np.sqrt(2.0/hidden_size))
        self.b2 = np.zeros(output_size, dtype=np.float32)

    def forward(self, x):
        """x: (input_size,) -> returns probs (output_size,)"""
        self.x   = x
        self.z1  = self.W1 @ x + self.b1         # hidden pre-activation
        self.a1  = np.maximum(0, self.z1)         # ReLU
        self.z2  = self.W2 @ self.a1 + self.b2   # output pre-activation
        self.out = softmax(self.z2)               # probabilities
        return self.out

    def backward(self, label):
        """Backprop for a single sample."""
        # Output layer gradient
        dz2      = self.out.copy()
        dz2[label] -= 1.0                         # softmax + cross-entropy gradient

        dW2      = np.outer(dz2, self.a1)
        db2      = dz2

        # Hidden layer gradient
        da1      = self.W2.T @ dz2
        dz1      = da1 * (self.z1 > 0).astype(np.float32)  # ReLU derivative

        dW1      = np.outer(dz1, self.x)
        db1      = dz1

        # Update weights
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, x):
        probs = self.forward(x)
        return np.argmax(probs)

    def save(self, path="model_weights.npz"):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"  Model saved to {path}")

    def load(self, path="model_weights.npz"):
        data     = np.load(path)
        self.W1  = data['W1']
        self.b1  = data['b1']
        self.W2  = data['W2']
        self.b2  = data['b2']
        print(f"  Model loaded from {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(X, y, nn):
    n = len(X)
    indices = np.arange(n)

    for epoch in range(1, EPOCHS + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        correct    = 0

        for i in indices:
            probs = nn.forward(X[i])
            total_loss += cross_entropy_loss(probs, y[i])
            correct    += (np.argmax(probs) == y[i])
            nn.backward(y[i])

        if epoch % 50 == 0 or epoch == 1:
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
        for true, pred in errors[:10]:  # show first 10 only
            print(f"    True: {SIGN_NAMES[true]:20s}  Predicted: {SIGN_NAMES[pred]}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more.")

    return acc


def predict_single(image_path, nn, kernels_l0, kernels_l1):
    """Predict a single image file."""
    arr      = load_and_binarize(image_path)
    features = extract_features(arr, kernels_l0, kernels_l1)
    pred     = nn.predict(features)
    print(f"\n  Image  : {image_path}")
    print(f"  Result : Sign {pred} — {SIGN_NAMES[pred]}")
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Traffic Sign CNN")
    print("=" * 60)

    # ── Fixed kernels ─────────────────────────────────────────
    kernels_l0 = get_fixed_kernels()       # (4, 5, 5)
    kernels_l1 = get_layer1_kernels()      # (4, 10, 2, 2)

    # ── Load training data ────────────────────────────────────
    print("\n[1] Loading training data...")
    X_raw, y_train = load_dataset(TRAIN_DIR)
    print(f"    {len(X_raw)} images loaded.")

    # ── Extract features ──────────────────────────────────────
    print("\n[2] Extracting features (CNN layers 0 & 1)...")
    X_train = np.array([extract_features(img, kernels_l0, kernels_l1)
                        for img in X_raw], dtype=np.float32)
    print(f"    Feature vector size: {X_train.shape[1]}")

    # Normalize features to [0,1]
    feat_max = X_train.max(axis=0) + 1e-8
    X_train  = X_train / feat_max

    # ── Build & train FC NN ───────────────────────────────────
    print(f"\n[3] Training FC Neural Network ({EPOCHS} epochs)...")
    nn = FCNetwork(input_size=10, hidden_size=HIDDEN_SIZE,
                   output_size=NUM_CLASSES, lr=LEARNING_RATE)
    train(X_train, y_train, nn)

    # ── Evaluate on training set ──────────────────────────────
    print("\n[4] Evaluating on training set...")
    evaluate(X_train, y_train, nn, split_name="Train")

    # ── Evaluate on testing set ───────────────────────────────
    if os.path.exists(TEST_DIR) and len(os.listdir(TEST_DIR)) > 0:
        print("\n[5] Loading & evaluating on testing set...")
        X_test_raw, y_test = load_dataset(TEST_DIR)
        X_test = np.array([extract_features(img, kernels_l0, kernels_l1)
                           for img in X_test_raw], dtype=np.float32)
        X_test = X_test / feat_max   # same normalization as training
        evaluate(X_test, y_test, nn, split_name="Test")
    else:
        print("\n[5] No testing_dataset found, skipping test evaluation.")

    # ── Save model ────────────────────────────────────────────
    print("\n[6] Saving model...")
    nn.save("model_weights.npz")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
