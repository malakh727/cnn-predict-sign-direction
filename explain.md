# CNN Traffic Sign Classifier — Code Explanation

## Big Picture

The program recognizes 10 traffic signs from 19×19 black-and-white images.
It follows the architecture from the lecture: two fixed convolutional layers extract features,
then a fully connected neural network classifies them.

**Pipeline:**

```
19×19 image → Layer 0 (Conv+Pool) → Layer 1 (Conv+Pool) → 40 features → FC Network → sign class (0–9)
```

---

## Step 1 — Loading & Binarizing Images

```python
def load_and_binarize(path):
```

- Opens the image, converts to grayscale, resizes to 19×19.
- Any pixel darker than 128 becomes **1** (foreground/ink), the rest become **0** (background).
- Result: a 19×19 NumPy array of 0s and 1s.

---

## Step 2 — Fixed Kernels (Layer 0)

We use **10 hand-designed 5×5 kernels** — they do NOT change during training (fixed by design).

| Kernel | Detects                                  |
| ------ | ---------------------------------------- |
| K0     | Horizontal line                          |
| K1     | Vertical line                            |
| K2     | Diagonal (top-left → bottom-right)       |
| K3     | Diagonal (top-right → bottom-left)       |
| K4     | Top-half filled (top edge)               |
| K5     | Bottom-half filled (bottom edge)         |
| K6     | Left-half filled (left edge)             |
| K7     | Right-half filled (right edge)           |
| K8     | Bottom-right L-shape (curved arrow bend) |
| K9     | Bottom-left L-shape (curved arrow bend)  |

**Why fixed?** The lecture approach uses expert-designed kernels for basic geometric features. The neural network then learns which combinations of these features correspond to each sign.

---

## Step 3 — Layer 0: Basic Feature Extraction

```
Input:  19×19×1
Conv:   10 kernels, 5×5, stride=2  →  8×8×10
ReLU:   keep only positive responses
Pool:   2×2 max pool, stride=2     →  4×4×10
```

- **Stride=2** in convolution shrinks 19→8 without needing padding.
- **ReLU** sets negative values to zero (kernel didn't match that region).
- **Max pooling** keeps the strongest response in each 2×2 region.
- Output: 10 feature maps, each 4×4.

---

## Step 4 — Layer 1: Composite Feature Extraction

```
Input:  4×4×10
Conv:   40 kernels (2×2, stride=2)  →  2×2×40
ReLU
Pool:   2×2 max pool                →  1×1×40
Output: 40-dimensional feature vector
```

Each of the 40 kernels is a **quadrant detector** — it checks whether a basic feature (from Layer 0) is active in the top-left, top-right, bottom-left, or bottom-right of the 4×4 map.

**Why 40?** 10 channels × 4 quadrants = 40. This tells the network _where_ in the image each basic feature appears, not just _whether_ it appears.

The final max pool collapses each 2×2 map to a single number → **40 features total**.

---

## Step 5 — Fully Connected Neural Network

```
Architecture: 40 → 128 → 10
```

- **Input layer**: the 40 features from Layer 1
- **Hidden layer**: 128 neurons with ReLU activation
- **Output layer**: 10 neurons (one per sign) with Softmax → probabilities

### Dropout (training only)

During training, 20% of hidden neurons are randomly disabled each forward pass.
This prevents the network from memorizing specific training images (overfitting).
We use **inverted dropout**: disabled neurons are zeroed and active ones are scaled up by 1/0.8,
so the output scale stays the same at inference time (no adjustment needed).

### Loss Function

Cross-entropy loss: measures how far the predicted probabilities are from the true label.
Lower is better.

### Backpropagation

Computes gradients of the loss with respect to W1, b1, W2, b2, then updates them using
gradient descent with learning rate 0.005.

---

## Step 6 — Training Loop

```python
for epoch in range(1, EPOCHS + 1):
    shuffle(indices)
    for each image:
        forward pass (with dropout)
        compute loss
        backward pass (update weights)
```

- **700 epochs**, shuffled each time to prevent order bias.
- Prints loss and accuracy every 100 epochs to track progress.

---

## Step 7 — Dataset Augmentation

The base dataset has only 1 image per sign (10 total).
`generate_dataset.py` creates **152 augmented versions per sign = 1520 total** using:

| Augmentation                | Purpose                     |
| --------------------------- | --------------------------- |
| Rotations (±3° to ±25°)     | Handle tilted signs         |
| Shifts (1–4 px)             | Handle off-center signs     |
| Scales (0.90–1.20)          | Handle different sizes      |
| Noise (2–8% pixels flipped) | Handle scanning artifacts   |
| Salt & pepper noise         | Handle damaged signs        |
| Thickening                  | Handle bold/thick strokes   |
| Combinations of the above   | Handle real-world variation |

---

## Key Design Decisions

**Why not use libraries like TensorFlow/PyTorch?**
Assignment requires implementing the algorithm from scratch using only NumPy and PIL.

**Why fixed kernels instead of learned ones?**
The lecture architecture uses fixed kernels for the convolutional layers. The learning happens entirely in the FC network. This also keeps the model fast and interpretable.

**Why vectorized convolution (`as_strided`)?**
The naive approach uses nested Python for-loops (slow). `np.lib.stride_tricks.as_strided` extracts all patches at once as a NumPy view — no Python loops, much faster.

**Why normalize features (`feat_max`)?**
Feature values can vary widely depending on which kernels activate. Dividing by the max of each feature across the training set puts all 40 features on the same scale [0,1], which helps gradient descent converge.

---

## Running the Code

```bash
python3 cnn.py
```

1. Loads training images from `./dataset/`
2. Extracts 40 features per image through the CNN layers
3. Trains the FC network for 700 epochs
4. Evaluates on training set
5. If `../TestingImages/` exists → evaluates on test set
6. Saves weights to `model_weights.npz`

---

## Results Summary

| Metric                     | Value               |
| -------------------------- | ------------------- |
| Train Accuracy             | 99.5%               |
| Test Accuracy (simulation) | 99.5%               |
| Training images            | 1520 (152 per sign) |
| Feature vector size        | 40                  |
| Network                    | 40 → 128 → 10       |
