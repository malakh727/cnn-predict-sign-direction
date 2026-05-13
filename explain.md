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

## Detailed Code Walkthrough

### Loading & Binarizing (`load_and_binarize`)

```python
def load_and_binarize(path):
    img = Image.open(path).convert("L").resize(IMG_SIZE, Image.NEAREST)
    arr = np.array(img, dtype=np.float32)
    return (arr < THRESHOLD).astype(np.float32)
```

- `.convert("L")` — converts any image (color, RGBA, etc.) to grayscale. Each pixel becomes a single number 0–255.
- `.resize(IMG_SIZE, Image.NEAREST)` — resizes to 19×19. `NEAREST` means "snap to the closest pixel" without blending — preserves sharp edges.
- `np.array(..., dtype=np.float32)` — turns the PIL image into a NumPy array of floats. Still 0–255 at this point.
- `(arr < THRESHOLD).astype(np.float32)` — pixels below 128 (dark ink) become **1.0**, pixels ≥ 128 (white background) become **0.0**. The result is a 19×19 array of 0s and 1s.

---

### Loading the Dataset (`load_dataset`)

```python
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
```

- Files are named `<sign_number>_<version>.png` — e.g., `3_17.png` is sign class 3, version 17.
- `fname.split('_')[0]` splits on underscore and takes the first part — the sign number.
- `int(...)` converts it to integer label. If a file has an unexpected name format the `ValueError` is caught and that file is skipped.
- `sorted(os.listdir(...))` — processes files in alphabetical order so the dataset is always loaded in the same order (reproducible).
- Returns `X` as a 3D float32 array of shape `(N, 19, 19)` and `y` as a 1D int32 array of shape `(N,)`.

---

### Fixed Kernels (`get_fixed_kernels`)

Each kernel is a 5×5 binary pattern. Convolution computes a dot product of the kernel with each 5×5 patch of the image. If the patch matches the kernel's pattern, the response is high; if it doesn't, the response is low.

```python
# K0: horizontal line — 5 ones in the middle row
K0 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],   ← fires when there is a horizontal line here
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
])
```

```python
return np.stack([K0, K1, ..., K9], axis=0)  # shape: (10, 5, 5)
```

`np.stack(..., axis=0)` combines the 10 separate 5×5 arrays into one 3D array of shape `(10, 5, 5)` — the first axis selects which kernel.

---

### Convolution (`conv2d`)

```python
def conv2d(image, kernel, stride=1, padding=0):
    H, W = image.shape
    F    = kernel.shape[0]
    H_out = (H - F) // stride + 1
    W_out = (W - F) // stride + 1
    shape   = (H_out, W_out, F, F)
    strides = (image.strides[0] * stride, image.strides[1] * stride,
               image.strides[0], image.strides[1])
    patches = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    return (patches * kernel).sum(axis=(2, 3)).astype(np.float32)
```

**What are strides?** In memory, a 2D array is stored row by row. `image.strides[0]` is how many bytes to jump to get to the next row; `image.strides[1]` is how many bytes to jump to get to the next column.

`as_strided` creates a 4D view `(H_out, W_out, F, F)` where `patches[r, c]` is the F×F patch starting at row `r*stride`, column `c*stride` — no data is copied. Then `(patches * kernel).sum(axis=(2,3))` multiplies each patch element-wise with the kernel and sums the result, giving one number per patch position. That is exactly convolution.

**Output size formula:** `(H - F) // stride + 1`
- 19×19 input, 5×5 kernel, stride=2 → `(19 - 5) // 2 + 1 = 8` → 8×8 output.

---

### ReLU

```python
def relu(x):
    return np.maximum(0, x)
```

Replaces every negative value with 0. If a kernel found no match in a patch, the dot product is zero or negative — ReLU zeroes it out. Only positive responses (real matches) pass through.

---

### Max Pooling (`maxpool2d`)

```python
def maxpool2d(fm, pool_size=2, stride=2):
    ...
    patches = np.lib.stride_tricks.as_strided(fm, shape=shape, strides=strides)
    return patches.max(axis=(2, 3)).astype(np.float32)
```

Same `as_strided` trick as convolution, but instead of a dot product we take `.max()` over each 2×2 patch. For an 8×8 input with pool=2, stride=2: output is 4×4. Each output cell holds the strongest response in its 2×2 region.

---

### Layer 0 (`layer0`)

```python
def layer0(image, kernels):
    maps = []
    for k in range(len(kernels)):
        fm = conv2d(image, kernels[k], stride=2, padding=0)  # 8×8
        fm = relu(fm)                                         # zeroes negatives
        fm = maxpool2d(fm, pool_size=2, stride=2)            # 4×4
        maps.append(fm)
    return np.stack(maps, axis=0)   # (10, 4, 4)
```

Runs each of the 10 kernels over the image independently and stacks the resulting 4×4 maps. The output `(10, 4, 4)` means: 10 feature channels, each 4×4.

---

### Layer 1 Kernels (`get_layer1_kernels`)

```python
def get_layer1_kernels(n_channels):
    n_features = n_channels * 4        # 10 × 4 = 40
    F = np.zeros((n_channels, n_features, 2, 2), dtype=np.float32)
    for i in range(n_channels):
        F[i, i*4 + 0] = [[1, 0], [0, 0]]   # top-left
        F[i, i*4 + 1] = [[0, 1], [0, 0]]   # top-right
        F[i, i*4 + 2] = [[0, 0], [1, 0]]   # bottom-left
        F[i, i*4 + 3] = [[0, 0], [0, 1]]   # bottom-right
    return F  # (10, 40, 2, 2)
```

Each channel `i` gets 4 kernels — one per quadrant. `F[i, i*4+0]` has a single `1` in the top-left corner, meaning "is channel i's feature active in the top-left of the 4×4 map?". Only kernel `F[i, j]` is non-zero for the pairing of channel `i` and feature `j` — all other channels have zero weight for that feature. This ensures each of the 40 output features corresponds to exactly one (channel, quadrant) pair.

---

### Layer 1 (`layer1`)

```python
def layer1(basic_maps, kernels_l1):
    composite = np.zeros((n_features, 2, 2), dtype=np.float32)
    for i in range(n_channels):
        for j in range(n_features):
            fm = conv2d(basic_maps[i], kernels_l1[i, j], stride=2)
            composite[j] += relu(fm)         # accumulate
    features = np.zeros(n_features, dtype=np.float32)
    for j in range(n_features):
        features[j] = np.max(composite[j])   # max pool → scalar
    return features   # (40,)
```

- `basic_maps[i]` is the 4×4 output of channel `i` from Layer 0.
- Convolving it with a 2×2 kernel at stride=2 gives a 2×2 output — each cell corresponds to one quadrant.
- After accumulating all channel contributions, `np.max(composite[j])` collapses the 2×2 to a single number — the maximum quadrant response for feature `j`.
- The final output is a flat 40-element vector: one number per (channel, quadrant) pair.

---

### Feature Normalization

```python
feat_max = X_train.max(axis=0) + 1e-8
X_train  = X_train / feat_max
```

- `X_train.max(axis=0)` — for each of the 40 features, finds the maximum value seen across all 1520 training images. Result is a vector of 40 values.
- `+ 1e-8` — tiny safety offset to prevent division by zero if a feature never activates.
- Dividing every image's features by this vector scales all 40 features to roughly [0, 1].
- **Why?** Without this, a feature with max response 5.0 and a feature with max response 0.3 would have very different scales. Gradient descent would move faster in the direction of large-scale features and barely update the small-scale ones, making training unstable.

---

### Weight Initialization

```python
self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
self.b1 = np.zeros(hidden_size, dtype=np.float32)
self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
self.b2 = np.zeros(output_size, dtype=np.float32)
```

- `np.random.randn(...)` generates random numbers from a standard normal distribution (mean 0, std 1).
- `* np.sqrt(2.0 / input_size)` — this is **He initialization**, designed for ReLU networks. It scales the initial weights so that the variance of the activations stays roughly the same through each layer. If weights start too large, activations explode; too small, activations vanish to zero and gradients can't flow.
- Biases start at 0 — they are just offsets and have no reason to be random.

---

### Forward Pass (`FCNetwork.forward`)

```python
def forward(self, x, training=False):
    self.x  = x                          # (40,) — save for backward
    self.z1 = self.W1 @ x + self.b1     # (128,) — linear transform
    self.a1 = np.maximum(0, self.z1)    # (128,) — ReLU

    if training:
        self.mask = (np.random.rand(*self.a1.shape) > self.dropout_rate).astype(np.float32)
        self.a1   = self.a1 * self.mask / (1.0 - self.dropout_rate)
    else:
        self.mask = np.ones_like(self.a1)

    self.z2  = self.W2 @ self.a1 + self.b2   # (10,) — output scores
    self.out = softmax(self.z2)               # (10,) — probabilities
    return self.out
```

- `W1 @ x` is matrix-vector multiplication: `(128×40) @ (40,)` → `(128,)`. Each hidden neuron computes a weighted sum of all 40 input features.
- `self.z1` and `self.x` are saved as instance variables because `backward()` needs them later.
- The **dropout mask** is a vector of 0s and 1s: `rand(...) > 0.2` is True (1.0) for 80% of neurons and False (0.0) for 20%. Multiplying `a1 * mask` zeroes the dropped neurons. Dividing by `0.8` scales up the survivors so the total expected magnitude stays the same.
- At inference (`training=False`) the mask is all ones — no dropout.

---

### Softmax

```python
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
```

- `x - np.max(x)` — subtracts the maximum value before exponentiating. This is a numerical stability trick: `exp(large_number)` can overflow to infinity, but `exp(0)` = 1 and all other values are ≤ 0, so no overflow.
- The result is still mathematically identical to softmax because subtracting a constant from all inputs doesn't change the ratios.
- Output sums to 1.0 — a valid probability distribution.

---

### Backward Pass (`FCNetwork.backward`)

```python
def backward(self, label):
    dz2 = self.out.copy()
    dz2[label] -= 1.0           # softmax + cross-entropy gradient combined
```

The gradient of (softmax + cross-entropy loss) w.r.t. the raw output scores `z2` is simply `probs - one_hot(label)`. If the true class is label=3 and `probs[3] = 0.9`, then `dz2[3] = 0.9 - 1.0 = -0.1` (needs to go up), and `dz2[j] = probs[j]` for all other j (needs to go down). This is the combined gradient of two operations, which simplifies to one line.

```python
    dW2 = np.outer(dz2, self.a1)   # outer product: (10,) × (128,) → (10×128)
    db2 = dz2                      # bias gradient is just the error signal

    da1 = self.W2.T @ dz2          # propagate error back through W2
    da1 = da1 * self.mask / (1.0 - self.dropout_rate)  # apply same dropout mask
    dz1 = da1 * (self.z1 > 0)     # ReLU gradient: zero out where z1 was ≤ 0

    dW1 = np.outer(dz1, self.x)   # gradient w.r.t. W1
    db1 = dz1
```

- `np.outer(a, b)` produces the outer product — each element of `a` multiplied by every element of `b`, resulting in a matrix. This is the gradient of a linear layer.
- `(self.z1 > 0)` is the ReLU gradient: 1 where the neuron was active, 0 where it was zeroed. Gradients don't flow through dead neurons.
- The dropout mask must be applied again in backward — dropped neurons had no effect on the output, so their gradients must also be zeroed.

```python
    self.W1 -= self.lr * dW1   # update: move weights opposite to gradient
    self.b1 -= self.lr * db1
    self.W2 -= self.lr * dW2
    self.b2 -= self.lr * db2
```

Each weight moves a small step (`lr = 0.005`) in the direction that reduces the loss.

---

### Training Loop (`train`)

```python
def train(X, y, nn):
    indices = np.arange(n)
    for epoch in range(1, EPOCHS + 1):
        np.random.shuffle(indices)          # new random order each epoch
        for i in indices:
            probs       = nn.forward(X[i], training=True)
            total_loss += cross_entropy_loss(probs, y[i])
            correct    += (np.argmax(probs) == y[i])
            nn.backward(y[i])               # update weights immediately
```

This is **online (stochastic) gradient descent** — weights are updated after every single image, not after a batch. `np.argmax(probs)` returns the index of the highest probability, which is the predicted class.

---

### Evaluation (`evaluate`)

```python
def evaluate(X, y, nn, split_name="Test"):
    for i in range(len(X)):
        pred = nn.predict(X[i])   # calls forward(training=False)
        if pred == y[i]:
            correct += 1
        else:
            errors.append((y[i], pred))
```

`nn.predict` calls `forward` with `training=False` — no dropout, deterministic output. Misclassified samples are collected and printed so you can see which signs confused the model.

---

### Main Flow (`main`)

The function branches on `TEST_MODE`:

**Test Mode (`TEST_MODE = True`):**
```
1. get_fixed_kernels() + get_layer1_kernels()   → build the two CNN layers
2. np.load('parameter.npz')                     → restore W1, b1, W2, b2
3. np.load('feat_max.npy')                      → restore normalization scale
4. load_test_dataset(TEST_DIR)                  → images + labels + filenames
5. extract_features(...)  / feat_max            → normalized 40-dim vectors
6. for each image: cnn.predict(...)             → Passed / Failed per image
7. print totals                                 → Passed X  /  Failed Y
```

**Train Mode (`TEST_MODE = False`):**
```
1. get_fixed_kernels() + get_layer1_kernels()   → build the two CNN layers
2. load_dataset(TRAIN_DIR)                      → raw 19×19 images
3. extract_features(...)                        → CNN pipeline → 40-dim vectors
4. feat_max normalization                       → scale all features to [0,1]
5. FCNetwork(40→128→10)                         → create network with He init
6. train(...)                                   → 700 epochs of gradient descent
7. evaluate(train set)                          → check training accuracy
8. evaluate(test set)                           → check generalization
9. np.savez('parameter.npz', W1, b1, W2, b2)   → save weights
10. np.save('feat_max.npy', feat_max)           → save normalization scale
```

Note: `feat_max` must be saved and loaded together with the weights. At inference, new images must be normalized using the same scale factors that were computed from the training set — using different scales would shift the feature values the network was trained on.

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

## Presentation Q&A

### Data Types

**Q: Why do we use `float32` everywhere? Why not `int` or `bool`?**

- **Why not `bool`?**
  Boolean (True/False) only works for the raw binarized image where pixels are 0 or 1.
  After convolution the output is a dot product — e.g., the horizontal kernel K0 has 5 ones, so the maximum response is 5.0.
  That *magnitude* carries critical information: a strong response means a clear feature match, a weak one means a partial match.
  If we used bool we would collapse every non-zero response to `True`, losing all magnitude information.

- **Why not `int`?**
  After feature normalization (`feat / feat_max`) the values become fractions like 0.73.
  Integers cannot represent fractions — every fractional value would be truncated to 0.
  Also, the weight matrices W1 and W2 are initialized with small values like 0.04231...
  and gradient updates (lr × gradient ≈ 0.000…) would all be rounded to 0, so the network could never learn.

- **Why `float32` and not `float64` (NumPy's default)?**
  `float32` uses 4 bytes per number vs. `float64`'s 8 bytes — half the memory for every array
  (images, feature maps, weight matrices, gradients).
  7 decimal digits of precision (float32) is more than enough for training a small network.
  It is also slightly faster on most CPUs.
  The explicit `dtype=np.float32` casts throughout the code are intentional to keep everything consistent.

---

### Architecture Questions

**Q: Why are the convolutional kernels fixed and not learned?**

The lecture architecture separates concerns: the fixed kernels handle low-level geometry (lines, edges, corners) using human knowledge, and the fully connected network learns which *combinations* of those geometric features identify each sign. This also makes the model faster to train because there are fewer parameters to update.

**Q: What would happen if you removed Layer 1 and fed the 4×4×10 output directly to the FC network?**

The FC network would receive 160 values (4×4×10) instead of 40. More importantly, it would lose spatial awareness — it wouldn't know *where* in the image each feature appeared. Layer 1's quadrant detectors compress spatial position into those 40 values. Without it, two signs that share the same features but in different positions could look identical.

**Q: Why does Layer 0 use stride=2 in the convolution?**

Stride=2 moves the kernel two pixels at a time instead of one. On a 19×19 input with a 5×5 kernel this produces an 8×8 output without needing zero-padding. It simultaneously reduces spatial size and computation — fewer output positions to compute.

**Q: What does max pooling do and why use it?**

Max pooling takes the *maximum* value in each 2×2 region. It keeps the strongest feature response while discarding its exact position within that region. This provides a small amount of translation invariance — a feature detected 1 pixel to the left or right still produces the same pooled output.

**Q: Why 40 features specifically?**

Layer 1 has 40 kernels = 10 channels × 4 quadrants. Each of the 10 Layer-0 feature channels is examined in 4 spatial regions (top-left, top-right, bottom-left, bottom-right). After the final max pool each becomes a single number, giving exactly 40 features.

---

### Training Questions

**Q: What is inverted dropout and why use it?**

Standard dropout zeroes out p% of neurons during training and then scales *down* all outputs by (1-p) at inference to match the expected magnitude. Inverted dropout instead scales *up* the active neurons by 1/(1-p) during training, so inference requires no adjustment at all. It is simpler and more common in practice.

**Q: What is an epoch and why do we need multiple epochs?**

One epoch = one full pass through all 1520 training images. After a single pass the weights have only seen each image once — the updates are noisy and the network hasn't had enough exposure to learn reliable patterns. Running multiple epochs lets the network revisit every image many times, gradually refining the weights with each pass until the loss stops improving. Think of it like studying: reading your notes once is rarely enough; reviewing them repeatedly is what makes the knowledge stick.

Without multiple epochs the network would be severely undertrained. With too many epochs it would start memorizing the training data (overfitting). 700 epochs was chosen because the loss converges and training accuracy reaches ~99.5% without signs of overfitting on the test set.

**Q: Why shuffle the training data each epoch?**

If the network always sees images in the same order (all sign-0 first, then all sign-1, etc.) it can develop an ordering bias — the gradients from the last few batches have a disproportionate effect. Shuffling ensures every epoch presents a different order, which stabilizes training.

**Q: Why learning rate 0.005?**

It is a balance: too large and gradient descent overshoots minima (loss oscillates); too small and training is extremely slow or gets stuck in local minima. 0.005 was empirically found to converge reliably within 700 epochs on this dataset.

**Q: What is cross-entropy loss?**

Given the softmax output probabilities and the true class label, cross-entropy is: `loss = -log(probability of correct class)`. If the model is very confident and correct the loss is near 0. If it is confident but wrong the loss is very large. It is the standard loss for multi-class classification.

**Q: How does backpropagation work here?**

Starting from the loss, we compute how much each weight contributed to the error using the chain rule:
1. `dL/dW2` — gradient of loss w.r.t. output layer weights
2. `dL/db2` — gradient of loss w.r.t. output biases
3. `dL/dW1`, `dL/db1` — propagated back through ReLU and the hidden layer
4. Each weight is updated: `W -= lr * gradient`

The convolutional layers are fixed, so gradients stop at the 40-feature vector.

---

### Dataset Questions

**Q: Why generate 152 images per sign instead of using the original 10?**

The original dataset has 1 image per sign (10 total). A neural network trained on 10 examples would memorize them exactly and fail on any real-world variation (slight tilt, noise, different lighting scan). 152 augmented versions per sign = 1520 total teaches the network to be robust to the exact types of variation the test data might contain.

**Q: Why was blur augmentation removed?**

At 19×19 pixels the signs are already very small. A 3×3 blur smears thin strokes (like the arrow shaft in turn-left/right signs) until they disappear or merge. This creates training images that look nothing like valid signs, confusing the network. The training accuracy remained the same but generalization suffered.

**Q: Why not use scales smaller than 0.90?**

At 0.70× scale a 19×19 image shrinks the sign to about 13×13 pixels, then it is centered on a white canvas. Most of the image becomes blank background and the sign detail is lost. These images are too degraded to be useful training examples — they hurt more than they help.

---

### Code Questions

**Q: What does `as_strided` do and why is it faster than loops?**

`np.lib.stride_tricks.as_strided` creates a *view* of an array with custom strides — it rearranges how NumPy interprets the memory without copying any data. This lets us extract all convolution patches at once as a 5D array and compute the entire convolution with a single `np.tensordot` call. The naive approach uses 4 nested Python loops (each one slow), while `as_strided` lets NumPy's C backend do all the work in one operation.

**Q: What does `feat_max` normalization do, and what if a feature is always zero?**

`feat_max[j]` is the maximum value of feature j across all training images. Dividing by it scales every feature to [0, 1]. If a kernel never activates on any training image its max would be 0, causing division-by-zero. The code handles this: `feat_max[feat_max == 0] = 1` replaces zeros with 1 so those features remain 0 after division (harmless).

**Q: What is softmax and why use it on the output layer?**

Softmax converts raw output scores (logits) into probabilities that sum to 1:
`softmax(z_i) = exp(z_i) / sum(exp(z_j))`.
The class with the highest score gets the highest probability. Using probabilities instead of raw scores allows cross-entropy loss to be computed and gives interpretable confidence values.

**Q: Could you change the hidden layer size from 128 to something else?**

Yes. Larger (e.g., 256) could capture more complex relationships but risks overfitting and is slower to train. Smaller (e.g., 64) trains faster but might underfit. 128 was chosen as a reasonable middle ground for a 40-input, 10-class problem. With 1520 training images and only 40 input features, 128 is already generous.

---

## Running the Code

The file has a single flag near the top that controls which mode runs:

```python
TEST_MODE = True   # True = test only (no training)  |  False = train + save
```

### Test Mode (`TEST_MODE = True`) — submitted version

```bash
python3 cnn.py
```

1. Loads saved weights from `parameter.npz` and normalization from `feat_max.npy`
2. Loads test images from `TEST_DIR` (default `../TestingImages/`)
3. Extracts features and runs inference — no training at all
4. Prints one line per image in the professor's format:
   ```
   Passed: Image Name = 0_0.png
   Failed: Image Name = 2_3.png
   ...
   Total of Passed Images = 199
   Total of Failed Images = 1
   ```

### Train Mode (`TEST_MODE = False`) — to retrain

```bash
python3 cnn.py
```

1. Loads training images from `./dataset/`
2. Extracts 40 features per image through the CNN layers
3. Trains the FC network for 700 epochs
4. Evaluates on training set and test set
5. Saves weights to `parameter.npz` and normalization to `feat_max.npy`

---

## Results Summary

| Metric                     | Value               |
| -------------------------- | ------------------- |
| Train Accuracy             | 99.5%               |
| Test Accuracy (simulation) | 99.5%               |
| Training images            | 1520 (152 per sign) |
| Feature vector size        | 40                  |
| Network                    | 40 → 128 → 10       |
