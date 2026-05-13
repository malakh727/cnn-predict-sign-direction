# Plan: Interactive Draw-and-Classify Feature

## What it does

A GUI app where a user draws a 19×19 traffic sign freehand, and the model classifies it, showing:
- Predicted sign name
- Model confidence % (from softmax)
- Shape similarity % (Jaccard index vs. base_images/{pred}_0.png)
- Small reference image of the predicted sign

## File to create

**`draw_and_classify.py`** — new standalone file, no changes to cnn.py

## UI Layout

```
┌─────────────────────────────────────────────────────┐
│  Traffic Sign Classifier — Draw a Sign               │
├──────────────────────────┬──────────────────────────┤
│                          │  Prediction:              │
│   [475×475 grid canvas]  │  ► Turn Left              │
│                          │                           │
│   (left-click = draw)    │  Confidence:  98.3%       │
│   (right-click = erase)  │  Similarity:  61.4%       │
│                          │                           │
│                          │  Reference sign:          │
│                          │  [95×95 image]            │
├──────────────────────────┴──────────────────────────┤
│       [  Classify  ]        [  Clear  ]              │
└─────────────────────────────────────────────────────┘
```

## Pipeline (mirrors cnn.py exactly)

```
user drawing (19×19 binary array)
  → extract_features(arr, kernels_l0, kernels_l1)   → 40 features
  → features / feat_max                              → normalized
  → nn.forward(features, training=False)             → 10 probs
  → argmax → pred, probs[pred]*100 → confidence %
  → jaccard(arr, base_images/{pred}_0.png) → similarity %
```

## Metrics

- **Confidence**: softmax probability of the top class × 100
- **Similarity**: Jaccard index = |drawing ∩ base| / |drawing ∪ base| × 100
  (compares black pixel overlap only, ignores background)

## Dependencies

- tkinter (built-in)
- numpy, PIL (already used in cnn.py)
- Imports from cnn.py: extract_features, get_fixed_kernels, get_layer1_kernels, FCNetwork, SIGN_NAMES, IMG_SIZE, THRESHOLD

## Files used

| File | Role |
|------|------|
| `draw_and_classify.py` | new — full feature |
| `cnn.py` | import pipeline — no changes |
| `model_weights.npz` | loaded at startup |
| `feat_max.npy` | loaded at startup |
| `base_images/{0-9}_0.png` | Jaccard reference + display |
