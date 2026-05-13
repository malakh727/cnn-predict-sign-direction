# How to Run the Project

## 1. Train the Model (only needed once)

```bash
python3 cnn.py
```

- Trains on `./dataset/` (1520 images)
- Saves `model_weights.npz` and `feat_max.npy` when done
- Takes ~1–2 minutes

---

## 2. Interactive Drawing App

```bash
python3 draw_and_classify.py
```

**Requirements:** `python3-tk` must be installed.
If not installed, run once:
```bash
sudo apt-get install -y python3-tk
```

**How to use:**
- Left-click or drag → draw black pixels
- Right-click or drag → erase pixels
- Click **Classify** → see the predicted sign, confidence %, and similarity %
- Click **Clear** → reset the canvas

---

## 3. Generate Dataset (only if dataset is missing)

```bash
python3 generate_dataset.py
```

Reads `./base_images/0_0.png … 9_0.png` and writes 152 augmented images
per sign (1520 total) into `./dataset/`.

---

## 4. Test on Professor's Test Set

Place test images in `../TestingImages/` then run:

```bash
python3 cnn.py
```

The model automatically evaluates `../TestingImages/` after training and
prints per-class accuracy.
