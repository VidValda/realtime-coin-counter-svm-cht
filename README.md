# Coin Counter C++

C++ coin detection pipeline (OpenCV): live detection and tracking, calibration, and classifier training. Requires OpenCV 4 (core, imgproc, videoio, highgui, ml).

**Build**

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

**Build with LibTorch (CNN/ResNet classifiers)**  
From the project root, either:

- **Automatic (downloads LibTorch CPU into `build/libtorch`):**
  ```bash
  ./build_with_torch.sh
  ```
- **Using an existing LibTorch** (e.g. from [pytorch.org](https://pytorch.org/get-started/locally/) → C++/LibTorch):
  ```bash
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
  make
  ```
  If your project path contains spaces, use a path without spaces for LibTorch (e.g. `-DCMAKE_PREFIX_PATH=/tmp/libtorch` with a symlink to the real path).

Then run `python export_torchscript.py` from the project root to generate `coin_cnn_traced.pt` and `coin_resnet18_traced.pt`. In `coin_counter`, keys `5`/`6` switch to CNN/ResNet18.

**If CNN/ResNet fail to load** with "maximum supported version for reading is 1": your LibTorch is too old. Delete `build/libtorch` and run `./build_with_torch.sh` again (it downloads LibTorch 2.5.1 by default).

---

## Executables

Run from `build/`; press `q` to quit unless noted.

- **coin_counter** — Live coin detection and tracking. Uses calibration and a classifier (SVM, KNN, RandomForest, NaiveBayes). Keys `1`–`4` switch classifier; shows total EUR and optional debug histogram.
- **train_acquisition** — Capture labeled crops by zone (6 euro classes, randomized). Writes images under `training_data/` and appends to the training manifest. Keys: `c` capture, `q` quit.
- **train_svm** — Train SVM from the training manifest, save model and scaler. Keys: `s` train and save, `q` quit.
- **camera_calibration** — Capture diameter samples by zone, fit radial and tilt distortion, save calibration. Keys: `c` capture, `s` solve and save, `q` quit.

---

## Data files

- **coin_calibration_robust.yaml** — From `camera_calibration` (base ratio, radial k, tilt p_y, p_x).
- **coin_svm.yaml**, **coin_scaler.yaml** — From `train_svm` (model and 4-feature scaler).
- **classifier_default.txt** — Optional; single digit 0–3 for default classifier index.
- **training_data/** — Images and manifest produced by `train_acquisition`, consumed by `train_svm`.

All paths are relative to the current working directory.
