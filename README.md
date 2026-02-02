# Coin Counter C++

Production C++ port of the Python coin detection pipeline.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Requires OpenCV 4 with `imgproc`, `videoio`, `highgui`, `ml`, and `core`.

## Executables

- **coin_counter** – Live coin detection, tracking, SVM classification, CSV export (up to 3000 points). Press `q` to quit.
- **train_svm** – Collect labeled samples by zone, train OpenCV SVM (RBF), save model and scaler. Keys: `c` capture, `s` train & save, `q` quit.
- **camera_calibration** – Collect diameter samples by zone, fit radial + tilt distortion, save calibration. Keys: `c` capture, `s` solve & save, `q` quit.

## Data Files

- **coin_calibration_robust.yaml** – Calibration (base ratio, distortion k, p_y, p_x). Created by `camera_calibration`.
- **coin_svm.yaml** – Trained SVM model. Created by `train_svm`.
- **coin_scaler.yaml** – Feature mean and scale (4 features). Created by `train_svm`.
- **coin_data.csv** – Exported coin features (diameter_mm, L, a, b). Written by `coin_counter` until 3000 rows.

Python `.pkl` files are not used; run the C++ tools to generate the YAML/CSV files.

## Run from project root

From `coin_counter_cpp/build`:

```bash
./coin_counter
./train_svm
./camera_calibration
```

Or run from the repo root with paths adjusted so the executables find `coin_calibration_robust.yaml`, `coin_svm.yaml`, and `coin_scaler.yaml` in the current working directory.
