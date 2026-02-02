from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "build"
TRAINING_BASE = OUT_DIR / "training_data"
MANIFEST_PATH = TRAINING_BASE / "manifest.csv"

CLASSIFIER_MODEL_NAMES = ["coin_svm.yaml", "coin_knn.yaml", "coin_rtrees.yaml", "coin_nb.yaml"]
SCALER_PATH = "coin_scaler.yaml"
DEFAULT_FILE = "classifier_default.txt"
MODEL_TYPE_NAMES = ["SVM", "KNN", "RandomForest", "NaiveBayes"]


def load_manifest(path: Path) -> list[tuple[str, int, float]]:
    rows = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines or "path" not in lines[0]:
        return rows
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            rel_path = parts[0].strip()
            class_id = int(parts[1].strip())
            sdiam = parts[2].strip().rstrip("\r\n ")
            diameter_mm = float(sdiam)
            rows.append((rel_path, class_id, diameter_mm))
        except (ValueError, IndexError):
            continue
    return rows


def sample_mean_lab_inside_circle(bgr: np.ndarray, center_xy: tuple[int, int], radius_px: int) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    cx, cy = center_xy
    inner_r = max(2, int(radius_px * 0.7))
    y0 = max(0, cy - inner_r)
    y1 = min(h, cy + inner_r + 1)
    x0 = max(0, cx - inner_r)
    x1 = min(w, cx + inner_r + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    roi = lab[y0:y1, x0:x1]
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    for y in range(roi.shape[0]):
        for x in range(roi.shape[1]):
            gx = x0 + x - cx
            gy = y0 + y - cy
            if gx * gx + gy * gy <= inner_r * inner_r:
                mask[y, x] = 255
    mean_val = cv2.mean(roi, mask=mask)
    return np.array([mean_val[0], mean_val[1], mean_val[2]], dtype=np.float64)


def extract_features(bgr: np.ndarray, diameter_mm: float) -> np.ndarray | None:
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    cx, cy = w // 2, h // 2
    radius_px = max(2, int(min(w, h) * 0.35))
    lab_vec = sample_mean_lab_inside_circle(bgr, (cx, cy), radius_px)
    if lab_vec is None:
        return None
    return np.array([diameter_mm, lab_vec[0], lab_vec[1], lab_vec[2]], dtype=np.float32)


def load_dataset(manifest_path: Path, base_dir: Path):
    rows = load_manifest(manifest_path)
    if not rows:
        return None, None
    X_list = []
    y_list = []
    for rel_path, class_id, diameter_mm in rows:
        full_path = base_dir / rel_path
        if not full_path.is_file():
            continue
        img = cv2.imread(str(full_path))
        feat = extract_features(img, diameter_mm)
        if feat is not None:
            X_list.append(feat)
            y_list.append(class_id)
    if not X_list:
        return None, None
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def scale_like_cpp(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    mean = np.mean(X, axis=0, keepdims=True).astype(np.float32)
    centered = X - mean
    var = np.mean(centered ** 2, axis=0, keepdims=True)
    scale = np.sqrt(var).astype(np.float32)
    scale = np.maximum(scale, 1e-9)
    X_scaled = (X - mean) / scale
    return X_scaled, mean, scale


def main() -> int:
    manifest = load_manifest(MANIFEST_PATH)
    if not manifest:
        print(f"No manifest or no rows in {MANIFEST_PATH}")
        print("Run the acquisition tool first: train_acquisition")
        return 1

    X, y = load_dataset(MANIFEST_PATH, TRAINING_BASE)
    if X is None or y is None:
        print("No valid samples loaded.")
        return 1

    n = len(X)
    if n <= 10:
        print(f"Not enough valid samples ({n}). Need more than 10.")
        return 1

    print(f"Loaded {n} samples from {TRAINING_BASE}")

    X_scaled, mean, scale = scale_like_cpp(X)

    rng = np.random.default_rng(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    n_test = max(1, n // 5)
    n_train = n - n_test
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train_part = X_scaled[train_idx]
    y_train_part = y[train_idx]
    X_test_part = X_scaled[test_idx]
    y_test_part = y[test_idx]

    def eval_model(model, X_te, y_te) -> float:
        correct = 0
        for i in range(len(y_te)):
            row = X_te[i : i + 1]
            pred = model.predict(row)
            if isinstance(pred, (tuple, list)) and len(pred) == 2:
                pred = int(pred[1].flat[0]) if pred[1].size else int(pred[0])
            else:
                pred = int(pred)
            if pred == y_te[i]:
                correct += 1
        return correct / len(y_te) if y_te.size else 0.0

    print(f"Testing models on {n_train} train / {n_test} test samples...")

    train_data_part = cv2.ml.TrainData_create(
        X_train_part,
        cv2.ml.ROW_SAMPLE,
        y_train_part.reshape(-1, 1).astype(np.int32),
    )

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setGamma(0.5)
    svm.setC(1.0)
    svm.train(train_data_part)
    score_svm = eval_model(svm, X_test_part, y_test_part)
    print(f"  SVM: {score_svm:.4f} accuracy")

    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(5)
    knn.train(train_data_part)
    score_knn = eval_model(knn, X_test_part, y_test_part)
    print(f"  KNN: {score_knn:.4f} accuracy")

    rtrees = cv2.ml.RTrees_create()
    rtrees.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.01))
    rtrees.train(train_data_part)
    score_rf = eval_model(rtrees, X_test_part, y_test_part)
    print(f"  RandomForest: {score_rf:.4f} accuracy")

    nb = cv2.ml.NormalBayesClassifier_create()
    nb.train(train_data_part)
    score_nb = eval_model(nb, X_test_part, y_test_part)
    print(f"  NaiveBayes: {score_nb:.4f} accuracy")

    scores = [score_svm, score_knn, score_rf, score_nb]
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]
    print(f"Winner: {MODEL_TYPE_NAMES[best_idx]} ({best_score:.4f})")
    try:
        line = input("Save which model? [1=SVM 2=KNN 3=RandomForest 4=NaiveBayes] (Enter=winner): ").strip()
    except EOFError:
        line = ""
    save_idx = best_idx
    if line == "1":
        save_idx = 0
    elif line == "2":
        save_idx = 1
    elif line == "3":
        save_idx = 2
    elif line == "4":
        save_idx = 3

    full_train_data = cv2.ml.TrainData_create(
        X_scaled,
        cv2.ml.ROW_SAMPLE,
        y.reshape(-1, 1).astype(np.int32),
    )
    svm.train(full_train_data)
    knn.train(full_train_data)
    rtrees.train(full_train_data)
    nb.train(full_train_data)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_paths = [OUT_DIR / name for name in CLASSIFIER_MODEL_NAMES]
    svm.save(str(model_paths[0]))
    knn.save(str(model_paths[1]))
    rtrees.save(str(model_paths[2]))
    nb.save(str(model_paths[3]))

    scaler_full_path = OUT_DIR / SCALER_PATH
    fs = cv2.FileStorage(str(scaler_full_path), cv2.FILE_STORAGE_WRITE)
    fs.write("mean", mean)
    fs.write("scale", scale)
    fs.write("model_type", MODEL_TYPE_NAMES[save_idx])
    fs.release()

    default_path = OUT_DIR / DEFAULT_FILE
    with open(default_path, "w") as f:
        f.write(f"{save_idx}\n")

    print(f"Saved all 4 models. Active (default): {MODEL_TYPE_NAMES[save_idx]} ({model_paths[save_idx]}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
