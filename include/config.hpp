#pragma once

namespace coin
{

  struct Config
  {
    static constexpr int CLAHE_CLIP = 5;
    static constexpr int CLAHE_GRID = 2;
    static constexpr int BLUR_KSIZE = 5;

    static constexpr double CANNY_THRESHOLD1 = 50.0;
    static constexpr double CANNY_THRESHOLD2 = 150.0;

    static constexpr double HOUGH_DP = 1.5;
    static constexpr int HOUGH_MIN_DIST = 47;
    static constexpr double HOUGH_PARAM1 = 105;
    static constexpr double HOUGH_PARAM2 = 30;
    static constexpr int MIN_RADIUS_PX = 22;
    static constexpr int MAX_RADIUS_PX = 43;

    static constexpr double MIN_CONTOUR_AREA = 100.0;
    static constexpr double MIN_CIRCULARITY = 0.5;
    static constexpr double DIAMETER_MM_MIN = 10.0;
    static constexpr double DIAMETER_MM_MAX = 40.0;

    static constexpr int CENTER_MATCH_PX = 20;
    static constexpr int DIAMETER_HISTORY_LEN = 100;
    static constexpr int MAX_FRAMES_MISSING = 5;
    static constexpr double MAX_DIAMETER_DEVIATION_MM = 3.0;
    static constexpr int MIN_SAMPLES_FOR_STABLE = 3;

    static constexpr int HIST_WIDTH = 320;
    static constexpr int HIST_HEIGHT = 240;
    static constexpr double HIST_BIN_MIN = 10.0;
    static constexpr double HIST_BIN_MAX = 41.0;
    static constexpr double HIST_BIN_STEP = 0.2;

    static constexpr const char *CSV_PATH = "coin_data.csv";
    static constexpr int CSV_TARGET_POINTS = 3000;

    static constexpr const char *CALIBRATION_PATH = "coin_calibration_robust.yaml";
    static constexpr const char *SVM_MODEL_PATH = "coin_svm.yaml";
    static constexpr const char *SVM_SCALER_PATH = "coin_scaler.yaml";
    static constexpr const char *CLASSIFIER_MODEL_PATHS[4] = {
        "coin_svm.yaml", "coin_knn.yaml", "coin_rtrees.yaml", "coin_nb.yaml"};
    static constexpr const char *CLASSIFIER_NAMES[4] = {"SVM", "KNN", "RandomForest", "NaiveBayes"};
    static constexpr const char *CLASSIFIER_DEFAULT_FILE = "classifier_default.txt";
    static constexpr const char *TRAINING_DATA_DIR = "training_data";
    static constexpr const char *TRAINING_MANIFEST = "training_data/manifest.csv";

    static constexpr double PAPER_WIDTH_MM = 330.0;
    static constexpr double PAPER_HEIGHT_MM = 216.0;
    static constexpr double SCALE_FACTOR = 3.0;

    static constexpr int CLUSTER_COLORS_BGR[6][3] = {
        {180, 119, 31}, {14, 127, 255}, {44, 160, 44}, {40, 39, 214}, {189, 103, 148}, {75, 86, 140}};

    static constexpr double CLASS_TO_VALUE_EUR[6] = {
        0.20, 0.10, 1.0, 0.01, 0.02, 0.05};
  };

}
