#pragma once

namespace coin
{

    struct Config
    {
        // Watershed pipeline (from debug_coin_detection.py defaults)
        static constexpr int CLAHE_CLIP = 9;
        static constexpr int CLAHE_GRID = 1;
        static constexpr int BLUR_KSIZE = 9;
        /** 0=Gray, 1=H, 2=S, 3=V, 4=L, 5=A, 6=B */
        static constexpr int CHANNEL_MODE = 2;
        /** 0=Otsu, 1=Adaptive */
        static constexpr int USE_ADAPTIVE = 0;
        static constexpr int ADAPTIVE_BLOCK = 23;
        static constexpr int ADAPTIVE_C = 10;
        static constexpr int INVERT_BINARY = 1;
        static constexpr int MORPH_OPEN_SIZE = 1;
        static constexpr int MORPH_CLOSE_SIZE = 3;
        static constexpr int MORPH_OPEN_ITERS = 2;
        static constexpr int MORPH_CLOSE_ITERS = 4;
        static constexpr int BG_DILATE_SIZE = 4;

        static constexpr int DIST_MASK_SIZE = 5;
        static constexpr double WATERSHED_FG_FRAC = 0.45;

        static constexpr double CANNY_THRESHOLD1 = 50.0;
        static constexpr double CANNY_THRESHOLD2 = 150.0;

        static constexpr double HOUGH_DP = 1.5;
        static constexpr int HOUGH_MIN_DIST = 47;
        static constexpr double HOUGH_PARAM1 = 105;
        static constexpr double HOUGH_PARAM2 = 30;
        static constexpr int MIN_RADIUS_PX = 22;
        static constexpr int MAX_RADIUS_PX = 43;

        static constexpr double MIN_CONTOUR_AREA = 464.0;
        static constexpr double MIN_CIRCULARITY = 0.56;
        static constexpr double DIAMETER_MM_MIN = 10.0;
        static constexpr double DIAMETER_MM_MAX = 40.0;

        static constexpr int CENTER_MATCH_PX = 20;
        static constexpr int DIAMETER_HISTORY_LEN = 100;
        static constexpr int MAX_FRAMES_MISSING = 5;
        static constexpr double MAX_DIAMETER_DEVIATION_MM = 3.0;
        static constexpr int MIN_SAMPLES_FOR_STABLE = 3;

        static constexpr const char *CSV_PATH = "coin_data.csv";
        static constexpr int CSV_TARGET_POINTS = 3000;

        static constexpr const char *CALIBRATION_PATH = "coin_calibration_robust.yaml";
        static constexpr const char *SVM_MODEL_PATH = "coin_svm.yaml";
        static constexpr const char *SVM_SCALER_PATH = "coin_scaler.yaml";
        static constexpr const char *CLASSIFIER_MODEL_PATHS[4] = {
            "coin_svm.yaml", "coin_knn.yaml", "coin_rtrees.yaml", "coin_nb.yaml"};
        static constexpr const char *CLASSIFIER_NAMES[6] = {"SVM", "KNN", "RandomForest", "NaiveBayes", "CNN", "ResNet18"};
        static constexpr const char *CLASSIFIER_DEFAULT_FILE = "classifier_default.txt";
        /** TorchScript models (export with export_torchscript.py from coin_cnn.pt / coin_resnet18.pt). */
        static constexpr const char *COIN_CNN_TRACED_PATH = "coin_cnn_traced.pt";
        static constexpr const char *COIN_RESNET18_TRACED_PATH = "coin_resnet18_traced.pt";
        static constexpr int COIN_CROP_SIZE = 150;
        static constexpr const char *TRAINING_DATA_DIR = "training_data_2";
        static constexpr const char *TRAINING_MANIFEST = "training_data_2/manifest.csv";

        /** Test videos for coin counter (used when USE_TEST_VIDEOS is true). */
        static constexpr const char *TEST_VIDEO_1 = "test1.mp4";
        static constexpr const char *TEST_VIDEO_2 = "test2.mp4";
        /** If true, read from test videos instead of camera. */
        static constexpr bool USE_TEST_VIDEOS = false;

        static constexpr double PAPER_WIDTH_MM = 330.0;
        static constexpr double PAPER_HEIGHT_MM = 216.0;
        static constexpr double SCALE_FACTOR = 5.46104;

        static constexpr int PAPER_LINE_MIN_LENGTH = 100;
        static constexpr int PAPER_MORPH_KERNEL = 3;
        static constexpr double PAPER_APPROX_EPS_FACTOR = 0.02;
        static constexpr int PAPER_MIN_AREA = 10000;
        /** Run find_paper_corners on frame scaled to this max width (0 = full res). Speeds up LSD. */
        static constexpr int PAPER_DETECT_MAX_WIDTH = 480;
        static constexpr int STABILIZER_WINDOW = 10;
        /** Run coin detection at this scale (0.5 = half res). Detection results scaled back. Lower = faster, ~4x at 0.5. */
        static constexpr double COIN_DETECT_SCALE = 1;

        /** Run find_paper_corners every this many frames (1 = every frame). Higher values reduce CPU when paper is stable. */
        static constexpr int PAPER_DETECT_EVERY_N_FRAMES = 50;

        /** Run coin detection (watershed) every this many frames (1 = every frame). Between detections the tracker
         *  state is reused and only the drawing/classification step runs. 3-5 is a good trade-off. */
        static constexpr int COIN_DETECT_EVERY_N_FRAMES = 5;

        /** Max width for display windows (imshow). Reduces memory and avoids GUI backend issues. */
        static constexpr int MAX_DISPLAY_WIDTH_PX = 960;

        /** If true, fill and show debug windows (Markers, Segmentation, Binary, Sure FG, Distance). Slower. */
        static constexpr bool SHOW_DEBUG_VIEWS = false;

        /** If true, print per-frame pipeline timings to stderr. Toggle at runtime with 't' key. */
        static constexpr bool PRINT_TIMINGS_DEFAULT = false;

        static constexpr int CLUSTER_COLORS_BGR[6][3] = {
            {180, 119, 31}, {14, 127, 255}, {44, 160, 44}, {40, 39, 214}, {189, 103, 148}, {75, 86, 140}};

        static constexpr double CLASS_TO_VALUE_EUR[6] = {
            0.20, 0.10, 1.0, 0.01, 0.02, 0.05};
    };

}
