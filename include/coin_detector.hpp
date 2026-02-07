#pragma once

#include "types.hpp"
#include "calibration.hpp"
#include <opencv2/core.hpp>
#include <vector>
#include <optional>

namespace coin
{

    /** Optional debug outputs: marker visualization and segmentation (watershed result). */
    struct DebugViews
    {
        cv::Mat markers_vis;  /**< Colored markers (one color per label). */
        cv::Mat segmentation; /**< Watershed segmentation overlay. */
        cv::Mat binary;       /**< Binary after morph. */
        cv::Mat sure_fg;      /**< Sure foreground. */
        cv::Mat dist_vis;     /**< Distance transform visualization. */
    };

    cv::Mat preprocess_for_circles(const cv::Mat &frame);

    std::vector<cv::Vec3f> find_circle_candidates(const cv::Mat &blurred);

    std::optional<Detection> measure_circle_diameter(const cv::Mat &frame_gray,
                                                     int x, int y, int r, double ratio_px_to_mm);

    /** Watershed-based coin detection. If out_debug is non-null, fills marker and segmentation views.
     * pixel_scale: when running on a downscaled image (e.g. 0.5), pass that scale so MIN_CONTOUR_AREA is adjusted. */
    Detections detect_and_measure_coins(const cv::Mat &frame, double ratio_px_to_mm,
                                        DebugViews *out_debug = nullptr, double pixel_scale = 1.0);

    std::optional<cv::Mat> find_paper_corners(const cv::Mat &frame);

    cv::Mat order_corners(const cv::Mat &corners);

    std::optional<CoinFeature> sample_mean_lab_inside_circle(const cv::Mat &frame_bgr,
                                                             cv::Point2i center, int radius_px);

    /** Same as above but uses pre-converted Lab image (avoids repeated BGR→Lab conversion). */
    std::optional<CoinFeature> sample_mean_lab_inside_circle_from_lab(const cv::Mat &frame_lab,
                                                                      cv::Point2i center, int radius_px);

    /** Converts BGR→Lab once, then samples per entry. Prefer over per-call conversion when many coins. */
    std::vector<CoinFeature> collect_coin_features(const cv::Mat &frame_bgr,
                                                   const std::vector<std::pair<cv::Point2i, double>> &entries,
                                                   double ratio_px_to_mm);

}
