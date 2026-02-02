#pragma once

#include "types.hpp"
#include "calibration.hpp"
#include <opencv2/core.hpp>
#include <vector>
#include <optional>

namespace coin
{

    cv::Mat preprocess_for_circles(const cv::Mat &frame);

    std::vector<cv::Vec3f> find_circle_candidates(const cv::Mat &blurred);

    std::optional<Detection> measure_circle_diameter(const cv::Mat &frame_gray,
                                                     int x, int y, int r, double ratio_px_to_mm, const CalibrationData *calibration);

    Detections detect_and_measure_coins(const cv::Mat &frame, double ratio_px_to_mm,
                                        const CalibrationData *calibration);

    std::optional<cv::Mat> find_paper_corners(const cv::Mat &frame);

    cv::Mat order_corners(const cv::Mat &corners);

    std::optional<CoinFeature> sample_mean_lab_inside_circle(const cv::Mat &frame_bgr,
                                                             cv::Point2i center, int radius_px);

    std::vector<CoinFeature> collect_coin_features(const cv::Mat &frame_bgr,
                                                   const std::vector<std::pair<cv::Point2i, double>> &entries,
                                                   double ratio_px_to_mm, const CalibrationData *calibration);

}
