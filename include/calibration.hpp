#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <optional>

namespace coin
{

  struct CalibrationData
  {
    double base_ratio = 0.0;
    double distortion_k = 0.0;
    double distortion_p_y = 0.0;
    double distortion_p_x = 0.0;
    cv::Point2d img_center;
    double max_dist_sq = 1.0;
    double width_half = 1.0;
    double height_half = 1.0;
    double height_px = 0.0;
    bool use_height_px = false;
  };

  std::optional<CalibrationData> load_calibration(const std::string &path);

  double pixel_diameter_to_mm(double cx, double cy, double diameter_px,
                              double ratio_px_to_mm, const CalibrationData *calibration);

  int diameter_mm_to_radius_px(double cx, double cy, double diameter_mm,
                               double ratio_px_to_mm, const CalibrationData *calibration);

}
