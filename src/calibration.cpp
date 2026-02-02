#include "calibration.hpp"
#include "config.hpp"
#include <opencv2/core/persistence.hpp>
#include <cmath>

namespace coin
{

  std::optional<CalibrationData> load_calibration(const std::string &path)
  {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened())
      return std::nullopt;
    CalibrationData cal;
    fs["base_ratio"] >> cal.base_ratio;
    fs["distortion_k"] >> cal.distortion_k;
    fs["distortion_p_y"] >> cal.distortion_p_y;
    fs["distortion_p_x"] >> cal.distortion_p_x;
    cv::FileNode center = fs["img_center"];
    if (!center.empty() && center.isSeq() && center.size() >= 2)
    {
      center[0] >> cal.img_center.x;
      center[1] >> cal.img_center.y;
    }
    fs["max_dist_sq"] >> cal.max_dist_sq;
    fs["width_half"] >> cal.width_half;
    fs["height_half"] >> cal.height_half;
    fs["height_px"] >> cal.height_px;
    fs["use_height_px"] >> cal.use_height_px;
    if (cal.height_px > 0 && !cal.use_height_px)
      cal.use_height_px = true;
    fs.release();
    return cal;
  }

  double pixel_diameter_to_mm(double cx, double cy, double diameter_px,
                              double ratio_px_to_mm, const CalibrationData *calibration)
  {
    if (!calibration)
      return diameter_px * ratio_px_to_mm;
    double dx = cx - calibration->img_center.x;
    double dy = cy - calibration->img_center.y;
    double dist_sq = (dx * dx + dy * dy) / calibration->max_dist_sq;
    double y_norm = calibration->use_height_px
                        ? (calibration->height_px - cy) / calibration->height_px
                        : (cy - calibration->img_center.y) / calibration->height_half;
    double x_norm = (cx - calibration->img_center.x) / calibration->width_half;
    double correction = 1.0 + calibration->distortion_k * dist_sq + calibration->distortion_p_y * y_norm + calibration->distortion_p_x * x_norm;
    return (diameter_px * calibration->base_ratio) * correction;
  }

  int diameter_mm_to_radius_px(double cx, double cy, double diameter_mm,
                               double ratio_px_to_mm, const CalibrationData *calibration)
  {
    if (!calibration)
      return std::max(2, static_cast<int>((diameter_mm / ratio_px_to_mm) / 2));
    double dx = cx - calibration->img_center.x;
    double dy = cy - calibration->img_center.y;
    double dist_sq = (dx * dx + dy * dy) / calibration->max_dist_sq;
    double y_norm = calibration->use_height_px
                        ? (calibration->height_px - cy) / calibration->height_px
                        : (cy - calibration->img_center.y) / calibration->height_half;
    double x_norm = (cx - calibration->img_center.x) / calibration->width_half;
    double correction = 1.0 + calibration->distortion_k * dist_sq + calibration->distortion_p_y * y_norm + calibration->distortion_p_x * x_norm;
    int r = static_cast<int>((diameter_mm / (calibration->base_ratio * correction)) / 2);
    return std::max(2, r);
  }

}
