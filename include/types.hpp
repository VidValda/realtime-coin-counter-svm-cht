#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <utility>

namespace coin
{

  using Point2i = cv::Point2i;
  using Point2f = cv::Point2f;

  struct Detection
  {
    cv::Point2i center;
    double diameter_mm;
  };

  struct CoinFeature
  {
    double diameter_mm;
    double L;
    double a;
    double b;
  };

  using Detections = std::vector<Detection>;
  using CoinFeatures = std::vector<CoinFeature>;

}
