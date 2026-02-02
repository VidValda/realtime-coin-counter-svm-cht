#pragma once

#include <opencv2/core.hpp>
#include <deque>
#include <optional>

namespace coin
{

  class CornerStabilizer
  {
  public:
    explicit CornerStabilizer(int window_size = 10);
    std::optional<cv::Mat> update(const cv::Mat *corners);

  private:
    std::deque<cv::Mat> history_;
    int window_size_;
  };

}
