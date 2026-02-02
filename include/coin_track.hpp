#pragma once

#include <opencv2/core.hpp>
#include <deque>
#include <optional>

namespace coin
{

  class CoinTrack
  {
  public:
    explicit CoinTrack(cv::Point2i center);
    bool accept_diameter(double diameter_mm);
    std::optional<double> stable_diameter_mm() const;
    void tick_missed();
    void mark_seen();

    cv::Point2i center;
    int frames_since_seen = 0;

  private:
    std::deque<double> diameters_;
  };

}
