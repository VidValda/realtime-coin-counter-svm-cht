#pragma once

#include "types.hpp"
#include "coin_track.hpp"
#include <memory>
#include <vector>

namespace coin
{

  class CoinTracker
  {
  public:
    void update(const Detections &detections);
    std::vector<std::pair<cv::Point2i, double>> get_stable_entries() const;

  private:
    std::vector<std::unique_ptr<CoinTrack>> tracks_;
  };

}
