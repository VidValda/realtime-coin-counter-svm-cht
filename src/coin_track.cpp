#include "coin_track.hpp"
#include "config.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace coin
{

  CoinTrack::CoinTrack(cv::Point2i center) : center(center) {}

  bool CoinTrack::accept_diameter(double diameter_mm)
  {
    if (diameters_.empty())
    {
      diameters_.push_back(diameter_mm);
      return true;
    }
    std::vector<double> sorted(diameters_.begin(), diameters_.end());
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    double median_d = sorted[sorted.size() / 2];
    if (std::abs(diameter_mm - median_d) <= Config::MAX_DIAMETER_DEVIATION_MM)
    {
      diameters_.push_back(diameter_mm);
      if (diameters_.size() > static_cast<size_t>(Config::DIAMETER_HISTORY_LEN))
        diameters_.pop_front();
      return true;
    }
    return false;
  }

  std::optional<double> CoinTrack::stable_diameter_mm() const
  {
    if (diameters_.size() < static_cast<size_t>(Config::MIN_SAMPLES_FOR_STABLE))
      return std::nullopt;
    std::vector<double> sorted(diameters_.begin(), diameters_.end());
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    return sorted[sorted.size() / 2];
  }

  void CoinTrack::tick_missed() { ++frames_since_seen; }
  void CoinTrack::mark_seen() { frames_since_seen = 0; }

}
