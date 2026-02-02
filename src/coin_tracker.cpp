#include "coin_tracker.hpp"
#include "config.hpp"
#include <cmath>
#include <algorithm>

namespace coin
{

  void CoinTracker::update(const Detections &detections)
  {
    for (auto &t : tracks_)
      t->tick_missed();
    std::vector<bool> used(tracks_.size(), false);

    for (const auto &det : detections)
    {
      int best_i = -1;
      double best_dist = Config::CENTER_MATCH_PX + 1.0;
      for (size_t i = 0; i < tracks_.size(); ++i)
      {
        if (used[i])
          continue;
        double dx = det.center.x - tracks_[i]->center.x;
        double dy = det.center.y - tracks_[i]->center.y;
        double d = std::hypot(dx, dy);
        if (d < best_dist)
        {
          best_dist = d;
          best_i = static_cast<int>(i);
        }
      }
      if (best_i >= 0)
      {
        tracks_[best_i]->center = det.center;
        tracks_[best_i]->accept_diameter(det.diameter_mm);
        tracks_[best_i]->mark_seen();
        used[best_i] = true;
      }
      else
      {
        auto track = std::make_unique<CoinTrack>(det.center);
        track->accept_diameter(det.diameter_mm);
        track->mark_seen();
        tracks_.push_back(std::move(track));
        used.push_back(true);
      }
    }

    std::vector<std::unique_ptr<CoinTrack>> kept;
    for (size_t i = 0; i < tracks_.size(); ++i)
    {
      bool was_used = (i < used.size() && used[i]);
      bool within_missing = tracks_[i]->frames_since_seen <= Config::MAX_FRAMES_MISSING;
      if (was_used || within_missing)
        kept.push_back(std::move(tracks_[i]));
    }
    tracks_ = std::move(kept);
  }

  std::vector<std::pair<cv::Point2i, double>> CoinTracker::get_stable_entries() const
  {
    std::vector<std::pair<cv::Point2i, double>> result;
    for (const auto &t : tracks_)
    {
      auto d = t->stable_diameter_mm();
      if (d.has_value())
        result.emplace_back(t->center, *d);
    }
    return result;
  }

}
