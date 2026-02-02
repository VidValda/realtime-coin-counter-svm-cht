#include "corner_stabilizer.hpp"

namespace coin
{

  CornerStabilizer::CornerStabilizer(int window_size) : window_size_(window_size) {}

  std::optional<cv::Mat> CornerStabilizer::update(const cv::Mat *corners)
  {
    if (!corners || corners->empty())
      return std::nullopt;
    history_.push_back(corners->clone());
    while (history_.size() > static_cast<size_t>(window_size_))
      history_.pop_front();
    cv::Mat avg = cv::Mat::zeros(history_.front().size(), history_.front().type());
    for (const auto &m : history_)
      avg += m;
    avg /= static_cast<int>(history_.size());
    cv::Mat result;
    avg.convertTo(result, CV_32F);
    return result;
  }

}
