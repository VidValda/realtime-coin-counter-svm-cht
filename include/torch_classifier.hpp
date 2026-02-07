#pragma once

#include "config.hpp"
#include "types.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <utility>
#include <vector>

#ifdef COIN_USE_TORCH
#include <torch/script.h>
#endif

namespace coin
{

  class TorchClassifier
  {
  public:
    TorchClassifier() = default;
    ~TorchClassifier() = default;

    bool load(const std::string &model_path);

    bool is_loaded() const { return module_ != nullptr; }

    int predict(const cv::Mat &frame_bgr, cv::Point2i center, int radius_px) const;

    std::vector<int> predict_batch(const cv::Mat &frame_bgr,
                                   const std::vector<std::pair<cv::Point2i, int>> &centers_radii) const;

  private:
#ifdef COIN_USE_TORCH
    struct Impl
    {
      torch::jit::script::Module module;
    };
#else
    struct Impl;
#endif
    std::unique_ptr<Impl> module_;
  };

}
