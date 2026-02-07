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

  /** Classifier using a TorchScript model (CNN or ResNet18). Loads traced .pt from export_torchscript.py. */
  class TorchClassifier
  {
  public:
    TorchClassifier() = default;
    ~TorchClassifier() = default;

    /** Load TorchScript model from path (e.g. coin_cnn_traced.pt or coin_resnet18_traced.pt). */
    bool load(const std::string &model_path);

    bool is_loaded() const { return module_ != nullptr; }

    /**
     * Predict class index [0..5] for a single coin: crop 150x150 around center in frame_bgr,
     * preprocess (BGR->RGB, ImageNet normalize), run model, return argmax.
     */
    int predict(const cv::Mat &frame_bgr, cv::Point2i center, int radius_px) const;

    /**
     * Batch prediction: one forward pass for all coins. centers_radii[i] = (center, radius_px).
     * Returns class index [0..5] per coin. Much faster than calling predict() in a loop.
     */
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
