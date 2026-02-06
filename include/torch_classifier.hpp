#pragma once

#include "config.hpp"
#include "types.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <memory>
#include <vector>

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

  private:
    struct Impl;
    std::unique_ptr<Impl> module_;
  };

}
