#pragma once

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <vector>
#include <optional>

namespace coin
{

  class SVMClassifier
  {
  public:
    bool load(const std::string &model_path, const std::string &scaler_path);
    bool load(const std::string &model_path, const std::string &scaler_path, const std::string &model_type);
    int predict(double diameter_mm, double L, double a, double b) const;
    bool is_loaded() const { return model_ != nullptr; }

  private:
    cv::Ptr<cv::ml::StatModel> model_;
    cv::Mat scaler_mean_;
    cv::Mat scaler_scale_;
  };

}
