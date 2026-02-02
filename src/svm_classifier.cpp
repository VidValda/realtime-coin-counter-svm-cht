#include "svm_classifier.hpp"
#include "config.hpp"
#include <opencv2/core/persistence.hpp>

namespace coin
{

  bool SVMClassifier::load(const std::string &model_path, const std::string &scaler_path)
  {
    cv::FileStorage fs(scaler_path, cv::FileStorage::READ);
    if (!fs.isOpened())
      return false;
    fs["mean"] >> scaler_mean_;
    fs["scale"] >> scaler_scale_;
    std::string model_type;
    cv::FileNode nt = fs["model_type"];
    if (!nt.empty())
      nt >> model_type;
    fs.release();
    return load(model_path, scaler_path, model_type);
  }

  bool SVMClassifier::load(const std::string &model_path, const std::string &scaler_path,
                           const std::string &model_type)
  {
    cv::FileStorage fs(scaler_path, cv::FileStorage::READ);
    if (!fs.isOpened())
      return false;
    fs["mean"] >> scaler_mean_;
    fs["scale"] >> scaler_scale_;
    fs.release();
    if (scaler_mean_.empty() || scaler_scale_.empty())
      return false;

    if (model_type == "KNN")
    {
      model_ = cv::ml::KNearest::load(model_path);
    }
    else if (model_type == "RandomForest")
    {
      model_ = cv::ml::RTrees::load(model_path);
    }
    else if (model_type == "NaiveBayes")
    {
      model_ = cv::ml::NormalBayesClassifier::load(model_path);
    }
    else
    {
      model_ = cv::ml::SVM::load(model_path);
    }
    return !model_.empty();
  }

  int SVMClassifier::predict(double diameter_mm, double L, double a, double b) const
  {
    if (!model_ || scaler_mean_.empty() || scaler_scale_.empty())
      return 0;
    cv::Mat feat(1, 4, CV_32F);
    feat.at<float>(0, 0) = static_cast<float>(diameter_mm);
    feat.at<float>(0, 1) = static_cast<float>(L);
    feat.at<float>(0, 2) = static_cast<float>(a);
    feat.at<float>(0, 3) = static_cast<float>(b);
    for (int c = 0; c < 4; ++c)
    {
      float v = feat.at<float>(0, c);
      float m = scaler_mean_.at<float>(0, c);
      float s = scaler_scale_.at<float>(0, c);
      if (s > 1e-9f)
        feat.at<float>(0, c) = (v - m) / s;
    }
    return static_cast<int>(model_->predict(feat));
  }

}
