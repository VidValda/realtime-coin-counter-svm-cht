#include "torch_classifier.hpp"
#include "config.hpp"
#include <torch/script.h>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace coin
{

  namespace
  {
    constexpr double IMAGENET_MEAN[] = {0.485, 0.456, 0.406};
    constexpr double IMAGENET_STD[] = {0.229, 0.224, 0.225};
    constexpr int CROP = Config::COIN_CROP_SIZE;
  }

  struct TorchClassifier::Impl
  {
    torch::jit::script::Module module;

    /** Extract 150x150 crop centered at (cx,cy); pad with black if out of bounds. */
    static cv::Mat extract_crop(const cv::Mat &frame_bgr, int cx, int cy)
    {
      const int h = frame_bgr.rows, w = frame_bgr.cols;
      const int half = CROP / 2;
      int x0 = cx - half, y0 = cy - half;
      int x1 = x0 + CROP, y1 = y0 + CROP;
      cv::Mat out = cv::Mat::zeros(CROP, CROP, frame_bgr.type());
      int src_x0 = std::max(0, x0), src_y0 = std::max(0, y0);
      int src_x1 = std::min(w, x1), src_y1 = std::min(h, y1);
      int dst_x0 = src_x0 - x0, dst_y0 = src_y0 - y0;
      int dst_x1 = dst_x0 + (src_x1 - src_x0), dst_y1 = dst_y0 + (src_y1 - src_y0);
      if (dst_x1 > dst_x0 && dst_y1 > dst_y0)
      {
        cv::Mat src_roi = frame_bgr(cv::Rect(src_x0, src_y0, src_x1 - src_x0, src_y1 - src_y0));
        src_roi.copyTo(out(cv::Rect(dst_x0, dst_y0, dst_x1 - dst_x0, dst_y1 - dst_y0)));
      }
      return out;
    }

    /** Convert BGR 150x150 OpenCV mat to 1x3x150x150 float tensor (ImageNet normalize). */
    static torch::Tensor mat_to_tensor(const cv::Mat &bgr)
    {
      cv::Mat rgb;
      cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
      if (rgb.isContinuous())
        rgb = rgb.clone();
      torch::Tensor t = torch::from_blob(rgb.data, {CROP, CROP, 3}, torch::kByte);
      t = t.permute({2, 0, 1}).to(torch::kFloat32).div(255.0);
      t = t.unsqueeze(0);
      for (int c = 0; c < 3; ++c)
      {
        t[0][c] = (t[0][c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
      }
      return t;
    }
  };

  bool TorchClassifier::load(const std::string &model_path)
  {
    try
    {
      auto impl = std::make_unique<Impl>();
      impl->module = torch::jit::load(model_path);
      impl->module.eval();
      module_ = std::move(impl);
      return true;
    }
    catch (const c10::Error &e)
    {
      std::cerr << "TorchClassifier: failed to load " << model_path << ": " << e.what() << "\n";
      return false;
    }
  }

  int TorchClassifier::predict(const cv::Mat &frame_bgr, cv::Point2i center, int radius_px) const
  {
    if (!module_)
      return 0;
    torch::NoGradGuard no_grad;
    cv::Mat crop = Impl::extract_crop(frame_bgr, center.x, center.y);
    torch::Tensor input = Impl::mat_to_tensor(crop);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    torch::Tensor output = module_->module.forward(inputs).toTensor();
    torch::Tensor pred = output.argmax(1);
    return pred.item<int>();
  }

  std::vector<int> TorchClassifier::predict_batch(const cv::Mat &frame_bgr,
                                                   const std::vector<std::pair<cv::Point2i, int>> &centers_radii) const
  {
    std::vector<int> out;
    if (!module_ || centers_radii.empty())
      return out;
    out.resize(centers_radii.size());
    torch::NoGradGuard no_grad;
    std::vector<torch::Tensor> tensors;
    tensors.reserve(centers_radii.size());
    for (const auto &cr : centers_radii)
    {
      cv::Mat crop = Impl::extract_crop(frame_bgr, cr.first.x, cr.first.y);
      tensors.push_back(Impl::mat_to_tensor(crop));
    }
    torch::Tensor batch = torch::stack(tensors, 0);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch);
    torch::Tensor output = module_->module.forward(inputs).toTensor();
    torch::Tensor pred = output.argmax(1);
    auto accessor = pred.accessor<int64_t, 1>();
    for (size_t i = 0; i < centers_radii.size(); ++i)
      out[i] = static_cast<int>(accessor[i]);
    return out;
  }

}
