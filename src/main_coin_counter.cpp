#include "config.hpp"
#include "calibration.hpp"
#include "svm_classifier.hpp"
#include "corner_stabilizer.hpp"
#include "coin_tracker.hpp"
#include "coin_detector.hpp"
#ifdef COIN_USE_TORCH
#include "torch_classifier.hpp"
#endif
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>

namespace
{
  /** Resize mat for display if wider than MAX_DISPLAY_WIDTH_PX to reduce memory and GUI pressure. */
  cv::Mat for_display(const cv::Mat &mat)
  {
    if (mat.cols <= coin::Config::MAX_DISPLAY_WIDTH_PX || mat.empty())
      return mat;
    double scale = static_cast<double>(coin::Config::MAX_DISPLAY_WIDTH_PX) / mat.cols;
    cv::Mat out;
    cv::resize(mat, out, cv::Size(), scale, scale, cv::INTER_LINEAR);
    return out;
  }

  void create_all_windows_once()
  {
    //cv::namedWindow("Scanner", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Anti-Glare Detection", cv::WINDOW_AUTOSIZE);
    //cv::namedWindow("Warped", cv::WINDOW_AUTOSIZE);
    //cv::namedWindow("Debug: Markers", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Debug: Segmentation", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Debug: Binary", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Debug: Sure FG", cv::WINDOW_AUTOSIZE);
    // cv::namedWindow("Debug: Distance", cv::WINDOW_AUTOSIZE);
  }

  double get_ratio_px_to_mm()
  {
    return 1.0 / coin::Config::SCALE_FACTOR;
  }

  int read_default_classifier_index()
  {
    std::ifstream f(coin::Config::CLASSIFIER_DEFAULT_FILE);
    int idx = 0;
    int max_idx =
#ifdef COIN_USE_TORCH
        5;
#else
        3;
#endif
    if (f && (f >> idx) && idx >= 0 && idx <= max_idx)
      return idx;
    return 0;
  }

  cv::Mat draw_coins(const cv::Mat &frame, coin::CoinTracker &tracker,
                     double ratio_px_to_mm, coin::SVMClassifier &svm,
                     const std::string &classifier_name,
#ifdef COIN_USE_TORCH
                     int classifier_index, coin::TorchClassifier *torch_clf)
#else
                     int /* classifier_index */, void *torch_clf)
#endif
  {
    (void)torch_clf;
    cv::Mat display;
    frame.copyTo(display);
    auto entries = tracker.get_stable_entries();
    auto rows = coin::collect_coin_features(frame, entries, ratio_px_to_mm);
    double total_eur = 0.0;

#ifdef COIN_USE_TORCH
    bool use_torch = (classifier_index >= 4 && torch_clf && torch_clf->is_loaded());
#else
    bool use_torch = false;
#endif

    for (size_t i = 0; i < entries.size(); ++i)
    {
      const auto &e = entries[i];
      int r = coin::diameter_mm_to_radius_px(e.second, ratio_px_to_mm);
      double display_diameter_mm = e.second;
      cv::Scalar color(0, 255, 0);
      int cid = 0;
      if (use_torch)
      {
#ifdef COIN_USE_TORCH
        cid = torch_clf->predict(frame, e.first, r);
        cid = cid % 6;
        color = cv::Scalar(coin::Config::CLUSTER_COLORS_BGR[cid][0],
                           coin::Config::CLUSTER_COLORS_BGR[cid][1],
                           coin::Config::CLUSTER_COLORS_BGR[cid][2]);
        total_eur += coin::Config::CLASS_TO_VALUE_EUR[cid];
#endif
      }
      else if (i < rows.size())
      {
        cid = svm.predict(rows[i].diameter_mm, rows[i].L, rows[i].a, rows[i].b);
        cid = cid % 6;
        color = cv::Scalar(coin::Config::CLUSTER_COLORS_BGR[cid][0],
                           coin::Config::CLUSTER_COLORS_BGR[cid][1],
                           coin::Config::CLUSTER_COLORS_BGR[cid][2]);
        total_eur += coin::Config::CLASS_TO_VALUE_EUR[cid];
      }
      cv::circle(display, e.first, r, color, 4);
      std::string label = std::to_string(static_cast<int>(display_diameter_mm * 10) / 10.0).substr(0, 4) + "mm";
      cv::putText(display, label, cv::Point(e.first.x - 20, e.first.y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 4);
    }

    cv::rectangle(display, cv::Point(10, 10), cv::Point(280, 90), cv::Scalar(0, 0, 0), -1);
    cv::rectangle(display, cv::Point(10, 10), cv::Point(280, 90), cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Coins: " + std::to_string(entries.size()), cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "Total: " << total_eur << " EUR";
    cv::putText(display, oss.str(), cv::Point(20, 62), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
#ifdef COIN_USE_TORCH
    cv::putText(display, "Clf: " + classifier_name + " (1-6)", cv::Point(20, 82),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
#else
    cv::putText(display, "Clf: " + classifier_name + " (1-4)", cv::Point(20, 82),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
#endif

    return display;
  }

  bool run_coin_detection(const cv::Mat &warped, double ratio_px_to_mm,
                          coin::CoinTracker &tracker, coin::SVMClassifier &svm, const std::string &classifier_name,
#ifdef COIN_USE_TORCH
                          int classifier_index, coin::TorchClassifier *torch_clf)
#else
                          int /* classifier_index */, void * /* torch_clf */)
#endif
  {
    const double keep_frac = 1;
    int cw = static_cast<int>(warped.cols * keep_frac);
    int ch = static_cast<int>(warped.rows * keep_frac);
    int cx = (warped.cols - cw) / 2;
    int cy = (warped.rows - ch) / 2;
    cv::Rect roi(cx, cy, cw, ch);
    cv::Mat warped_crop = warped(roi).clone();

    // coin::DebugViews debug;
    coin::DebugViews debug;
    auto detections = coin::detect_and_measure_coins(warped_crop, ratio_px_to_mm, &debug);
    for (auto &d : detections)
      d.center += cv::Point2i(roi.x, roi.y);
    tracker.update(detections);
#ifdef COIN_USE_TORCH
    cv::Mat display = draw_coins(warped, tracker, ratio_px_to_mm, svm, classifier_name, classifier_index, torch_clf);
#else
    cv::Mat display = draw_coins(warped, tracker, ratio_px_to_mm, svm, classifier_name, 0, nullptr);
#endif
    {
      static int64_t prev_ticks = cv::getTickCount();
      int64_t ticks = cv::getTickCount();
      double fps = cv::getTickFrequency() / std::max(ticks - prev_ticks, int64_t(1));
      prev_ticks = ticks;
      cv::putText(display, "FPS: " + std::to_string(static_cast<int>(std::round(fps))),
                  cv::Point(display.cols - 100, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("Anti-Glare Detection", for_display(display));

    //if (!debug.markers_vis.empty() && debug.markers_vis.total() > 0)
    //  cv::imshow("Debug: Markers", for_display(debug.markers_vis));
    // if (!debug.segmentation.empty() && debug.segmentation.total() > 0)
    //   cv::imshow("Debug: Segmentation", for_display(debug.segmentation));
    // if (!debug.binary.empty() && debug.binary.total() > 0)
    //   cv::imshow("Debug: Binary", for_display(debug.binary));
    // if (!debug.sure_fg.empty() && debug.sure_fg.total() > 0)
    //   cv::imshow("Debug: Sure FG", for_display(debug.sure_fg));
    // if (!debug.dist_vis.empty() && debug.dist_vis.total() > 0)
    //   cv::imshow("Debug: Distance", for_display(debug.dist_vis));
    return true;
  }

}

int main()
{
  cv::VideoCapture cap(2, cv::CAP_V4L2);
  if (!cap.isOpened())
  {
    std::cerr << "Could not open camera (index 2). Check device or try another index.\n";
    return 1;
  }
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

  create_all_windows_once();

  double ratio_px_to_mm = get_ratio_px_to_mm();
  int width_px = static_cast<int>(coin::Config::PAPER_WIDTH_MM * coin::Config::SCALE_FACTOR);
  int height_px = static_cast<int>(coin::Config::PAPER_HEIGHT_MM * coin::Config::SCALE_FACTOR);

  std::cout << "Using px-to-mm ratio: " << ratio_px_to_mm << " mm/px (from SCALE_FACTOR).\n";

  coin::CornerStabilizer stabilizer(coin::Config::STABILIZER_WINDOW);
  coin::CoinTracker tracker;
  cv::Mat dst_corners = (cv::Mat_<float>(4, 2) << 0, 0, width_px - 1, 0, width_px - 1, height_px - 1, 0, height_px - 1);

  coin::SVMClassifier svm;
  int classifier_index = read_default_classifier_index();
#ifdef COIN_USE_TORCH
  coin::TorchClassifier torch_cnn, torch_resnet;
  coin::TorchClassifier *active_torch = nullptr;
#endif
  auto load_classifier = [&svm](int idx)
  {
    if (idx < 0 || idx > 3)
      return true;
    return svm.load(coin::Config::CLASSIFIER_MODEL_PATHS[idx],
                    coin::Config::SVM_SCALER_PATH,
                    coin::Config::CLASSIFIER_NAMES[idx]);
  };
  auto ensure_classifier = [&]()
  {
#ifdef COIN_USE_TORCH
    active_torch = nullptr;
#endif
    if (classifier_index <= 3)
    {
      if (!load_classifier(classifier_index))
        classifier_index = 0;
      if (!load_classifier(classifier_index))
      {
        std::cerr << "Could not load classifier. Run train_svm from build dir first.\n";
        return false;
      }
    }
#ifdef COIN_USE_TORCH
    else if (classifier_index == 4)
    {
      if (!torch_cnn.is_loaded() && !torch_cnn.load(coin::Config::COIN_CNN_TRACED_PATH))
      {
        std::cerr << "Could not load CNN. Run: python export_torchscript.py\n";
        return false;
      }
      active_torch = &torch_cnn;
    }
    else if (classifier_index == 5)
    {
      if (!torch_resnet.is_loaded() && !torch_resnet.load(coin::Config::COIN_RESNET18_TRACED_PATH))
      {
        std::cerr << "Could not load ResNet18. Run: python export_torchscript.py\n";
        return false;
      }
      active_torch = &torch_resnet;
    }
#endif
    return true;
  };
  if (!ensure_classifier())
    return 1;
  std::cout << "Classifier: " << coin::Config::CLASSIFIER_NAMES[classifier_index]
#ifdef COIN_USE_TORCH
            << " (press 1-6 to switch)\n";
#else
            << " (press 1-4 to switch)\n";
#endif

  while (true)
  {
    double t_frame_start = cv::getTickCount();
    cv::Mat frame;
    if (!cap.read(frame))
    {
      std::cerr << "Camera read failed (disconnected or EOF).\n";
      break;
    }
    if (frame.empty())
    {
      std::cerr << "Dropping empty frame.\n";
      continue;
    }

    try
    {
      auto raw_corners = coin::find_paper_corners(frame);
      std::optional<cv::Mat> stable_corners = raw_corners.has_value()
                                                  ? stabilizer.update(&*raw_corners)
                                                  : stabilizer.update(nullptr);

      if (stable_corners.has_value() && !stable_corners->empty())
      {
        cv::Mat rect = coin::order_corners(*stable_corners);
        cv::Mat M = cv::getPerspectiveTransform(rect, dst_corners);
        double det = cv::determinant(M);
        if (std::abs(det) > 1e-6)
        {
          cv::Mat warped;
          cv::warpPerspective(frame, warped, M, cv::Size(width_px, height_px));
          if (!warped.empty())
          {
#ifdef COIN_USE_TORCH
            run_coin_detection(warped, ratio_px_to_mm, tracker, svm,
                               coin::Config::CLASSIFIER_NAMES[classifier_index],
                               classifier_index, active_torch);
#else
            run_coin_detection(warped, ratio_px_to_mm, tracker, svm,
                               coin::Config::CLASSIFIER_NAMES[classifier_index],
                               0, nullptr);
#endif
            cv::Mat warped_small;
            cv::resize(warped, warped_small, cv::Size(), 0.5, 0.5);
            //cv::imshow("Warped", for_display(warped_small));
          }
        }
      }

      if (raw_corners.has_value())
      {
        std::vector<std::vector<cv::Point>> contour(1);
        for (int i = 0; i < 4; ++i)
          contour[0].emplace_back(static_cast<int>(raw_corners->at<float>(i, 0)), static_cast<int>(raw_corners->at<float>(i, 1)));
        cv::drawContours(frame, contour, -1, cv::Scalar(0, 255, 0), 2);
      }
      //double fps = cv::getTickFrequency() / (cv::getTickCount() - t_frame_start);
      //cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(std::round(fps))),
      //            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
      //cv::imshow("Scanner", for_display(frame));
    }
    catch (const cv::Exception &e)
    {
      std::cerr << "OpenCV error (frame skipped): " << e.what() << "\n";
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error (frame skipped): " << e.what() << "\n";
    }

    int key = cv::waitKey(1);
    if (key == 'q')
      break;
#ifdef COIN_USE_TORCH
    if (key >= '1' && key <= '6')
    {
      classifier_index = key - '1';
      if (ensure_classifier())
        std::cout << "Switched to " << coin::Config::CLASSIFIER_NAMES[classifier_index] << "\n";
    }
#else
    if (key == '1' || key == '2' || key == '3' || key == '4')
    {
      int new_idx = key - '1';
      if (load_classifier(new_idx))
      {
        classifier_index = new_idx;
        std::cout << "Switched to " << coin::Config::CLASSIFIER_NAMES[classifier_index] << "\n";
      }
    }
#endif
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
