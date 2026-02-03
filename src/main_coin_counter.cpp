#include "config.hpp"
#include "calibration.hpp"
#include "svm_classifier.hpp"
#include "corner_stabilizer.hpp"
#include "coin_tracker.hpp"
#include "coin_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>

namespace
{

  double get_ratio_px_to_mm()
  {
    return 1.0 / coin::Config::SCALE_FACTOR;
  }

  int read_default_classifier_index()
  {
    std::ifstream f(coin::Config::CLASSIFIER_DEFAULT_FILE);
    int idx = 0;
    if (f && (f >> idx) && idx >= 0 && idx <= 3)
      return idx;
    return 0;
  }

  cv::Mat draw_coins_and_histogram(const cv::Mat &frame, coin::CoinTracker &tracker,
                                   double ratio_px_to_mm, coin::SVMClassifier &svm, bool show_debug,
                                   const std::string &classifier_name = "SVM")
  {
    cv::Mat display;
    frame.copyTo(display);
    auto entries = tracker.get_stable_entries();
    auto rows = coin::collect_coin_features(frame, entries, ratio_px_to_mm);
    double total_eur = 0.0;

    for (size_t i = 0; i < entries.size(); ++i)
    {
      const auto &e = entries[i];
      int r = coin::diameter_mm_to_radius_px(e.second, ratio_px_to_mm);
      double display_diameter_mm = e.second;
      cv::Scalar color(0, 255, 0);
      if (i < rows.size())
      {
        int cid = svm.predict(rows[i].diameter_mm, rows[i].L, rows[i].a, rows[i].b);
        cid = cid % 6;
        color = cv::Scalar(coin::Config::CLUSTER_COLORS_BGR[cid][0],
                           coin::Config::CLUSTER_COLORS_BGR[cid][1],
                           coin::Config::CLUSTER_COLORS_BGR[cid][2]);
        total_eur += coin::Config::CLASS_TO_VALUE_EUR[cid];
      }
      cv::circle(display, e.first, r, color, 2);
      std::string label = std::to_string(static_cast<int>(display_diameter_mm * 10) / 10.0).substr(0, 4) + "mm";
      cv::putText(display, label, cv::Point(e.first.x - 20, e.first.y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }

    cv::rectangle(display, cv::Point(10, 10), cv::Point(280, 90), cv::Scalar(0, 0, 0), -1);
    cv::rectangle(display, cv::Point(10, 10), cv::Point(280, 90), cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Coins: " + std::to_string(entries.size()), cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "Total: " << total_eur << " EUR";
    cv::putText(display, oss.str(), cv::Point(20, 62), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Clf: " + classifier_name + " (1-4)", cv::Point(20, 82),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

    if (show_debug && !entries.empty())
    {
      std::vector<double> diameters;
      for (const auto &e : entries)
        diameters.push_back(e.second);
      double min_d = *std::min_element(diameters.begin(), diameters.end());
      double max_d = *std::max_element(diameters.begin(), diameters.end());
      int n_bins = static_cast<int>((coin::Config::HIST_BIN_MAX - coin::Config::HIST_BIN_MIN) / coin::Config::HIST_BIN_STEP);
      std::vector<int> counts(n_bins, 0);
      for (double d : diameters)
      {
        int bin = static_cast<int>((d - coin::Config::HIST_BIN_MIN) / coin::Config::HIST_BIN_STEP);
        if (bin >= 0 && bin < n_bins)
          counts[bin]++;
      }
      int max_count = *std::max_element(counts.begin(), counts.end());
      if (max_count == 0)
        max_count = 1;
      cv::Mat hist_img(coin::Config::HIST_HEIGHT, coin::Config::HIST_WIDTH, CV_8UC3, cv::Scalar(255, 255, 255));
      int bar_w = std::max(1, (coin::Config::HIST_WIDTH - 40) / n_bins - 2);
      for (int i = 0; i < n_bins; ++i)
      {
        int bar_h = static_cast<int>((counts[i] / static_cast<double>(max_count)) * (coin::Config::HIST_HEIGHT - 50));
        int x1 = 30 + i * (bar_w + 2);
        int y1 = coin::Config::HIST_HEIGHT - 30 - bar_h;
        int x2 = x1 + bar_w;
        int y2 = coin::Config::HIST_HEIGHT - 30;
        cv::rectangle(hist_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(180, 130, 70), -1);
        cv::rectangle(hist_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(50, 50, 50), 1);
      }
      cv::putText(hist_img, "Diameter (mm)", cv::Point(coin::Config::HIST_WIDTH / 2 - 50, coin::Config::HIST_HEIGHT - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
      cv::putText(hist_img, "n=" + std::to_string(diameters.size()), cv::Point(10, 20),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
      cv::imshow("Diameter histogram", hist_img);
    }
    else if (show_debug)
    {
      cv::Mat hist_img(coin::Config::HIST_HEIGHT, coin::Config::HIST_WIDTH, CV_8UC3, cv::Scalar(255, 255, 255));
      cv::putText(hist_img, "No diameters", cv::Point(coin::Config::HIST_WIDTH / 2 - 50, coin::Config::HIST_HEIGHT / 2 - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(100, 100, 100), 1);
      cv::imshow("Diameter histogram", hist_img);
    }

    return display;
  }

  bool run_coin_detection(const cv::Mat &warped, double ratio_px_to_mm,
                          coin::CoinTracker &tracker, coin::SVMClassifier &svm, bool show_debug, const std::string &classifier_name)
  {
    const double keep_frac = 0.8;
    int cw = static_cast<int>(warped.cols * keep_frac);
    int ch = static_cast<int>(warped.rows * keep_frac);
    int cx = (warped.cols - cw) / 2;
    int cy = (warped.rows - ch) / 2;
    cv::Rect roi(cx, cy, cw, ch);
    cv::Mat warped_crop = warped(roi).clone();

    auto detections = coin::detect_and_measure_coins(warped_crop, ratio_px_to_mm);
    for (auto &d : detections)
      d.center += cv::Point2i(roi.x, roi.y);
    tracker.update(detections);
    cv::Mat display = draw_coins_and_histogram(warped, tracker, ratio_px_to_mm, svm, show_debug, classifier_name);
    cv::imshow("Anti-Glare Detection", display);
    // if (show_debug)
    // {
    //   cv::Mat blurred = coin::preprocess_for_circles(warped);
    //   cv::imshow("Preprocess", blurred);
    //   cv::Mat edges;
    //   cv::Canny(blurred, edges, coin::Config::CANNY_THRESHOLD1, coin::Config::CANNY_THRESHOLD2);
    //   cv::imshow("Canny", edges);
    // }
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

  double ratio_px_to_mm = get_ratio_px_to_mm();
  int width_px = static_cast<int>(coin::Config::PAPER_WIDTH_MM * coin::Config::SCALE_FACTOR);
  int height_px = static_cast<int>(coin::Config::PAPER_HEIGHT_MM * coin::Config::SCALE_FACTOR);

  std::cout << "Using px-to-mm ratio: " << ratio_px_to_mm << " mm/px (from SCALE_FACTOR).\n";

  coin::CornerStabilizer stabilizer(coin::Config::STABILIZER_WINDOW);
  coin::CoinTracker tracker;
  cv::Mat dst_corners = (cv::Mat_<float>(4, 2) << 0, 0, width_px - 1, 0, width_px - 1, height_px - 1, 0, height_px - 1);

  coin::SVMClassifier svm;
  int classifier_index = read_default_classifier_index();
  auto load_classifier = [&svm](int idx)
  {
    return svm.load(coin::Config::CLASSIFIER_MODEL_PATHS[idx],
                    coin::Config::SVM_SCALER_PATH,
                    coin::Config::CLASSIFIER_NAMES[idx]);
  };
  if (!load_classifier(classifier_index))
    classifier_index = 0;
  if (!load_classifier(classifier_index))
  {
    std::cerr << "Could not load classifier. Run train_svm from build dir first.\n";
    return 1;
  }
  std::cout << "Classifier: " << coin::Config::CLASSIFIER_NAMES[classifier_index]
            << " (press 1-4 to switch)\n";

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
            run_coin_detection(warped, ratio_px_to_mm, tracker, svm, true,
                               coin::Config::CLASSIFIER_NAMES[classifier_index]);
            cv::Mat warped_small;
            cv::resize(warped, warped_small, cv::Size(), 0.5, 0.5);
            cv::imshow("Warped", warped_small);
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
      double fps = cv::getTickFrequency() / (cv::getTickCount() - t_frame_start);
      cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(std::round(fps))),
                  cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
      cv::imshow("Scanner", frame);
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
    if (key == '1' || key == '2' || key == '3' || key == '4')
    {
      int new_idx = key - '1';
      if (load_classifier(new_idx))
      {
        classifier_index = new_idx;
        std::cout << "Switched to " << coin::Config::CLASSIFIER_NAMES[classifier_index] << "\n";
      }
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
