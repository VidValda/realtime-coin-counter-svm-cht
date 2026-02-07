#include "config.hpp"
#include "calibration.hpp"
#include "corner_stabilizer.hpp"
#include "coin_tracker.hpp"
#include "coin_detector.hpp"
#include "torch_classifier.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <chrono>

namespace
{
  using Clock = std::chrono::high_resolution_clock;
  using Ms = std::chrono::duration<double, std::milli>;

  struct PipelineTimings
  {
    double capture_ms = 0;
    double paper_ms = 0;
    double stabilizer_ms = 0;
    double order_warp_ms = 0;   /* order_corners + getPerspectiveTransform + warpPerspective */
    double det_prep_ms = 0;     /* resize/clone for detection input */
    double detect_coins_ms = 0;
    double tracker_ms = 0;
    double torch_ms = 0;       /* predict_batch inside draw_coins */
    double draw_rest_ms = 0;    /* copy, circles, text, no torch */
    double display_ms = 0;     /* for_display + imshow */
    double total_frame_ms = 0;
  };

  void print_timings(const PipelineTimings &t, int frame_id)
  {
    std::cout << std::fixed << std::setprecision(2)
              << "frame " << frame_id
              << " | capture=" << t.capture_ms << " ms"
              << " | paper=" << t.paper_ms << " ms"
              << " | stabilizer=" << t.stabilizer_ms << " ms"
              << " | order+warp=" << t.order_warp_ms << " ms"
              << " | det_prep=" << t.det_prep_ms << " ms"
              << " | detect_coins=" << t.detect_coins_ms << " ms"
              << " | tracker=" << t.tracker_ms << " ms"
              << " | torch=" << t.torch_ms << " ms"
              << " | draw_rest=" << t.draw_rest_ms << " ms"
              << " | display=" << t.display_ms << " ms"
              << " | TOTAL=" << t.total_frame_ms << " ms"
              << " (" << (1000.0 / t.total_frame_ms) << " FPS)\n";
  }

  cv::Mat for_display(const cv::Mat &mat)
  {
    if (mat.cols <= coin::Config::MAX_DISPLAY_WIDTH_PX || mat.empty())
      return mat;
    double scale = static_cast<double>(coin::Config::MAX_DISPLAY_WIDTH_PX) / mat.cols;
    cv::Mat out;
    cv::resize(mat, out, cv::Size(), scale, scale, cv::INTER_AREA);
    return out;
  }

  void create_all_windows_once()
  {
    cv::namedWindow("Anti-Glare Detection", cv::WINDOW_AUTOSIZE);
    if (coin::Config::SHOW_DEBUG_VIEWS)
    {
      cv::namedWindow("Debug: Markers", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("Debug: Segmentation", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("Debug: Binary", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("Debug: Sure FG", cv::WINDOW_AUTOSIZE);
      cv::namedWindow("Debug: Distance", cv::WINDOW_AUTOSIZE);
    }
  }

  double get_ratio_px_to_mm()
  {
    return 1.0 / coin::Config::SCALE_FACTOR;
  }

  /** DL-only: no feature calculation; classify with Torch batch inference. */
  cv::Mat draw_coins(const cv::Mat &frame, coin::CoinTracker &tracker,
                     double ratio_px_to_mm, coin::TorchClassifier *torch_clf,
                     const std::string &classifier_name, PipelineTimings *out_timings = nullptr)
  {
    auto t_draw_start = Clock::now();
    cv::Mat display;
    frame.copyTo(display);
    auto entries = tracker.get_stable_entries();
    double total_eur = 0.0;

    std::vector<int> cids;
    if (torch_clf && torch_clf->is_loaded() && !entries.empty())
    {
      std::vector<std::pair<cv::Point2i, int>> centers_radii;
      centers_radii.reserve(entries.size());
      for (const auto &e : entries)
        centers_radii.emplace_back(e.first, coin::diameter_mm_to_radius_px(e.second, ratio_px_to_mm));
      auto t_torch_start = Clock::now();
      cids = torch_clf->predict_batch(frame, centers_radii);
      if (out_timings)
        out_timings->torch_ms = Ms(Clock::now() - t_torch_start).count();
    }

    for (size_t i = 0; i < entries.size(); ++i)
    {
      const auto &e = entries[i];
      int r = coin::diameter_mm_to_radius_px(e.second, ratio_px_to_mm);
      double display_diameter_mm = e.second;
      int cid = (i < cids.size()) ? (cids[i] % 6) : 0;
      cv::Scalar color(coin::Config::CLUSTER_COLORS_BGR[cid][0],
                       coin::Config::CLUSTER_COLORS_BGR[cid][1],
                       coin::Config::CLUSTER_COLORS_BGR[cid][2]);
      total_eur += coin::Config::CLASS_TO_VALUE_EUR[cid];
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
    cv::putText(display, "Clf: " + classifier_name + " (1=CNN 2=ResNet)", cv::Point(20, 82),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 2);

    if (out_timings)
      out_timings->draw_rest_ms = Ms(Clock::now() - t_draw_start).count() - out_timings->torch_ms;
    return display;
  }

  bool run_coin_detection(const cv::Mat &warped, double ratio_px_to_mm,
                          coin::CoinTracker &tracker, coin::TorchClassifier *torch_clf,
                          const std::string &classifier_name, PipelineTimings *out_timings = nullptr)
  {
    const double keep_frac = 1;
    int cw = static_cast<int>(warped.cols * keep_frac);
    int ch = static_cast<int>(warped.rows * keep_frac);
    int cx = (warped.cols - cw) / 2;
    int cy = (warped.rows - ch) / 2;
    cv::Rect roi(cx, cy, cw, ch);
    const bool full_roi = (cw == warped.cols && ch == warped.rows);

    auto t0 = Clock::now();
    cv::Mat det_input = full_roi ? warped : warped(roi).clone();
    if (out_timings)
      out_timings->det_prep_ms = Ms(Clock::now() - t0).count();

    const bool show_debug = coin::Config::SHOW_DEBUG_VIEWS;
    coin::DebugViews debug;
    coin::DebugViews *out_debug = show_debug ? &debug : nullptr;
    const double det_scale = coin::Config::COIN_DETECT_SCALE;
    std::vector<coin::Detection> detections;

    t0 = Clock::now();
    if (det_scale <= 0 || det_scale >= 1.0)
    {
      detections = coin::detect_and_measure_coins(det_input, ratio_px_to_mm, out_debug);
    }
    else
    {
      cv::Mat small;
      cv::resize(det_input, small, cv::Size(), det_scale, det_scale, cv::INTER_LINEAR);
      double ratio_small = ratio_px_to_mm / det_scale;
      detections = coin::detect_and_measure_coins(small, ratio_small, out_debug, det_scale);
      const double inv = 1.0 / det_scale;
      for (auto &d : detections)
      {
        d.center.x = static_cast<int>(std::round(d.center.x * inv));
        d.center.y = static_cast<int>(std::round(d.center.y * inv));
      }
    }
    if (out_timings)
      out_timings->detect_coins_ms = Ms(Clock::now() - t0).count();

    t0 = Clock::now();
    for (auto &d : detections)
      d.center += cv::Point2i(roi.x, roi.y);
    tracker.update(detections);
    if (out_timings)
      out_timings->tracker_ms = Ms(Clock::now() - t0).count();

    t0 = Clock::now();
    cv::Mat display = draw_coins(warped, tracker, ratio_px_to_mm, torch_clf, classifier_name, out_timings);
    {
      static int64_t prev_ticks = cv::getTickCount();
      int64_t ticks = cv::getTickCount();
      double fps = cv::getTickFrequency() / std::max(ticks - prev_ticks, int64_t(1));
      prev_ticks = ticks;
      cv::putText(display, "FPS: " + std::to_string(static_cast<int>(std::round(fps))),
                  cv::Point(display.cols - 200, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
    }
    auto t_disp = Clock::now();
    cv::imshow("Anti-Glare Detection", for_display(display));
    if (out_timings)
      out_timings->display_ms = Ms(Clock::now() - t_disp).count();

    if (show_debug)
    {
      if (!debug.markers_vis.empty() && debug.markers_vis.total() > 0)
        cv::imshow("Debug: Markers", for_display(debug.markers_vis));
      if (!debug.segmentation.empty() && debug.segmentation.total() > 0)
        cv::imshow("Debug: Segmentation", for_display(debug.segmentation));
      if (!debug.binary.empty() && debug.binary.total() > 0)
        cv::imshow("Debug: Binary", for_display(debug.binary));
      if (!debug.sure_fg.empty() && debug.sure_fg.total() > 0)
        cv::imshow("Debug: Sure FG", for_display(debug.sure_fg));
      if (!debug.dist_vis.empty() && debug.dist_vis.total() > 0)
        cv::imshow("Debug: Distance", for_display(debug.dist_vis));
    }
    return true;
  }
}

int main()
{
  cv::setUseOptimized(true);

  const char *test_videos_init[] = {coin::Config::TEST_VIDEO_1, coin::Config::TEST_VIDEO_2};
  const int num_test_videos_init = sizeof(test_videos_init) / sizeof(test_videos_init[0]);
  int first_video_index = 0;

  cv::VideoCapture cap;
  if (coin::Config::USE_TEST_VIDEOS)
  {
    for (; first_video_index < num_test_videos_init; ++first_video_index)
    {
      cap.open(test_videos_init[first_video_index]);
      if (cap.isOpened())
      {
        std::cout << "Using test video: " << test_videos_init[first_video_index] << "\n";
        break;
      }
    }
    if (!cap.isOpened())
    {
      std::cerr << "Could not open test videos.\n";
      return 1;
    }
  }
  else
  {
    cap.open(2, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
      std::cerr << "Could not open camera (index 2). Check device or try another index.\n";
      return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  }

  create_all_windows_once();

  double ratio_px_to_mm = get_ratio_px_to_mm();
  int width_px = static_cast<int>(coin::Config::PAPER_WIDTH_MM * coin::Config::SCALE_FACTOR);
  int height_px = static_cast<int>(coin::Config::PAPER_HEIGHT_MM * coin::Config::SCALE_FACTOR);

  std::cout << "coin_counter_dl: DL-only pipeline (no feature extraction).\n";
  std::cout << "Using px-to-mm ratio: " << ratio_px_to_mm << " mm/px.\n";

  coin::CornerStabilizer stabilizer(coin::Config::STABILIZER_WINDOW);
  coin::CoinTracker tracker;
  cv::Mat dst_corners = (cv::Mat_<float>(4, 2) << 0, 0, width_px - 1, 0, width_px - 1, height_px - 1, 0, height_px - 1);

  static constexpr const char *DL_NAMES[2] = {"CNN", "ResNet18"};
  static constexpr const char *DL_PATHS[2] = {
      coin::Config::COIN_CNN_TRACED_PATH,
      coin::Config::COIN_RESNET18_TRACED_PATH};
  coin::TorchClassifier torch_cnn, torch_resnet;
  int classifier_index = 0; // 0 = CNN, 1 = ResNet18

  auto ensure_classifier = [&]() -> bool
  {
    if (classifier_index == 0)
    {
      if (!torch_cnn.is_loaded() && !torch_cnn.load(DL_PATHS[0]))
      {
        std::cerr << "Could not load CNN. Run: python export_torchscript.py\n";
        return false;
      }
      return true;
    }
    if (classifier_index == 1)
    {
      if (!torch_resnet.is_loaded() && !torch_resnet.load(DL_PATHS[1]))
      {
        std::cerr << "Could not load ResNet18. Run: python export_torchscript.py\n";
        return false;
      }
      return true;
    }
    return false;
  };

  if (!ensure_classifier())
    return 1;
  coin::TorchClassifier *active_torch = (classifier_index == 0) ? &torch_cnn : &torch_resnet;
  std::cout << "Classifier: " << DL_NAMES[classifier_index] << " (press 1=CNN 2=ResNet, q=quit)\n";
  std::cout << "Pipeline timings (ms) printed every frame. Header re-printed every 60 frames.\n\n";

  int frame_count = 0;
  std::optional<cv::Mat> last_raw_corners;

  const char *test_videos[] = {coin::Config::TEST_VIDEO_1, coin::Config::TEST_VIDEO_2};
  const int num_test_videos = sizeof(test_videos) / sizeof(test_videos[0]);
  int current_video_index = coin::Config::USE_TEST_VIDEOS ? first_video_index : -1;

  while (true)
  {
    auto frame_start = Clock::now();
    PipelineTimings timings = {};

    cv::Mat frame;
    auto t_cap = Clock::now();
    if (!cap.read(frame))
    {
      if (coin::Config::USE_TEST_VIDEOS && current_video_index >= 0 && current_video_index + 1 < num_test_videos)
      {
        cap.release();
        ++current_video_index;
        cap.open(test_videos[current_video_index]);
        if (cap.isOpened())
        {
          std::cout << "Next test video: " << test_videos[current_video_index] << "\n";
          continue;
        }
      }
      std::cerr << "Camera/video read failed.\n";
      break;
    }
    if (frame.empty())
      continue;
    timings.capture_ms = Ms(Clock::now() - t_cap).count();

    try
    {
      const int every_n = std::max(1, coin::Config::PAPER_DETECT_EVERY_N_FRAMES);
      std::optional<cv::Mat> raw_corners;
      auto t_paper = Clock::now();
      if (every_n <= 1 || (++frame_count % every_n) == 1)
      {
        raw_corners = coin::find_paper_corners(frame);
        if (raw_corners.has_value())
          last_raw_corners = raw_corners;
      }
      else if (last_raw_corners.has_value())
        raw_corners = last_raw_corners;
      timings.paper_ms = Ms(Clock::now() - t_paper).count();

      auto t_stab = Clock::now();
      std::optional<cv::Mat> stable_corners = raw_corners.has_value()
                                                  ? stabilizer.update(&*raw_corners)
                                                  : stabilizer.update(nullptr);
      timings.stabilizer_ms = Ms(Clock::now() - t_stab).count();

      if (stable_corners.has_value() && !stable_corners->empty() &&
          stable_corners->rows == 4 && stable_corners->cols >= 2)
      {
        auto t_ow = Clock::now();
        cv::Mat rect = coin::order_corners(*stable_corners);
        cv::Mat M = cv::getPerspectiveTransform(rect, dst_corners);
        if (std::abs(cv::determinant(M)) > 1e-6)
        {
          cv::Mat warped;
          cv::warpPerspective(frame, warped, M, cv::Size(width_px, height_px));
          timings.order_warp_ms = Ms(Clock::now() - t_ow).count();
          if (!warped.empty() && warped.rows > 0 && warped.cols > 0)
          {
            active_torch = (classifier_index == 0) ? &torch_cnn : &torch_resnet;
            run_coin_detection(warped, ratio_px_to_mm, tracker, active_torch, DL_NAMES[classifier_index], &timings);
          }
        }
        else
          timings.order_warp_ms = Ms(Clock::now() - t_ow).count();
      }
      else
        timings.order_warp_ms = 0;

      if (raw_corners.has_value())
      {
        std::vector<std::vector<cv::Point>> contour(1);
        for (int i = 0; i < 4; ++i)
          contour[0].emplace_back(static_cast<int>(raw_corners->at<float>(i, 0)), static_cast<int>(raw_corners->at<float>(i, 1)));
        cv::drawContours(frame, contour, -1, cv::Scalar(0, 255, 0), 2);
      }

      timings.total_frame_ms = Ms(Clock::now() - frame_start).count();
      if (frame_count % 60 == 1)
        std::cout << "frame    | capture | paper | stabilizer | order+warp | det_prep | detect_coins | tracker | torch | draw_rest | display | TOTAL (ms) | FPS\n";
      print_timings(timings, frame_count);
    }
    catch (const cv::Exception &e)
    {
      std::cerr << "OpenCV error: " << e.what() << "\n";
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error: " << e.what() << "\n";
    }

    int key = cv::waitKey(1);
    if (key == 'q')
      break;
    if (key == '1' || key == '2')
    {
      classifier_index = key - '1';
      if (ensure_classifier())
      {
        active_torch = (classifier_index == 0) ? &torch_cnn : &torch_resnet;
        std::cout << "Switched to " << DL_NAMES[classifier_index] << "\n";
      }
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
