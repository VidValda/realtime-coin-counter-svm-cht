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
    double order_warp_ms = 0;
    double detect_coins_ms = 0;
    double tracker_ms = 0;
    double classify_ms = 0;
    double draw_rest_ms = 0;
    double display_ms = 0;
    double total_frame_ms = 0;
  };

  static bool s_print_timings = coin::Config::PRINT_TIMINGS_DEFAULT;

  void print_timings(const PipelineTimings &t, int frame_id)
  {
    std::cerr << std::fixed << std::setprecision(2)
              << "frame " << frame_id
              << " | capture=" << t.capture_ms << " ms"
              << " | paper=" << t.paper_ms << " ms"
              << " | stabilizer=" << t.stabilizer_ms << " ms"
              << " | order+warp=" << t.order_warp_ms << " ms"
              << " | detect_coins=" << t.detect_coins_ms << " ms"
              << " | tracker=" << t.tracker_ms << " ms"
              << " | classify=" << t.classify_ms << " ms"
              << " | draw_rest=" << t.draw_rest_ms << " ms"
              << " | display=" << t.display_ms << " ms"
              << " | TOTAL=" << t.total_frame_ms << " ms"
              << " (" << (1000.0 / std::max(t.total_frame_ms, 0.001)) << " FPS)\n";
  }
  /** Resize mat for display if wider than MAX_DISPLAY_WIDTH_PX to reduce memory and GUI pressure. */
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

  /** Cached classification results to avoid re-classifying on skip frames. */
  struct ClassificationCache
  {
    std::vector<int> class_ids;
    double total_eur = 0.0;
    size_t num_entries = 0;
    bool valid = false;
  };
  static ClassificationCache s_clf_cache;

  cv::Mat draw_coins(const cv::Mat &frame, coin::CoinTracker &tracker,
                     double ratio_px_to_mm, coin::SVMClassifier &svm,
                     const std::string &classifier_name,
#ifdef COIN_USE_TORCH
                     int classifier_index, coin::TorchClassifier *torch_clf,
#else
                     int /* classifier_index */, void *torch_clf,
#endif
                     bool reclassify, PipelineTimings *out_timings = nullptr)
  {
    (void)torch_clf;
    auto t_draw_start = Clock::now();
    cv::Mat display;
    frame.copyTo(display);
    auto entries = tracker.get_stable_entries();

#ifdef COIN_USE_TORCH
    bool use_torch = (classifier_index >= 4 && torch_clf && torch_clf->is_loaded());
#else
    bool use_torch = false;
#endif

    if (reclassify || !s_clf_cache.valid || s_clf_cache.num_entries != entries.size())
    {
      s_clf_cache.class_ids.resize(entries.size(), 0);
      s_clf_cache.total_eur = 0.0;
      s_clf_cache.num_entries = entries.size();

      auto t_clf_start = Clock::now();

#ifdef COIN_USE_TORCH
      std::vector<int> torch_cids;
      if (use_torch && !entries.empty())
      {
        std::vector<std::pair<cv::Point2i, int>> centers_radii;
        centers_radii.reserve(entries.size());
        for (const auto &e : entries)
          centers_radii.emplace_back(e.first, coin::diameter_mm_to_radius_px(e.second, ratio_px_to_mm));
        torch_cids = torch_clf->predict_batch(frame, centers_radii);
      }
#endif

      if (!use_torch)
      {
        auto rows = coin::collect_coin_features(frame, entries, ratio_px_to_mm);
        for (size_t i = 0; i < entries.size(); ++i)
        {
          int cid = 0;
          if (i < rows.size())
            cid = svm.predict(rows[i].diameter_mm, rows[i].L, rows[i].a, rows[i].b) % 6;
          s_clf_cache.class_ids[i] = cid;
          s_clf_cache.total_eur += coin::Config::CLASS_TO_VALUE_EUR[cid];
        }
      }
      else
      {
#ifdef COIN_USE_TORCH
        for (size_t i = 0; i < entries.size(); ++i)
        {
          int cid = (i < torch_cids.size()) ? (torch_cids[i] % 6) : 0;
          s_clf_cache.class_ids[i] = cid;
          s_clf_cache.total_eur += coin::Config::CLASS_TO_VALUE_EUR[cid];
        }
#endif
      }
      s_clf_cache.valid = true;
      if (out_timings)
        out_timings->classify_ms = Ms(Clock::now() - t_clf_start).count();
    }

    for (size_t i = 0; i < entries.size(); ++i)
    {
      const auto &e = entries[i];
      int r = coin::diameter_mm_to_radius_px(e.second, ratio_px_to_mm);
      int cid = (i < s_clf_cache.class_ids.size()) ? s_clf_cache.class_ids[i] : 0;
      cv::Scalar color(coin::Config::CLUSTER_COLORS_BGR[cid][0],
                       coin::Config::CLUSTER_COLORS_BGR[cid][1],
                       coin::Config::CLUSTER_COLORS_BGR[cid][2]);
      cv::circle(display, e.first, r, color, 4);
      std::string label = std::to_string(static_cast<int>(e.second * 10) / 10.0).substr(0, 4) + "mm";
      cv::putText(display, label, cv::Point(e.first.x - 20, e.first.y - 10),
                  cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 4);
    }

    cv::rectangle(display, cv::Point(10, 10), cv::Point(280, 90), cv::Scalar(0, 0, 0), -1);
    cv::rectangle(display, cv::Point(10, 10), cv::Point(280, 90), cv::Scalar(255, 255, 255), 2);
    cv::putText(display, "Coins: " + std::to_string(entries.size()), cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "Total: " << s_clf_cache.total_eur << " EUR";
    cv::putText(display, oss.str(), cv::Point(20, 62), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
#ifdef COIN_USE_TORCH
    cv::putText(display, "Clf: " + classifier_name + " (1-6)", cv::Point(20, 82),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 2);
#else
    cv::putText(display, "Clf: " + classifier_name + " (1-4)", cv::Point(20, 82),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 2);
#endif

    if (out_timings)
      out_timings->draw_rest_ms = Ms(Clock::now() - t_draw_start).count() - out_timings->classify_ms;
    return display;
  }

  bool run_coin_detection(const cv::Mat &warped, double ratio_px_to_mm,
                          coin::CoinTracker &tracker, coin::SVMClassifier &svm, const std::string &classifier_name,
#ifdef COIN_USE_TORCH
                          int classifier_index, coin::TorchClassifier *torch_clf,
#else
                          int /* classifier_index */, void * /* torch_clf */,
#endif
                          bool skip_detection = false, PipelineTimings *out_timings = nullptr)
  {
    if (!skip_detection)
    {
      const double det_scale = coin::Config::COIN_DETECT_SCALE;
      const bool show_debug = coin::Config::SHOW_DEBUG_VIEWS;
      coin::DebugViews debug;
      coin::DebugViews *out_debug = show_debug ? &debug : nullptr;
      std::vector<coin::Detection> detections;

      auto t0 = Clock::now();
      if (det_scale <= 0 || det_scale >= 1.0)
      {
        detections = coin::detect_and_measure_coins(warped, ratio_px_to_mm, out_debug);
      }
      else
      {
        cv::Mat small;
        cv::resize(warped, small, cv::Size(), det_scale, det_scale, cv::INTER_LINEAR);
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
      tracker.update(detections);
      if (out_timings)
        out_timings->tracker_ms = Ms(Clock::now() - t0).count();

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
    }

#ifdef COIN_USE_TORCH
    cv::Mat display = draw_coins(warped, tracker, ratio_px_to_mm, svm, classifier_name, classifier_index, torch_clf, !skip_detection, out_timings);
#else
    cv::Mat display = draw_coins(warped, tracker, ratio_px_to_mm, svm, classifier_name, 0, nullptr, !skip_detection, out_timings);
#endif
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
      std::cerr << "Could not open test videos (tried " << coin::Config::TEST_VIDEO_1
                << ", " << coin::Config::TEST_VIDEO_2 << ").\n";
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
            << " (press 1-6 to switch, t=timings, q=quit)\n";
#else
            << " (press 1-4 to switch, t=timings, q=quit)\n";
#endif
  std::cout << "Timings: " << (s_print_timings ? "ON" : "OFF") << " (press 't' to toggle)\n" << std::endl;

  int frame_count = 0;
  int coin_detect_count = 0;
  std::optional<cv::Mat> last_raw_corners;
  cv::Mat warped(height_px, width_px, CV_8UC3);
  cv::Mat cached_M;

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
      std::cerr << "Camera/video read failed (disconnected or EOF).\n";
      break;
    }
    if (frame.empty())
    {
      std::cerr << "Dropping empty frame.\n";
      continue;
    }
    timings.capture_ms = Ms(Clock::now() - t_cap).count();

    try
    {
      // Run expensive paper (LSD) detection only every N frames; reuse last when stable
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
        cv::Mat rect = coin::order_corners(*stable_corners);
        cached_M = cv::getPerspectiveTransform(rect, dst_corners);
      }

      if (!cached_M.empty())
      {
        double det = cv::determinant(cached_M);
        if (std::abs(det) > 1e-6)
        {
          auto t_ow = Clock::now();
          cv::warpPerspective(frame, warped, cached_M, cv::Size(width_px, height_px));
          timings.order_warp_ms = Ms(Clock::now() - t_ow).count();
          if (!warped.empty() && warped.rows > 0 && warped.cols > 0)
          {
            const int coin_every_n = std::max(1, coin::Config::COIN_DETECT_EVERY_N_FRAMES);
            bool do_detect = (coin_every_n <= 1 || (++coin_detect_count % coin_every_n) == 1);
#ifdef COIN_USE_TORCH
            run_coin_detection(warped, ratio_px_to_mm, tracker, svm,
                               coin::Config::CLASSIFIER_NAMES[classifier_index],
                               classifier_index, active_torch, !do_detect, &timings);
#else
            run_coin_detection(warped, ratio_px_to_mm, tracker, svm,
                               coin::Config::CLASSIFIER_NAMES[classifier_index],
                               0, nullptr, !do_detect, &timings);
#endif
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
      timings.total_frame_ms = Ms(Clock::now() - frame_start).count();
      if (s_print_timings)
      {
        if (frame_count % 60 == 1)
          std::cerr << "frame    | capture | paper | stabilizer | order+warp | detect_coins | tracker | classify | draw_rest | display | TOTAL (ms) | FPS\n";
        print_timings(timings, frame_count);
      }
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
    if (key == 't')
    {
      s_print_timings = !s_print_timings;
      std::cout << "Timings: " << (s_print_timings ? "ON" : "OFF") << "\n";
    }
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
