#include "config.hpp"
#include "corner_stabilizer.hpp"
#include "coin_tracker.hpp"
#include "coin_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/persistence.hpp>
#include <iostream>
#include <vector>
#include <map>

namespace
{

  constexpr double PAPER_WIDTH_MM = 330.0;
  constexpr double PAPER_HEIGHT_MM = 216.0;
  constexpr double SCALE_FACTOR = 3.0;
  constexpr int WIDTH_PX = static_cast<int>(PAPER_WIDTH_MM * SCALE_FACTOR);
  constexpr int HEIGHT_PX = static_cast<int>(PAPER_HEIGHT_MM * SCALE_FACTOR);
  constexpr double INITIAL_RATIO = 1.0 / SCALE_FACTOR;
  constexpr int ZONE_WIDTH = WIDTH_PX / 6;

  struct Zone
  {
    int limit_x;
    const char *name;
    int class_id;
  };
  const Zone ZONES[] = {
      {ZONE_WIDTH * 1, "20 cent", 0}, {ZONE_WIDTH * 2, "10 cent", 1}, {ZONE_WIDTH * 3, "1 Euro", 2}, {ZONE_WIDTH * 4, "1 cent", 3}, {ZONE_WIDTH * 5, "2 cent", 4}, {9999, "5 cent", 5}};
  const double GROUND_TRUTH_DIAMETER_MM[] = {22.25, 19.75, 23.25, 16.25, 18.75, 21.25};

}

int main()
{
  cv::VideoCapture cap(2, cv::CAP_V4L2);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

  coin::CornerStabilizer stabilizer(5);
  coin::CoinTracker tracker;
  cv::Mat dst_corners = (cv::Mat_<float>(4, 2) << 0, 0, WIDTH_PX - 1, 0, WIDTH_PX - 1, HEIGHT_PX - 1, 0, HEIGHT_PX - 1);

  std::vector<double> collected_px_dia, collected_true_mm;
  std::map<int, int> samples_count;
  for (const auto &z : ZONES)
    samples_count[z.class_id] = 0;

  std::cout << "--- PX-TO-MM RATIO CALIBRATION ---\n";
  std::cout << "Place coins in zones. 'c' = Capture, 's' = Solve & Save ratio, 'q' = Quit\n";

  while (true)
  {
    cv::Mat frame;
    if (!cap.read(frame))
      break;

    auto raw_corners = coin::find_paper_corners(frame);
    std::optional<cv::Mat> stable_corners = raw_corners.has_value()
                                                ? stabilizer.update(&*raw_corners)
                                                : stabilizer.update(nullptr);
    cv::Mat display;
    frame.copyTo(display);

    if (stable_corners.has_value() && !stable_corners->empty())
    {
      cv::Mat rect = coin::order_corners(*stable_corners);
      cv::Mat M = cv::getPerspectiveTransform(rect, dst_corners);
      cv::Mat warped;
      cv::warpPerspective(frame, warped, M, cv::Size(WIDTH_PX, HEIGHT_PX));
      cv::Mat debug_warped;
      warped.copyTo(debug_warped);

      auto detections = coin::detect_and_measure_coins(warped, INITIAL_RATIO);
      tracker.update(detections);
      auto entries = tracker.get_stable_entries();

      for (const auto &e : entries)
      {
        int zone_id = -1;
        for (const auto &z : ZONES)
        {
          if (e.first.x < z.limit_x)
          {
            zone_id = z.class_id;
            break;
          }
        }
        if (zone_id < 0)
          continue;
        double true_mm = GROUND_TRUTH_DIAMETER_MM[zone_id];
        double pixel_dia = e.second / INITIAL_RATIO;
        int r = static_cast<int>(pixel_dia / 2);
        cv::circle(debug_warped, e.first, r, cv::Scalar(0, 255, 0), 2);
      }

      int prev_x = 0;
      for (const auto &z : ZONES)
      {
        if (z.limit_x < WIDTH_PX)
          cv::line(debug_warped, cv::Point(z.limit_x, 0), cv::Point(z.limit_x, HEIGHT_PX), cv::Scalar(0, 0, 255), 2);
        int cx_zone = (prev_x + std::min(z.limit_x, WIDTH_PX)) / 2;
        std::string label = std::string(z.name) + ": " + std::to_string(samples_count[z.class_id]);
        cv::putText(debug_warped, label, cv::Point(cx_zone - 40, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        prev_x = z.limit_x;
      }
      cv::putText(debug_warped, "Samples: " + std::to_string(collected_px_dia.size()),
                  cv::Point(10, HEIGHT_PX - 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
      cv::Mat small;
      cv::resize(debug_warped, small, cv::Size(), 0.6, 0.6);
      cv::imshow("Calibration", small);

      int key = cv::waitKey(1);
      if (key == 'q')
        break;
      if (key == 'c')
      {
        for (const auto &e : entries)
        {
          int zone_id = -1;
          for (const auto &z : ZONES)
          {
            if (e.first.x < z.limit_x)
            {
              zone_id = z.class_id;
              break;
            }
          }
          if (zone_id < 0)
            continue;
          double true_mm = GROUND_TRUTH_DIAMETER_MM[zone_id];
          double pixel_dia = e.second / INITIAL_RATIO;
          collected_px_dia.push_back(pixel_dia);
          collected_true_mm.push_back(true_mm);
          samples_count[zone_id]++;
        }
        std::cout << "Captured! Total samples: " << collected_px_dia.size() << "\n";
      }
      if (key == 's')
      {
        if (collected_px_dia.size() < 6)
        {
          std::cout << "Need at least 6 samples (one per zone).\n";
          continue;
        }
        double sum_mm = 0, sum_px = 0;
        for (size_t i = 0; i < collected_px_dia.size(); ++i)
        {
          sum_mm += collected_true_mm[i];
          sum_px += collected_px_dia[i];
        }
        double ratio_px_to_mm = sum_mm / sum_px;
        double scale_factor = 1.0 / ratio_px_to_mm;
        std::cout << "--- PX-TO-MM RATIO ---\n";
        std::cout << "ratio_px_to_mm: " << ratio_px_to_mm << " mm/px\n";
        std::cout << "SCALE_FACTOR (px/mm): " << scale_factor << " (use in Config if needed)\n";

        cv::FileStorage fs(coin::Config::CALIBRATION_PATH, cv::FileStorage::WRITE);
        fs << "ratio_px_to_mm" << ratio_px_to_mm;
        fs.release();
        std::cout << "Saved '" << coin::Config::CALIBRATION_PATH << "' (ratio only).\n";
        break;
      }
    }
    else
    {
      cv::imshow("Calibration", display);
      if (cv::waitKey(1) == 'q')
        break;
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
