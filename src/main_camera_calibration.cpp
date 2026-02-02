#include "config.hpp"
#include "calibration.hpp"
#include "corner_stabilizer.hpp"
#include "coin_tracker.hpp"
#include "coin_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/persistence.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
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

  void solve_distortion(const std::vector<double> &px_dia, const std::vector<double> &r_sq,
                        const std::vector<double> &y_norm, const std::vector<double> &x_norm,
                        const std::vector<double> &true_mm, double &base_ratio, double &k_rad, double &p_y, double &p_x)
  {
    int n = static_cast<int>(px_dia.size());
    const double eps = 1e-8;
    const double lambda0 = 1e-3;
    double lambda = lambda0;
    base_ratio = INITIAL_RATIO;
    k_rad = 0.0;
    p_y = 0.0;
    p_x = 0.0;
    for (int iter = 0; iter < 80; ++iter)
    {
      cv::Mat J(n, 4, CV_64F);
      cv::Mat r(n, 1, CV_64F);
      for (int i = 0; i < n; ++i)
      {
        double corr = 1.0 + k_rad * r_sq[i] + p_y * y_norm[i] + p_x * x_norm[i];
        double pred = (px_dia[i] * base_ratio) * corr;
        r.at<double>(i, 0) = pred - true_mm[i];
        J.at<double>(i, 0) = px_dia[i] * corr;
        J.at<double>(i, 1) = px_dia[i] * base_ratio * r_sq[i];
        J.at<double>(i, 2) = px_dia[i] * base_ratio * y_norm[i];
        J.at<double>(i, 3) = px_dia[i] * base_ratio * x_norm[i];
      }
      cv::Mat Jt = J.t();
      cv::Mat JtJ = Jt * J;
      for (int i = 0; i < 4; ++i)
        JtJ.at<double>(i, i) += lambda;
      cv::Mat Jtr = Jt * r;
      cv::Mat delta;
      cv::solve(JtJ, -Jtr, delta, cv::DECOMP_CHOLESKY);
      double base_ratio_new = base_ratio + delta.at<double>(0, 0);
      double k_new = k_rad + delta.at<double>(1, 0);
      double py_new = p_y + delta.at<double>(2, 0);
      double px_new = p_x + delta.at<double>(3, 0);
      if (base_ratio_new < 0.1 || base_ratio_new > 2.0)
      {
        lambda *= 10;
        continue;
      }
      double err_old = 0, err_new = 0;
      for (int i = 0; i < n; ++i)
      {
        double corr_old = 1.0 + k_rad * r_sq[i] + p_y * y_norm[i] + p_x * x_norm[i];
        double corr_new = 1.0 + k_new * r_sq[i] + py_new * y_norm[i] + px_new * x_norm[i];
        double pred_old = (px_dia[i] * base_ratio) * corr_old;
        double pred_new = (px_dia[i] * base_ratio_new) * corr_new;
        err_old += (pred_old - true_mm[i]) * (pred_old - true_mm[i]);
        err_new += (pred_new - true_mm[i]) * (pred_new - true_mm[i]);
      }
      if (err_new < err_old)
      {
        base_ratio = base_ratio_new;
        k_rad = k_new;
        p_y = py_new;
        p_x = px_new;
        lambda *= 0.5;
        if (std::abs(delta.at<double>(0, 0)) < eps && std::abs(delta.at<double>(1, 0)) < eps && std::abs(delta.at<double>(2, 0)) < eps && std::abs(delta.at<double>(3, 0)) < eps)
          break;
      }
      else
      {
        lambda *= 10;
      }
    }
  }

}

int main()
{
  cv::VideoCapture cap(2, cv::CAP_V4L2);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

  coin::CornerStabilizer stabilizer(5);
  coin::CoinTracker tracker;
  cv::Mat dst_corners = (cv::Mat_<float>(4, 2) << 0, 0, WIDTH_PX - 1, 0, WIDTH_PX - 1, HEIGHT_PX - 1, 0, HEIGHT_PX - 1);

  double img_cx = WIDTH_PX / 2.0, img_cy = HEIGHT_PX / 2.0;
  double max_dist_sq = (WIDTH_PX / 2.0) * (WIDTH_PX / 2.0) + (HEIGHT_PX / 2.0) * (HEIGHT_PX / 2.0);
  double width_half = WIDTH_PX / 2.0;
  double height_px = HEIGHT_PX;

  std::vector<double> collected_px_dia, collected_r_sq, collected_y_norm, collected_x_norm, collected_true_mm;
  std::map<int, int> samples_count;
  for (const auto &z : ZONES)
    samples_count[z.class_id] = 0;

  std::cout << "--- ROBUST DISTORTION CALIBRATION ---\n";
  std::cout << "Move coins AROUND the image (Center, Edges, Corners).\n";
  std::cout << "'c' = Capture, 's' = Solve & Save, 'q' = Quit\n";

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

      auto detections = coin::detect_and_measure_coins(warped, INITIAL_RATIO, nullptr);
      tracker.update(detections);
      auto entries = tracker.get_stable_entries();

      struct Sample
      {
        double px_dia, r_sq, y_norm, x_norm, true_mm;
        int zone_id;
      };
      std::vector<Sample> current_frame_data;
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
        double dx = e.first.x - img_cx, dy = e.first.y - img_cy;
        double dist_sq = (dx * dx + dy * dy) / max_dist_sq;
        double y_norm = (height_px - e.first.y) / height_px;
        double x_norm = (e.first.x - img_cx) / width_half;
        current_frame_data.push_back({pixel_dia, dist_sq, y_norm, x_norm, true_mm, zone_id});
        int r = static_cast<int>(pixel_dia / 2);
        cv::circle(debug_warped, e.first, r, cv::Scalar(0, 255, std::min(255, static_cast<int>(255 * std::min(1.0, dist_sq)))), 2);
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
        for (const auto &s : current_frame_data)
        {
          collected_px_dia.push_back(s.px_dia);
          collected_r_sq.push_back(s.r_sq);
          collected_y_norm.push_back(s.y_norm);
          collected_x_norm.push_back(s.x_norm);
          collected_true_mm.push_back(s.true_mm);
          samples_count[s.zone_id]++;
        }
        std::cout << "Captured! Total samples: " << collected_px_dia.size() << "\n";
      }
      if (key == 's')
      {
        if (collected_px_dia.size() < 20)
        {
          std::cout << "Need more samples! Cover edges and center.\n";
          continue;
        }
        std::cout << "Solving for Distortion (Radial + Vertical + Horizontal Tilt)...\n";
        double best_ratio, best_k, best_p_y, best_p_x;
        solve_distortion(collected_px_dia, collected_r_sq, collected_y_norm, collected_x_norm,
                         collected_true_mm, best_ratio, best_k, best_p_y, best_p_x);
        std::cout << "--- RESULTS ---\n";
        std::cout << "Base Ratio: " << best_ratio << " mm/px\n";
        std::cout << "Radial (k): " << best_k << "\n";
        std::cout << "Vertical tilt (p_y): " << best_p_y << "\n";
        std::cout << "Horizontal tilt (p_x): " << best_p_x << "\n";

        cv::FileStorage fs(coin::Config::CALIBRATION_PATH, cv::FileStorage::WRITE);
        fs << "base_ratio" << best_ratio;
        fs << "distortion_k" << best_k;
        fs << "distortion_p_y" << best_p_y;
        fs << "distortion_p_x" << best_p_x;
        fs << "img_center" << "[" << img_cx << img_cy << "]";
        fs << "max_dist_sq" << max_dist_sq;
        fs << "height_px" << height_px;
        fs << "width_half" << width_half;
        fs << "height_half" << (HEIGHT_PX / 2.0);
        fs << "use_height_px" << true;
        fs.release();
        std::cout << "Saved '" << coin::Config::CALIBRATION_PATH << "'\n";
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
