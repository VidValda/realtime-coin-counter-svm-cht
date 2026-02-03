#include "config.hpp"
#include "calibration.hpp"
#include "corner_stabilizer.hpp"
#include "coin_tracker.hpp"
#include "coin_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <algorithm>
#include <random>
#include <chrono>
#include <sys/stat.h>
#include <filesystem>
#include <regex>

namespace
{

  constexpr int WIDTH_PX = static_cast<int>(coin::Config::PAPER_WIDTH_MM * coin::Config::SCALE_FACTOR);
  constexpr int HEIGHT_PX = static_cast<int>(coin::Config::PAPER_HEIGHT_MM * coin::Config::SCALE_FACTOR);
  constexpr double RATIO_PX_TO_MM = 1.0 / coin::Config::SCALE_FACTOR;
  constexpr int ZONE_WIDTH = WIDTH_PX / 6;

  struct Zone
  {
    int limit_x;
    const char *name;
    const char *dir_name;
    int class_id;
  };

  struct ClassInfo
  {
    const char *name;
    const char *dir_name;
    int class_id;
  };

  std::vector<Zone> build_randomized_zones(std::mt19937 &rng)
  {
    const int limits[] = {ZONE_WIDTH * 1, ZONE_WIDTH * 2, ZONE_WIDTH * 3, ZONE_WIDTH * 4, ZONE_WIDTH * 5, 9999};
    const ClassInfo classes[] = {
        {"20 cent", "20cent", 0},
        {"10 cent", "10cent", 1},
        {"1 Euro", "1euro", 2},
        {"1 cent", "1cent", 3},
        {"2 cent", "2cent", 4},
        {"5 cent", "5cent", 5}};
    std::vector<int> perm = {0, 1, 2, 3, 4, 5};
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<Zone> zones;
    zones.reserve(6);
    for (int i = 0; i < 6; ++i)
      zones.push_back({limits[i], classes[perm[i]].name, classes[perm[i]].dir_name, classes[perm[i]].class_id});
    return zones;
  }

  bool ensure_dir(const std::string &path)
  {
    struct stat st;
    if (stat(path.c_str(), &st) == 0)
      return S_ISDIR(st.st_mode);
#ifdef _WIN32
    return mkdir(path.c_str()) == 0;
#else
    return mkdir(path.c_str(), 0755) == 0;
#endif
  }

  void ensure_training_dirs(const std::vector<Zone> &zones)
  {
    ensure_dir(coin::Config::TRAINING_DATA_DIR);
    for (const auto &z : zones)
      ensure_dir(std::string(coin::Config::TRAINING_DATA_DIR) + "/" + z.dir_name);
  }

  void ensure_manifest_header()
  {
    std::ifstream in(coin::Config::TRAINING_MANIFEST);
    if (in.good())
      return;
    std::ofstream out(coin::Config::TRAINING_MANIFEST);
    if (out)
      out << "path,class_id,diameter_mm\n";
  }

  bool append_manifest(const std::string &path, int class_id, double diameter_mm)
  {
    std::ofstream out(coin::Config::TRAINING_MANIFEST, std::ios::app);
    if (!out)
      return false;
    out << path << "," << class_id << "," << std::fixed << diameter_mm << "\n";
    return true;
  }

  int get_next_index_for_dir(const std::string &dir_path)
  {
    namespace fs = std::filesystem;
    std::regex num_re(R"((\d+)\.png)", std::regex::icase);
    int max_n = -1;
    if (!fs::is_directory(dir_path))
      return 0;
    for (const auto &entry : fs::directory_iterator(dir_path))
    {
      if (!entry.is_regular_file())
        continue;
      std::string name = entry.path().filename().string();
      std::smatch m;
      if (std::regex_match(name, m, num_re))
        max_n = std::max(max_n, std::stoi(m[1].str()));
    }
    return max_n + 1;
  }

  std::map<int, int> get_next_indices_for_classes(const std::vector<Zone> &zones)
  {
    std::map<int, int> next_index;
    for (const auto &z : zones)
    {
      std::string dir_path = std::string(coin::Config::TRAINING_DATA_DIR) + "/" + z.dir_name;
      next_index[z.class_id] = get_next_index_for_dir(dir_path);
    }
    return next_index;
  }

}

int main()
{
  cv::VideoCapture cap(2, cv::CAP_V4L2);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

  std::cout << "Using px-to-mm ratio: " << RATIO_PX_TO_MM << " mm/px (from SCALE_FACTOR)." << std::endl;

  std::mt19937 rng(static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::vector<Zone> zones = build_randomized_zones(rng);

  ensure_training_dirs(zones);
  ensure_manifest_header();

  coin::CornerStabilizer stabilizer(coin::Config::STABILIZER_WINDOW);
  coin::CoinTracker tracker;
  cv::Mat dst_corners = (cv::Mat_<float>(4, 2) << 0, 0, WIDTH_PX - 1, 0, WIDTH_PX - 1, HEIGHT_PX - 1, 0, HEIGHT_PX - 1);

  std::map<int, int> class_counts = get_next_indices_for_classes(zones);

  std::cout << "--- TRAINING DATA ACQUISITION ---\n";
  std::cout << "Data: " << coin::Config::TRAINING_DATA_DIR << "/ + " << coin::Config::TRAINING_MANIFEST
            << " (same as train_classifier.py).\n";
  std::cout << "Zones (randomized this run): left to right = ";
  for (size_t i = 0; i < zones.size(); ++i)
    std::cout << zones[i].name << (i + 1 < zones.size() ? " | " : "\n");
  std::cout << "Align paper. Place coins in zones.\n";
  std::cout << "  'c' = capture (save coin crops)\n";
  std::cout << "  1/2/3/4 = set default classifier for coin_counter (1=SVM 2=KNN 3=RF 4=NB)\n";
  std::cout << "  'q' = quit\n";

  int default_classifier = 0;
  auto write_default_classifier = [](int idx)
  {
    std::ofstream f(coin::Config::CLASSIFIER_DEFAULT_FILE);
    if (f)
      f << idx << "\n";
  };
  write_default_classifier(default_classifier);

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

      int prev_x = 0;
      for (const auto &z : zones)
      {
        if (z.limit_x < WIDTH_PX)
          cv::line(debug_warped, cv::Point(z.limit_x, 0), cv::Point(z.limit_x, HEIGHT_PX), cv::Scalar(0, 0, 255), 2);
        int cx_zone = (prev_x + std::min(z.limit_x, WIDTH_PX)) / 2;
        std::string label = std::string(z.name) + ": " + std::to_string(class_counts[z.class_id]);
        cv::putText(debug_warped, label, cv::Point(cx_zone - 40, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        prev_x = z.limit_x;
      }

      auto detections = coin::detect_and_measure_coins(warped, RATIO_PX_TO_MM);
      tracker.update(detections);
      auto entries = tracker.get_stable_entries();
      for (const auto &e : entries)
      {
        int r = coin::diameter_mm_to_radius_px(e.second, RATIO_PX_TO_MM);
        cv::circle(debug_warped, e.first, r, cv::Scalar(0, 255, 0), 2);
      }
      cv::Mat small;
      cv::resize(debug_warped, small, cv::Size(), 0.6, 0.6);
      cv::imshow("Acquisition", small);
    }
    else
    {
      cv::imshow("Acquisition", display);
    }

    int key = cv::waitKey(1);
    if (key == 'q')
      break;
    if (key == '1' || key == '2' || key == '3' || key == '4')
    {
      default_classifier = key - '1';
      write_default_classifier(default_classifier);
      std::cout << "Default classifier set to " << coin::Config::CLASSIFIER_NAMES[default_classifier] << " for coin_counter.\n";
    }
    if (key == 'c' && stable_corners.has_value() && !stable_corners->empty())
    {
      cv::Mat rect = coin::order_corners(*stable_corners);
      cv::Mat M = cv::getPerspectiveTransform(rect, dst_corners);
      cv::Mat warped;
      cv::warpPerspective(frame, warped, M, cv::Size(WIDTH_PX, HEIGHT_PX));
      auto detections = coin::detect_and_measure_coins(warped, RATIO_PX_TO_MM);
      std::vector<std::pair<cv::Point2i, double>> entries;
      for (const auto &d : detections)
        entries.emplace_back(d.center, d.diameter_mm);
      int n_captured = 0;
      for (const auto &e : entries)
      {
        int label_id = -1;
        const Zone *zone_ptr = nullptr;
        for (const auto &z : zones)
        {
          if (e.first.x < z.limit_x)
          {
            label_id = z.class_id;
            zone_ptr = &z;
            break;
          }
        }
        if (label_id < 0 || !zone_ptr)
          continue;
        int radius_px = coin::diameter_mm_to_radius_px(e.second, RATIO_PX_TO_MM);
        if (radius_px < 5)
          continue;
        int pad = std::min(10, radius_px / 2);
        int side = std::min(2 * (radius_px + pad), std::min(warped.cols, warped.rows));
        int x0 = std::max(0, e.first.x - side / 2);
        int y0 = std::max(0, e.first.y - side / 2);
        if (x0 + side > warped.cols)
          x0 = warped.cols - side;
        if (y0 + side > warped.rows)
          y0 = warped.rows - side;
        cv::Rect roi(x0, y0, side, side);
        cv::Mat crop = warped(roi).clone();

        int count = class_counts[label_id];
        std::string rel_path = std::string(zone_ptr->dir_name) + "/" + std::to_string(count) + ".png";
        std::string full_path = std::string(coin::Config::TRAINING_DATA_DIR) + "/" + rel_path;
        if (cv::imwrite(full_path, crop) && append_manifest(rel_path, label_id, e.second))
        {
          class_counts[label_id]++;
          n_captured++;
        }
      }
      std::cout << "Saved " << n_captured << " coin images. Total per class: ";
      for (const auto &z : zones)
        std::cout << z.name << "=" << class_counts[z.class_id] << " ";
      std::cout << "\n";
      cv::Mat drain;
      for (int i = 0; i < 15; ++i)
        cap.read(drain);
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
