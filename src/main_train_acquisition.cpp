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
#include <algorithm>
#include <sys/stat.h>
#include <filesystem>
#include <regex>

namespace
{

  constexpr int WIDTH_PX = static_cast<int>(coin::Config::PAPER_WIDTH_MM * coin::Config::SCALE_FACTOR);
  constexpr int HEIGHT_PX = static_cast<int>(coin::Config::PAPER_HEIGHT_MM * coin::Config::SCALE_FACTOR);
  constexpr double RATIO_PX_TO_MM = 1.0 / coin::Config::SCALE_FACTOR;
  constexpr int ZONE_WIDTH = WIDTH_PX / 6;

  const char *IMAGES_DIR = "images";
  const char *LABELS_DIR = "labels";

  /** Class id to label string for JSON (index = class_id). */
  const char *CLASS_LABELS[] = {"20cent", "10cent", "1euro", "1cent", "2cent", "5cent"};

  struct Zone
  {
    int limit_x;
    const char *name;
    int class_id;
  };

  struct ClassInfo
  {
    const char *name;
    int class_id;
  };

  struct Shape
  {
    std::string label;
    int center_x, center_y;
    int radius;
  };

  const ClassInfo CLASSES[] = {
      {"20 cent", 0},
      {"10 cent", 1},
      {"1 Euro", 2},
      {"1 cent", 3},
      {"2 cent", 4},
      {"5 cent", 5}};

  /** 6 fixed sequences: each row is the class_id order for zones left-to-right. */
  const int SEQUENCES[6][6] = {
      {0, 1, 2, 3, 4, 5}, /* seq 0: 20c, 10c, 1€, 1c, 2c, 5c */
      {1, 0, 3, 4, 5, 2}, /* seq 1 */
      {2, 3, 4, 5, 0, 1}, /* seq 2 */
      {3, 4, 5, 0, 1, 2}, /* seq 3 */
      {4, 5, 0, 1, 2, 3}, /* seq 4 */
      {5, 0, 1, 2, 3, 4}, /* seq 5 */
  };

  std::vector<Zone> build_zones_for_sequence(int sequence_index)
  {
    const int limits[] = {ZONE_WIDTH * 1, ZONE_WIDTH * 2, ZONE_WIDTH * 3, ZONE_WIDTH * 4, ZONE_WIDTH * 5, 9999};
    sequence_index = sequence_index % 6;
    if (sequence_index < 0)
      sequence_index += 6;
    std::vector<Zone> zones;
    zones.reserve(6);
    for (int i = 0; i < 6; ++i)
    {
      int class_id = SEQUENCES[sequence_index][i];
      zones.push_back({limits[i], CLASSES[class_id].name, class_id});
    }
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

  void ensure_training_dirs(const std::string &base_dir)
  {
    ensure_dir(base_dir);
    ensure_dir(base_dir + "/" + IMAGES_DIR);
    ensure_dir(base_dir + "/" + LABELS_DIR);
  }

  int get_next_image_index(const std::string &images_dir)
  {
    namespace fs = std::filesystem;
    std::regex num_re(R"((\d+)\.png)", std::regex::icase);
    int max_n = -1;
    if (!fs::is_directory(images_dir))
      return 0;
    for (const auto &entry : fs::directory_iterator(images_dir))
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

  bool write_annotation_json(const std::string &path, const std::string &image_path,
                             int image_width, int image_height,
                             const std::vector<Shape> &shapes)
  {
    std::ofstream out(path);
    if (!out)
      return false;
    out << "{\n";
    out << "  \"imagePath\": \"" << image_path << "\",\n";
    out << "  \"imageHeight\": " << image_height << ",\n";
    out << "  \"imageWidth\": " << image_width << ",\n";
    out << "  \"shapes\": [\n";
    for (size_t i = 0; i < shapes.size(); ++i)
    {
      const auto &s = shapes[i];
      out << "    {\n";
      out << "      \"label\": \"" << s.label << "\",\n";
      out << "      \"center\": [" << s.center_x << ", " << s.center_y << "],\n";
      out << "      \"radius\": " << s.radius << "\n";
      out << "    }";
      if (i + 1 < shapes.size())
        out << ",";
      out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return true;
  }

}

int main()
{
  cv::VideoCapture cap(2, cv::CAP_V4L2);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

  std::cout << "Using px-to-mm ratio: " << RATIO_PX_TO_MM << " mm/px (from SCALE_FACTOR)." << std::endl;

  int current_sequence = 0;
  std::vector<Zone> zones = build_zones_for_sequence(current_sequence);

  const std::string base_dir(coin::Config::TRAINING_DATA_DIR);
  const std::string images_dir = base_dir + "/" + IMAGES_DIR;
  const std::string labels_dir = base_dir + "/" + LABELS_DIR;
  ensure_training_dirs(base_dir);

  coin::CornerStabilizer stabilizer(coin::Config::STABILIZER_WINDOW);
  coin::CoinTracker tracker;
  cv::Mat dst_corners = (cv::Mat_<float>(4, 2) << 0, 0, WIDTH_PX - 1, 0, WIDTH_PX - 1, HEIGHT_PX - 1, 0, HEIGHT_PX - 1);

  int next_image_index = get_next_image_index(images_dir);

  std::cout << "--- TRAINING DATA ACQUISITION ---\n";
  std::cout << "Output: " << base_dir << "/ (" << IMAGES_DIR << "/ + " << LABELS_DIR << "/ JSON)\n";
  std::cout << "6 sequences; change with 's'. Current: sequence " << (current_sequence + 1) << "/6. Zones left to right = ";
  for (size_t i = 0; i < zones.size(); ++i)
    std::cout << zones[i].name << (i + 1 < zones.size() ? " | " : "\n");
  std::cout << "  's' = next sequence (1-6)\n";
  std::cout << "  'c' = capture (save full image + annotation JSON)\n";
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
        cv::putText(debug_warped, z.name, cv::Point(cx_zone - 40, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 4);
        prev_x = z.limit_x;
      }
      cv::putText(debug_warped, "Seq " + std::to_string(current_sequence + 1) + "/6", cv::Point(10, HEIGHT_PX - 15), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

      auto detections = coin::detect_and_measure_coins(warped, RATIO_PX_TO_MM);
      tracker.update(detections);
      auto entries = tracker.get_stable_entries();
      for (const auto &e : entries)
      {
        int r = coin::diameter_mm_to_radius_px(e.second, RATIO_PX_TO_MM);
        cv::circle(debug_warped, e.first, r, cv::Scalar(0, 0, 255), 3);
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
    if (key == 's')
    {
      current_sequence = (current_sequence + 1) % 6;
      zones = build_zones_for_sequence(current_sequence);
      std::cout << "Sequence " << (current_sequence + 1) << "/6. Zones: ";
      for (size_t i = 0; i < zones.size(); ++i)
        std::cout << zones[i].name << (i + 1 < zones.size() ? " | " : "\n");
    }
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

      std::vector<Shape> shapes;
      for (const auto &d : detections)
      {
        int radius_px = coin::diameter_mm_to_radius_px(d.diameter_mm, RATIO_PX_TO_MM);
        if (radius_px < 5)
          continue;
        int class_id = -1;
        for (const auto &z : zones)
        {
          if (d.center.x < z.limit_x)
          {
            class_id = z.class_id;
            break;
          }
        }
        if (class_id < 0)
          continue;
        shapes.push_back({CLASS_LABELS[class_id], d.center.x, d.center.y, radius_px});
      }

      std::string base_name = std::to_string(next_image_index) + ".png";
      std::string json_name = std::to_string(next_image_index) + ".json";
      std::string image_path_full = images_dir + "/" + base_name;
      std::string json_path_full = labels_dir + "/" + json_name;

      if (cv::imwrite(image_path_full, warped) &&
          write_annotation_json(json_path_full, base_name, warped.cols, warped.rows, shapes))
      {
        std::cout << "Saved " << base_name << " + " << json_name << " (" << shapes.size() << " shapes)\n";
        next_image_index++;
      }
      else
        std::cerr << "Failed to write image or JSON\n";

      cv::Mat drain;
      for (int i = 0; i < 15; ++i)
        cap.read(drain);
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}
