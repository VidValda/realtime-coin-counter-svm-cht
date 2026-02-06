#include "coin_detector.hpp"
#include "config.hpp"
#include "calibration.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <algorithm>
#include <random>

namespace coin
{

  static int ensure_odd(int k) { return std::max(1, k | 1); }

  /** Get single channel for watershed: 0=Gray, 1=H, 2=S, 3=V, 4=L, 5=A, 6=B */
  static cv::Mat get_channel(const cv::Mat &frame, int mode)
  {
    mode = std::max(0, std::min(6, mode));
    if (mode == 0)
    {
      cv::Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      return gray;
    }
    if (mode >= 1 && mode <= 3)
    {
      cv::Mat hsv, ch;
      cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
      cv::extractChannel(hsv, ch, mode - 1);
      return ch;
    }
    cv::Mat lab, ch;
    cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
    cv::extractChannel(lab, ch, mode - 4);
    return ch;
  }

  /** Preprocess for watershed: channel -> CLAHE -> median blur. CLAHE is cached to avoid alloc per frame. */
  static cv::Mat preprocess_for_watershed(const cv::Mat &frame)
  {
    cv::Mat ch = get_channel(frame, Config::CHANNEL_MODE);
    cv::Mat enhanced;
    static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
        static_cast<double>(std::max(1, Config::CLAHE_CLIP)),
        cv::Size(std::max(1, Config::CLAHE_GRID), std::max(1, Config::CLAHE_GRID)));
    clahe->apply(ch, enhanced);
    int k = ensure_odd(Config::BLUR_KSIZE);
    cv::Mat blurred;
    cv::medianBlur(enhanced, blurred, k);
    return blurred;
  }

  cv::Mat preprocess_for_circles(const cv::Mat &frame)
  {
    cv::Mat ch = get_channel(frame, Config::CHANNEL_MODE);
    cv::Mat enhanced, blurred;
    static cv::Ptr<cv::CLAHE> clahe_circles = cv::createCLAHE(
        static_cast<double>(Config::CLAHE_CLIP),
        cv::Size(std::max(1, Config::CLAHE_GRID), std::max(1, Config::CLAHE_GRID)));
    clahe_circles->apply(ch, enhanced);
    cv::medianBlur(enhanced, blurred, ensure_odd(Config::BLUR_KSIZE));
    return blurred;
  }

  std::vector<cv::Vec3f> find_circle_candidates(const cv::Mat &blurred)
  {
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred, circles, cv::HOUGH_GRADIENT, Config::HOUGH_DP,
                     Config::HOUGH_MIN_DIST, Config::HOUGH_PARAM1, Config::HOUGH_PARAM2,
                     Config::MIN_RADIUS_PX, Config::MAX_RADIUS_PX);
    return circles;
  }

  std::optional<Detection> measure_circle_diameter(const cv::Mat &frame_gray,
                                                   int x, int y, int r, double ratio_px_to_mm)
  {
    int h = frame_gray.rows, w = frame_gray.cols;
    cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);
    cv::circle(mask, cv::Point(x, y), r, 255, -1);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
      return std::nullopt;
    auto it = std::max_element(contours.begin(), contours.end(),
                               [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
                               {
                                 return cv::contourArea(a) < cv::contourArea(b);
                               });
    double area = cv::contourArea(*it);
    if (area < Config::MIN_CONTOUR_AREA)
      return std::nullopt;
    double perimeter = cv::arcLength(*it, true);
    if (perimeter <= 0)
      return std::nullopt;
    double circularity = 4 * CV_PI * (area / (perimeter * perimeter));
    if (circularity < Config::MIN_CIRCULARITY)
      return std::nullopt;
    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(*it, center, radius);
    double diameter_px = 2.0 * radius;
    double diameter_mm = pixel_diameter_to_mm(diameter_px, ratio_px_to_mm);
    if (diameter_mm < Config::DIAMETER_MM_MIN || diameter_mm > Config::DIAMETER_MM_MAX)
      return std::nullopt;
    return Detection{cv::Point2i(static_cast<int>(center.x), static_cast<int>(center.y)), diameter_mm};
  }

  Detections detect_and_measure_coins(const cv::Mat &frame, double ratio_px_to_mm,
                                      DebugViews *out_debug, double pixel_scale)
  {
    if (pixel_scale <= 0)
      pixel_scale = 1.0;
    const double min_contour_area = Config::MIN_CONTOUR_AREA * pixel_scale * pixel_scale;

    cv::Mat blurred = preprocess_for_watershed(frame);

    // 1. Threshold: Otsu or Adaptive
    cv::Mat binary;
    if (Config::USE_ADAPTIVE)
    {
      int block = ensure_odd(std::max(3, std::min(51, Config::ADAPTIVE_BLOCK)));
      cv::adaptiveThreshold(blurred, binary, 255,
                            cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                            block, Config::ADAPTIVE_C);
    }
    else
    {
      cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);
    }
    if (Config::INVERT_BINARY)
      cv::bitwise_not(binary, binary);

    // 2. Morphological open
    int k_open = ensure_odd(std::max(1, Config::MORPH_OPEN_SIZE));
    cv::Mat k_open_el = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_open, k_open));
    cv::Mat after_open;
    cv::morphologyEx(binary, after_open, cv::MORPH_OPEN, k_open_el,
                     cv::Point(-1, -1), std::max(0, Config::MORPH_OPEN_ITERS));

    // 3. Morphological close
    int k_close = ensure_odd(std::max(1, Config::MORPH_CLOSE_SIZE));
    cv::Mat k_close_el = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_close, k_close));
    cv::Mat after_close;
    cv::morphologyEx(after_open, after_close, cv::MORPH_CLOSE, k_close_el,
                     cv::Point(-1, -1), std::max(0, Config::MORPH_CLOSE_ITERS));
    binary = after_close;

    // 4. Sure background
    int k_bg = ensure_odd(std::max(1, Config::BG_DILATE_SIZE));
    cv::Mat k_bg_el = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_bg, k_bg));
    cv::Mat sure_bg;
    cv::dilate(binary, sure_bg, k_bg_el);

    // 5. Distance transform -> sure foreground
    int dsize = (Config::DIST_MASK_SIZE >= 4) ? 5 : 3;
    cv::Mat dist;
    cv::distanceTransform(binary, dist, cv::DIST_L2, dsize);
    double dist_max;
    cv::minMaxLoc(dist, nullptr, &dist_max);

    if (out_debug && dist_max > 0)
    {
      cv::Mat dist_norm;
      cv::normalize(dist, dist_norm, 0, 255, cv::NORM_MINMAX);
      dist_norm.convertTo(out_debug->dist_vis, CV_8UC1);
    }
    if (dist_max <= 0)
      return {};

    double frac = std::max(0.2, std::min(0.6, Config::WATERSHED_FG_FRAC));
    cv::Mat sure_fg;
    cv::threshold(dist, sure_fg, frac * dist_max, 255, cv::THRESH_BINARY);
    sure_fg.convertTo(sure_fg, CV_8UC1);

    if (out_debug)
      sure_fg.copyTo(out_debug->sure_fg);
    if (out_debug)
      binary.copyTo(out_debug->binary);

    // 6. Unknown = sure_bg - sure_fg
    cv::Mat unknown;
    cv::subtract(sure_bg, sure_fg, unknown);

    // 7. Markers: connected components on sure_fg, then mark unknown as 0
    cv::Mat markers;
    int num_labels = cv::connectedComponents(sure_fg, markers);
    markers.convertTo(markers, CV_32S);
    markers += 1;
    markers.setTo(0, unknown);

    // Marker visualization (color by label)
    if (!markers.empty())
    {
      cv::Mat markers_vis = cv::Mat::zeros(markers.rows, markers.cols, CV_8UC3);
      std::mt19937 rng(42);
      std::uniform_int_distribution<int> u(150, 255);
      for (int lid = 1; lid <= num_labels; ++lid)
      {
        cv::Vec3b c(static_cast<uchar>(u(rng)), static_cast<uchar>(u(rng)), static_cast<uchar>(u(rng)));
        markers_vis.setTo(c, markers == lid);
        cv::imshow("Debug: Markers", markers_vis);
      }
    }

    // 8. Watershed (requires 3-channel image)
    cv::Mat watershed_input;
    cv::cvtColor(after_close, watershed_input, cv::COLOR_GRAY2BGR);
    cv::Mat markers_out = markers.clone();
    cv::watershed(watershed_input, markers_out);

    // 9. Extract regions, filter by area/circularity/diameter, NMS
    std::vector<Detection> detections;
    cv::Mat segmentation_vis;
    if (out_debug)
    {
      segmentation_vis = cv::Mat(frame.rows, frame.cols, CV_8UC3);
      segmentation_vis.setTo(cv::Scalar(180, 180, 180));
      // Overlay original frame with transparency
      if (frame.channels() == 3)
        cv::addWeighted(segmentation_vis, 0.5, frame, 0.5, 0, segmentation_vis);
      else if (frame.channels() == 1)
      {
        cv::Mat frame_bgr;
        cv::cvtColor(frame, frame_bgr, cv::COLOR_GRAY2BGR);
        cv::addWeighted(segmentation_vis, 0.5, frame_bgr, 0.5, 0, segmentation_vis);
      }
    }
    // Same palette as marker visualization (label index 1..num_labels)
    std::vector<cv::Vec3b> seg_palette;
    if (out_debug && num_labels >= 1)
    {
      seg_palette.resize(num_labels + 1);
      std::mt19937 rng_p(42);
      std::uniform_int_distribution<int> u_p(150, 255);
      for (int lid = 1; lid <= num_labels; ++lid)
        seg_palette[lid] = cv::Vec3b(static_cast<uchar>(u_p(rng_p)), static_cast<uchar>(u_p(rng_p)), static_cast<uchar>(u_p(rng_p)));
    }
    for (int label = 2; label <= num_labels; ++label)
    {
      cv::Mat mask = (markers_out == label);
      mask.convertTo(mask, CV_8UC1, 255);
      std::vector<std::vector<cv::Point>> contours;
      cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      if (contours.empty())
        continue;
      const auto &cnt = *std::max_element(contours.begin(), contours.end(),
                                          [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
                                          { return cv::contourArea(a) < cv::contourArea(b); });
      double area = cv::contourArea(cnt);
      if (area < min_contour_area)
        continue;
      double perimeter = cv::arcLength(cnt, true);
      if (perimeter <= 0)
        continue;
      double circularity = 4 * CV_PI * (area / (perimeter * perimeter));
      if (circularity < Config::MIN_CIRCULARITY)
        continue;
      cv::Point2f center;
      float radius;
      cv::minEnclosingCircle(cnt, center, radius);
      if (radius < 1.f)
        continue;
      double diameter_px = 2.0 * static_cast<double>(radius);
      double diameter_mm = pixel_diameter_to_mm(diameter_px, ratio_px_to_mm);
      if (diameter_mm < Config::DIAMETER_MM_MIN || diameter_mm > Config::DIAMETER_MM_MAX)
        continue;
      Detection det;
      det.center = cv::Point2i(static_cast<int>(std::round(center.x)), static_cast<int>(std::round(center.y)));
      det.diameter_mm = diameter_mm;
      detections.push_back(det);
      if (out_debug && !segmentation_vis.empty() && label < static_cast<int>(seg_palette.size()))
      {
        const cv::Vec3b &c = seg_palette[label];
        // Draw colors directly on mask areas with high opacity
        cv::Mat colored_layer(segmentation_vis.size(), CV_8UC3, cv::Scalar::all(0));
        colored_layer.setTo(c, mask);
        cv::addWeighted(segmentation_vis, 0.2, colored_layer, 0.8, 0, segmentation_vis);
      }
    }

    if (out_debug && !segmentation_vis.empty())
      out_debug->segmentation = segmentation_vis;

    // NMS by center distance
    Detections kept;
    for (const auto &d : detections)
    {
      bool too_close = false;
      for (const auto &k : kept)
      {
        double dx = d.center.x - k.center.x;
        double dy = d.center.y - k.center.y;
        if (std::hypot(dx, dy) < Config::CENTER_MATCH_PX)
        {
          too_close = true;
          break;
        }
      }
      if (!too_close)
        kept.push_back(d);
    }
    return kept;
  }

  std::optional<cv::Mat> find_paper_corners(const cv::Mat &frame)
  {
    const int max_w = Config::PAPER_DETECT_MAX_WIDTH;
    double scale = 1.0;
    cv::Mat work = frame;
    if (max_w > 0 && frame.cols > max_w)
    {
      scale = static_cast<double>(max_w) / frame.cols;
      cv::resize(frame, work, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }
    const int min_area_scaled = static_cast<int>(Config::PAPER_MIN_AREA * scale * scale);
    const int line_min = static_cast<int>(Config::PAPER_LINE_MIN_LENGTH * scale);

    cv::Mat gray, gray_filtered;
    cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray_filtered, cv::Size(5, 5), 0);
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(0);
    std::vector<cv::Vec4f> lines;
    lsd->detect(gray_filtered, lines);
    cv::Mat line_mask = cv::Mat::zeros(gray_filtered.size(), CV_8UC1);
    for (const auto &line : lines)
    {
      double dx = line[2] - line[0], dy = line[3] - line[1];
      if (std::hypot(dx, dy) > line_min)
      {
        cv::line(line_mask, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), 255, 2);
      }
    }
    int k = std::max(1, Config::PAPER_MORPH_KERNEL | 1);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
    cv::Mat closed;
    cv::morphologyEx(line_mask, closed, cv::MORPH_CLOSE, kernel);
    cv::dilate(closed, closed, kernel);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(),
              [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b)
              {
                return cv::contourArea(a) > cv::contourArea(b);
              });
    for (size_t i = 0; i < std::min(size_t(3), contours.size()); ++i)
    {
      const auto &cnt = contours[i];
      double peri = cv::arcLength(cnt, true);
      std::vector<cv::Point> approx;
      cv::approxPolyDP(cnt, approx, Config::PAPER_APPROX_EPS_FACTOR * peri, true);
      if (approx.size() == 4 && cv::contourArea(approx) > min_area_scaled)
      {
        cv::Mat corners(4, 2, CV_32F);
        const double inv_scale = 1.0 / scale;
        for (int j = 0; j < 4; ++j)
        {
          corners.at<float>(j, 0) = static_cast<float>(approx[j].x * inv_scale);
          corners.at<float>(j, 1) = static_cast<float>(approx[j].y * inv_scale);
        }
        return corners;
      }
    }
    return std::nullopt;
  }

  cv::Mat order_corners(const cv::Mat &corners)
  {
    std::vector<cv::Point2f> pts(4);
    for (int i = 0; i < 4; ++i)
      pts[i] = cv::Point2f(corners.at<float>(i, 0), corners.at<float>(i, 1));
    std::sort(pts.begin(), pts.end(), [](const cv::Point2f &a, const cv::Point2f &b)
              { return a.y < b.y; });
    cv::Point2f top0 = pts[0], top1 = pts[1], bot0 = pts[2], bot1 = pts[3];
    if (top0.x > top1.x)
      std::swap(top0, top1);
    if (bot0.x > bot1.x)
      std::swap(bot0, bot1);
    cv::Mat rect(4, 2, CV_32F);
    rect.at<float>(0, 0) = top0.x;
    rect.at<float>(0, 1) = top0.y;
    rect.at<float>(1, 0) = top1.x;
    rect.at<float>(1, 1) = top1.y;
    rect.at<float>(2, 0) = bot1.x;
    rect.at<float>(2, 1) = bot1.y;
    rect.at<float>(3, 0) = bot0.x;
    rect.at<float>(3, 1) = bot0.y;
    return rect;
  }

  std::optional<CoinFeature> sample_mean_lab_inside_circle(const cv::Mat &frame_bgr,
                                                           cv::Point2i center, int radius_px)
  {
    int h = frame_bgr.rows, w = frame_bgr.cols;
    int cx = center.x, cy = center.y;
    int inner_r = std::max(2, static_cast<int>(radius_px * 0.7));
    int y0 = std::max(0, cy - inner_r), y1 = std::min(h, cy + inner_r + 1);
    int x0 = std::max(0, cx - inner_r), x1 = std::min(w, cx + inner_r + 1);
    if (y1 <= y0 || x1 <= x0)
      return std::nullopt;
    cv::Mat lab;
    cv::cvtColor(frame_bgr, lab, cv::COLOR_BGR2Lab);
    cv::Mat roi = lab(cv::Range(y0, y1), cv::Range(x0, x1));
    cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);
    for (int y = 0; y < roi.rows; ++y)
    {
      for (int x = 0; x < roi.cols; ++x)
      {
        int gx = x0 + x - cx, gy = y0 + y - cy;
        if (gx * gx + gy * gy <= inner_r * inner_r)
          mask.at<uchar>(y, x) = 255;
      }
    }
    cv::Scalar mean_val = cv::mean(roi, mask);
    return CoinFeature{0.0, mean_val[0], mean_val[1], mean_val[2]};
  }

  std::vector<CoinFeature> collect_coin_features(const cv::Mat &frame_bgr,
                                                 const std::vector<std::pair<cv::Point2i, double>> &entries,
                                                 double ratio_px_to_mm)
  {
    std::vector<CoinFeature> rows;
    for (const auto &e : entries)
    {
      int r = diameter_mm_to_radius_px(e.second, ratio_px_to_mm);
      auto lab = sample_mean_lab_inside_circle(frame_bgr, e.first, r);
      if (lab.has_value())
      {
        lab->diameter_mm = e.second;
        rows.push_back(*lab);
      }
    }
    return rows;
  }

}
