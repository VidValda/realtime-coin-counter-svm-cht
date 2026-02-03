#include "coin_detector.hpp"
#include "config.hpp"
#include "calibration.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>

namespace coin
{

  cv::Mat preprocess_for_circles(const cv::Mat &frame)
  {
    cv::Mat gray, enhanced, blurred;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(Config::CLAHE_CLIP, cv::Size(Config::CLAHE_GRID, Config::CLAHE_GRID));
    clahe->apply(gray, enhanced);
    cv::medianBlur(enhanced, blurred, Config::BLUR_KSIZE);
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

  Detections detect_and_measure_coins(const cv::Mat &frame, double ratio_px_to_mm)
  {
    cv::Mat blurred = preprocess_for_circles(frame);
    cv::Mat edges;
    cv::Canny(blurred, edges, Config::CANNY_THRESHOLD1, Config::CANNY_THRESHOLD2);
    std::vector<cv::Vec3f> circles = find_circle_candidates(edges);
    if (circles.empty())
      return {};
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    Detections detections;
    for (const auto &c : circles)
    {
      int x = static_cast<int>(std::round(c[0]));
      int y = static_cast<int>(std::round(c[1]));
      int r = static_cast<int>(std::round(c[2]));
      auto result = measure_circle_diameter(gray, x, y, r, ratio_px_to_mm);
      if (result.has_value())
        detections.push_back(*result);
    }
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
    cv::Mat lab, l_ch;
    cv::cvtColor(frame, lab, 44);
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    l_ch = channels[0];
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(0);
    std::vector<cv::Vec4f> lines;
    lsd->detect(l_ch, lines);
    cv::Mat line_mask = cv::Mat::zeros(l_ch.size(), CV_8UC1);
    for (const auto &line : lines)
    {
      double dx = line[2] - line[0], dy = line[3] - line[1];
      if (std::hypot(dx, dy) > 100)
      {
        cv::line(line_mask, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), 255, 3);
      }
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
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
      cv::approxPolyDP(cnt, approx, 0.02 * peri, true);
      if (approx.size() == 4 && cv::contourArea(approx) > 10000)
      {
        cv::Mat corners(4, 2, CV_32F);
        for (int j = 0; j < 4; ++j)
        {
          corners.at<float>(j, 0) = static_cast<float>(approx[j].x);
          corners.at<float>(j, 1) = static_cast<float>(approx[j].y);
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
    cv::cvtColor(frame_bgr, lab, 44);
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
