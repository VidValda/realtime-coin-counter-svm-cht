#include "config.hpp"
#include "coin_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core/persistence.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>

namespace
{

  struct ManifestRow
  {
    std::string path;
    int class_id;
    double diameter_mm;
  };

  std::vector<ManifestRow> load_manifest(const std::string &path)
  {
    std::vector<ManifestRow> rows;
    std::ifstream in(path);
    if (!in)
      return rows;
    std::string line;
    if (!std::getline(in, line) || line.find("path") == std::string::npos)
      return rows;
    while (std::getline(in, line))
    {
      if (line.empty())
        continue;
      ManifestRow r;
      std::istringstream ss(line);
      if (!std::getline(ss, r.path, ','))
        continue;
      std::string sid, sdiam;
      if (!std::getline(ss, sid, ',') || !std::getline(ss, sdiam))
        continue;
      try
      {
        r.class_id = std::stoi(sid);
        while (!sdiam.empty() && (sdiam.back() == '\r' || sdiam.back() == '\n' || sdiam.back() == ' '))
          sdiam.pop_back();
        r.diameter_mm = std::stod(sdiam);
      }
      catch (...)
      {
        continue;
      }
      rows.push_back(r);
    }
    return rows;
  }

  std::optional<coin::CoinFeature> lab_from_crop(const cv::Mat &crop_bgr, double diameter_mm)
  {
    if (crop_bgr.empty())
      return std::nullopt;
    int cx = crop_bgr.cols / 2;
    int cy = crop_bgr.rows / 2;
    int radius_px = std::max(2, static_cast<int>(std::min(crop_bgr.cols, crop_bgr.rows) * 0.35));
    auto feat = coin::sample_mean_lab_inside_circle(crop_bgr, cv::Point2i(cx, cy), radius_px);
    if (!feat)
      return std::nullopt;
    feat->diameter_mm = diameter_mm;
    return feat;
  }

}

int main()
{
  std::string training_dir = coin::Config::TRAINING_DATA_DIR;
  std::string manifest_path = coin::Config::TRAINING_MANIFEST;

  std::vector<ManifestRow> manifest = load_manifest(manifest_path);
  if (manifest.empty())
  {
    std::cerr << "No manifest or no rows in " << manifest_path << "\n";
    std::cerr << "Run the acquisition tool first: train_acquisition\n";
    return 1;
  }

  std::vector<std::vector<float>> X_train;
  std::vector<int> y_train;

  for (const auto &row : manifest)
  {
    std::string full_path = training_dir + "/" + row.path;
    cv::Mat img = cv::imread(full_path);
    auto feat = lab_from_crop(img, row.diameter_mm);
    if (!feat)
    {
      std::cerr << "Skip (no LAB): " << full_path << "\n";
      continue;
    }
    X_train.push_back({static_cast<float>(feat->diameter_mm), static_cast<float>(feat->L),
                       static_cast<float>(feat->a), static_cast<float>(feat->b)});
    y_train.push_back(row.class_id);
  }

  if (X_train.size() <= 10)
  {
    std::cerr << "Not enough valid samples (" << X_train.size() << "). Need more than 10.\n";
    return 1;
  }

  std::cout << "Loaded " << X_train.size() << " samples from " << training_dir << "\n";

  int n = static_cast<int>(X_train.size());
  cv::Mat X(n, 4, CV_32F);
  cv::Mat y(n, 1, CV_32S);
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < 4; ++j)
      X.at<float>(i, j) = X_train[i][j];
    y.at<int>(i, 0) = y_train[i];
  }

  cv::Mat mean, scale;
  cv::reduce(X, mean, 0, cv::REDUCE_AVG);
  cv::Mat mean_expanded;
  cv::repeat(mean, n, 1, mean_expanded);
  cv::Mat centered;
  cv::subtract(X, mean_expanded, centered);
  cv::Mat sq;
  cv::multiply(centered, centered, sq);
  cv::Mat var;
  cv::reduce(sq, var, 0, cv::REDUCE_AVG);
  cv::sqrt(var, scale);
  for (int j = 0; j < 4; ++j)
    if (scale.at<float>(0, j) < 1e-9f)
      scale.at<float>(0, j) = 1.0f;
  cv::Mat X_scaled(n, 4, CV_32F);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < 4; ++j)
      X_scaled.at<float>(i, j) = (X.at<float>(i, j) - mean.at<float>(0, j)) / scale.at<float>(0, j);

  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 rng(42);
  std::shuffle(indices.begin(), indices.end(), rng);
  int n_test = std::max(1, n / 5);
  int n_train = n - n_test;
  cv::Mat X_train_part(n_train, 4, CV_32F);
  cv::Mat y_train_part(n_train, 1, CV_32S);
  cv::Mat X_test(n_test, 4, CV_32F);
  cv::Mat y_test(n_test, 1, CV_32S);
  for (int i = 0; i < n_train; ++i)
  {
    int idx = indices[i];
    for (int j = 0; j < 4; ++j)
      X_train_part.at<float>(i, j) = X_scaled.at<float>(idx, j);
    y_train_part.at<int>(i, 0) = y.at<int>(idx, 0);
  }
  for (int i = 0; i < n_test; ++i)
  {
    int idx = indices[n_train + i];
    for (int j = 0; j < 4; ++j)
      X_test.at<float>(i, j) = X_scaled.at<float>(idx, j);
    y_test.at<int>(i, 0) = y.at<int>(idx, 0);
  }

  auto eval = [&](cv::Ptr<cv::ml::StatModel> model)
  {
    int correct = 0;
    for (int i = 0; i < n_test; ++i)
    {
      cv::Mat row = X_test.row(i);
      int pred = static_cast<int>(model->predict(row));
      if (pred == y_test.at<int>(i, 0))
        correct++;
    }
    return static_cast<double>(correct) / n_test;
  };

  std::cout << "Testing models on " << n_train << " train / " << n_test << " test samples...\n";

  cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(X_train_part, cv::ml::ROW_SAMPLE, y_train_part);

  cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
  svm->setType(cv::ml::SVM::C_SVC);
  svm->setKernel(cv::ml::SVM::RBF);
  svm->setGamma(0.5);
  svm->setC(1.0);
  svm->train(train_data);
  double score_svm = eval(svm);
  std::cout << "  SVM: " << std::fixed << std::setprecision(4) << score_svm << " accuracy\n";

  cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
  knn->setDefaultK(5);
  knn->train(train_data);
  double score_knn = eval(knn);
  std::cout << "  KNN: " << std::fixed << std::setprecision(4) << score_knn << " accuracy\n";

  cv::Ptr<cv::ml::RTrees> rtrees = cv::ml::RTrees::create();
  rtrees->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 100, 0.01));
  rtrees->train(train_data);
  double score_rf = eval(rtrees);
  std::cout << "  RandomForest: " << std::fixed << std::setprecision(4) << score_rf << " accuracy\n";

  cv::Ptr<cv::ml::NormalBayesClassifier> nb = cv::ml::NormalBayesClassifier::create();
  nb->train(train_data);
  double score_nb = eval(nb);
  std::cout << "  NaiveBayes: " << std::fixed << std::setprecision(4) << score_nb << " accuracy\n";

  double best_score = score_svm;
  int best_idx = 0;
  if (score_knn > best_score)
  {
    best_score = score_knn;
    best_idx = 1;
  }
  if (score_rf > best_score)
  {
    best_score = score_rf;
    best_idx = 2;
  }
  if (score_nb > best_score)
  {
    best_score = score_nb;
    best_idx = 3;
  }

  const char *names[] = {"SVM", "KNN", "RandomForest", "NaiveBayes"};
  std::cout << "Winner: " << names[best_idx] << " (" << std::fixed << std::setprecision(4) << best_score << ")\n";
  std::cout << "Save which model? [1=SVM 2=KNN 3=RandomForest 4=NaiveBayes] (Enter=winner): ";
  std::string line;
  std::getline(std::cin, line);
  int save_idx = best_idx;
  if (!line.empty())
  {
    if (line == "1")
      save_idx = 0;
    else if (line == "2")
      save_idx = 1;
    else if (line == "3")
      save_idx = 2;
    else if (line == "4")
      save_idx = 3;
  }

  cv::Ptr<cv::ml::TrainData> full_data = cv::ml::TrainData::create(X_scaled, cv::ml::ROW_SAMPLE, y);
  svm->train(full_data);
  knn->train(full_data);
  rtrees->train(full_data);
  nb->train(full_data);
  svm->save(coin::Config::CLASSIFIER_MODEL_PATHS[0]);
  knn->save(coin::Config::CLASSIFIER_MODEL_PATHS[1]);
  rtrees->save(coin::Config::CLASSIFIER_MODEL_PATHS[2]);
  nb->save(coin::Config::CLASSIFIER_MODEL_PATHS[3]);

  cv::FileStorage fs(coin::Config::SVM_SCALER_PATH, cv::FileStorage::WRITE);
  fs << "mean" << mean;
  fs << "scale" << scale;
  fs << "model_type" << names[save_idx];
  fs.release();

  std::ofstream default_file(coin::Config::CLASSIFIER_DEFAULT_FILE);
  if (default_file)
    default_file << save_idx << "\n";
  std::cout << "Saved all 4 models. Active (default): " << names[save_idx] << " ("
            << coin::Config::CLASSIFIER_MODEL_PATHS[save_idx] << ").\n";
  return 0;
}
