cmake_minimum_required(VERSION 3.14)
project(coin_counter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

set(COMMON_SOURCES
  src/calibration.cpp
  src/svm_classifier.cpp
  src/corner_stabilizer.cpp
  src/coin_track.cpp
  src/coin_tracker.cpp
  src/coin_detector.cpp
)

add_library(coin_common ${COMMON_SOURCES})
target_include_directories(coin_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(coin_common PUBLIC ${OpenCV_LIBS})

add_executable(coin_counter src/main_coin_counter.cpp)
target_link_libraries(coin_counter PRIVATE coin_common)

add_executable(train_acquisition src/main_train_acquisition.cpp)
target_link_libraries(train_acquisition PRIVATE coin_common)

add_executable(train_svm src/main_train_svm.cpp)
target_link_libraries(train_svm PRIVATE coin_common)

add_executable(camera_calibration src/main_camera_calibration.cpp)
target_link_libraries(camera_calibration PRIVATE coin_common)
