cmake_minimum_required(VERSION 3.14)
project(coin_counter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Prefer system OpenCV (e.g. from apt) to avoid link errors when /usr/local OpenCV
# was built with OpenEXR 2.3 but the system has OpenEXR 2.5. Set to ON if you get
# "undefined reference to `Imf_2_3::'" errors.
option(USE_SYSTEM_OPENCV "Prefer system OpenCV (e.g. /usr) over /usr/local" ON)
if(USE_SYSTEM_OPENCV)
  set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
  if(NOT EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
    set(OpenCV_DIR "")
  endif()
endif()
find_package(OpenCV REQUIRED)

# LibTorch (optional): set USE_TORCH=ON and point to LibTorch so find_package(Torch) can find it.
# Example: cmake -DUSE_TORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
# Or:     cmake -DUSE_TORCH=ON -DLibTorch_DIR=/path/to/libtorch/share/cmake/Torch ..
option(USE_TORCH "Build with LibTorch for CNN/ResNet classifiers" OFF)
if(USE_TORCH)
  if(DEFINED LibTorch_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${LibTorch_DIR}")
  endif()
  find_package(Torch REQUIRED)
  message(STATUS "Building with LibTorch: ${Torch_DIR}")
else()
  find_package(Torch QUIET)
  if(Torch_FOUND)
    message(STATUS "LibTorch found; building with Torch support")
  endif()
endif()

# Link all standard OpenCV modules we need (core, imgproc, videoio, highgui, ml, imgcodecs).
# With system OpenCV this avoids OpenEXR version mismatch; with /usr/local OpenCV you may
# need OpenEXR 2.3 or -DUSE_SYSTEM_OPENCV=ON.
set(OPENCV_LIBS_COIN opencv_core opencv_imgproc opencv_videoio opencv_highgui opencv_ml opencv_imgcodecs)

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
target_link_libraries(coin_common PUBLIC ${OPENCV_LIBS_COIN})

if(Torch_FOUND)
  target_sources(coin_common PRIVATE src/torch_classifier.cpp)
  # Prefer imported target (LibTorch 1.9+); fallback for older configs
  if(TARGET torch::torch)
    target_link_libraries(coin_common PUBLIC torch::torch)
  else()
    target_link_libraries(coin_common PUBLIC ${TORCH_LIBRARIES})
  endif()
  target_compile_definitions(coin_common PUBLIC COIN_USE_TORCH=1)
endif()

add_executable(coin_counter src/main_coin_counter.cpp)
target_link_libraries(coin_counter PRIVATE coin_common)

add_executable(camera_calibration src/main_camera_calibration.cpp)
target_link_libraries(camera_calibration PRIVATE coin_common)

add_executable(train_acquisition src/main_train_acquisition.cpp)
target_link_libraries(train_acquisition PRIVATE coin_common)

add_executable(train_svm src/main_train_svm.cpp)
target_link_libraries(train_svm PRIVATE coin_common)
