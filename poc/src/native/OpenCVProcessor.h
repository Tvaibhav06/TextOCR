#pragma once

#include <string>
#include <memory>
#include <jni.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>

namespace facebook {
namespace react {

class OpenCVProcessor {
public:
  static void decodeMask(
      const std::string& base64,
      int width,
      int height,
      cv::Mat& outMask);

  static std::string inpaintImage(
      const std::string& imageUri,
      const std::string& maskBase64,
      int radius,
      int method,
      const std::string& tempPath);

  static std::string blendImages(
      const std::string& inpaintedUri,
      const std::string& originalUri,
      const std::string& maskBase64,
      const std::string& tempPath);

  static std::string resolveUri(const std::string& uri);
  static std::string allocTempPath();
  static std::string writeMat(const cv::Mat& mat, const std::string& path);
};

} // namespace react
} // namespace facebook
