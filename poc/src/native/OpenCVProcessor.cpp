#include "OpenCVProcessor.h"
#include <android/log.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <string>
#include <vector>

#define LOG_TAG "OpenCVProcessor"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace facebook {
namespace react {

void OpenCVProcessor::decodeMask(
    const std::string& base64,
    int width,
    int height,
    cv::Mat& outMask) {

  static const std::string B64CHARS =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::vector<int> lookup(256, -1);
  for (int i = 0; i < 64; i++) lookup[static_cast<unsigned char>(B64CHARS[i])] = i;

  std::vector<uint8_t> decoded;
  decoded.reserve(base64.size() * 3 / 4);

  int val = 0, valb = -8;
  for (unsigned char c : base64) {
    if (lookup[c] == -1) continue;
    val   = (val << 6) + lookup[c];
    valb += 6;
    if (valb >= 0) {
      decoded.push_back(static_cast<uint8_t>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }

  const int expectedBytes = width * height * 4;
  outMask = cv::Mat::zeros(height, width, CV_8UC1);

  const int pixelCount = width * height;
  const int availBytes = static_cast<int>(decoded.size());

  for (int i = 0; i < pixelCount; i++) {
    const int byteIdx = i * 4;
    if (byteIdx >= availBytes) break;
    outMask.at<uint8_t>(i / width, i % width) = decoded[byteIdx];
  }
}

std::string OpenCVProcessor::inpaintImage(
    const std::string& imageUri,
    const std::string& maskBase64,
    int radius,
    int method,
    const std::string& tempPath) {

  const std::string srcPath = resolveUri(imageUri);
  cv::Mat src = cv::imread(srcPath, cv::IMREAD_COLOR);
  if (src.empty()) {
    throw std::runtime_error("inpaintImage: could not read image: " + srcPath);
  }

  cv::Mat mask;
  decodeMask(maskBase64, src.cols, src.rows, mask);

  if (cv::countNonZero(mask) == 0) {
    return imageUri;
  }

  cv::Mat dst;
  cv::inpaint(src, mask, dst, radius, method);
  return writeMat(dst, tempPath);
}

std::string OpenCVProcessor::blendImages(
    const std::string& inpaintedUri,
    const std::string& originalUri,
    const std::string& maskBase64,
    const std::string& tempPath) {

  const std::string inpPath  = resolveUri(inpaintedUri);
  const std::string origPath = resolveUri(originalUri);

  cv::Mat src = cv::imread(inpPath,  cv::IMREAD_COLOR);
  cv::Mat dst = cv::imread(origPath, cv::IMREAD_COLOR);

  if (src.empty()) throw std::runtime_error("blendImages: could not read inpainted: " + inpPath);
  if (dst.empty()) throw std::runtime_error("blendImages: could not read original: "  + origPath);

  if (src.size() != dst.size()) {
    cv::resize(src, src, dst.size(), 0, 0, cv::INTER_LINEAR);
  }

  cv::Mat mask8u;
  decodeMask(maskBase64, dst.cols, dst.rows, mask8u);

  if (cv::countNonZero(mask8u) == 0) {
    return inpaintedUri;
  }

  cv::Rect bounds = cv::boundingRect(mask8u);
  cv::Point center(
      bounds.x + bounds.width  / 2,
      bounds.y + bounds.height / 2);

  cv::Mat blended;
  try {
    cv::seamlessClone(src, dst, mask8u, center, blended, cv::NORMAL_CLONE);
  } catch (const cv::Exception& e) {
    return inpaintedUri;
  }

  return writeMat(blended, tempPath);
}

std::string OpenCVProcessor::resolveUri(const std::string& uri) {
  if (uri.substr(0, 8) == "file:///") return uri.substr(7);
  if (uri.substr(0, 7) == "file://")  return uri.substr(7);
  return uri;
}

std::string OpenCVProcessor::allocTempPath() {
  const char* tmpdir = std::getenv("TMPDIR");
  if (!tmpdir || tmpdir[0] == '\0') tmpdir = "/data/local/tmp";

  const auto now = std::chrono::high_resolution_clock::now();
  const auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(
      now.time_since_epoch()).count();

  return std::string(tmpdir) + "/opencv_" + std::to_string(ns) + ".jpg";
}

std::string OpenCVProcessor::writeMat(const cv::Mat& mat, const std::string& path) {
  std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 92 };
  if (!cv::imwrite(path, mat, params)) {
    throw std::runtime_error("writeMat: cv::imwrite failed for path: " + path);
  }
  return "file://" + path;
}

} // namespace react
} // namespace facebook

// ─── Standard JNI Bridge ───────────────────────────────────────────────────────

extern "C" JNIEXPORT jstring JNICALL
Java_com_anonymous_textremoverpoc_OpenCVProcessorModule_nativeProcessImage(
    JNIEnv* env,
    jobject /* this */,
    jstring imageUriStr,
    jstring maskBase64Str,
    jstring methodStr,
    jstring originalUriStr,
    jint radius,
    jint kernelSize) {

  const char* imageUriChars = env->GetStringUTFChars(imageUriStr, nullptr);
  const char* maskBase64Chars = env->GetStringUTFChars(maskBase64Str, nullptr);
  const char* methodChars = env->GetStringUTFChars(methodStr, nullptr);
  const char* originalUriChars = env->GetStringUTFChars(originalUriStr, nullptr);

  std::string imageUri = imageUriChars;
  std::string maskBase64 = maskBase64Chars;
  std::string method = methodChars;
  std::string originalUri = originalUriChars;

  env->ReleaseStringUTFChars(imageUriStr, imageUriChars);
  env->ReleaseStringUTFChars(maskBase64Str, maskBase64Chars);
  env->ReleaseStringUTFChars(methodStr, methodChars);
  env->ReleaseStringUTFChars(originalUriStr, originalUriChars);

  std::string result;
  
  try {
    std::string tempPath = facebook::react::OpenCVProcessor::allocTempPath();

    if (method == "inpaint_telea") {
      result = facebook::react::OpenCVProcessor::inpaintImage(imageUri, maskBase64, radius, cv::INPAINT_TELEA, tempPath);
    } else if (method == "inpaint_ns") {
      result = facebook::react::OpenCVProcessor::inpaintImage(imageUri, maskBase64, radius, cv::INPAINT_NS, tempPath);
    } else if (method == "blend") {
      if (originalUri.empty()) {
        result = imageUri;
      } else {
        result = facebook::react::OpenCVProcessor::blendImages(imageUri, originalUri, maskBase64, tempPath);
      }
    } else if (method == "dilate") {
      const std::string srcPath = facebook::react::OpenCVProcessor::resolveUri(imageUri);
      cv::Mat img = cv::imread(srcPath, cv::IMREAD_COLOR);
      if (img.empty()) throw std::runtime_error("dilate: could not read image");

      cv::Mat mask;
      facebook::react::OpenCVProcessor::decodeMask(maskBase64, img.cols, img.rows, mask);
      cv::Mat kernel = cv::getStructuringElement(
          cv::MORPH_ELLIPSE,
          cv::Size(kernelSize * 2 + 1, kernelSize * 2 + 1));
      cv::Mat dilatedMask;
      cv::dilate(mask, dilatedMask, kernel);

      img.setTo(cv::Scalar(0, 0, 0), dilatedMask);
      result = facebook::react::OpenCVProcessor::writeMat(img, tempPath);
    } else {
      result = imageUri;
    }
  } catch (const std::exception& e) {
    env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
    return nullptr;
  }

  return env->NewStringUTF(result.c_str());
}
