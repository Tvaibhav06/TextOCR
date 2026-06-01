package com.anonymous.textremoverpoc

import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReadableMap

class OpenCVProcessorModule(reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

  init {
    System.loadLibrary("RNOpenCVProcessor")
  }

  override fun getName(): String {
    return NAME
  }

  @ReactMethod
  fun processImage(imageUri: String, maskBase64: String, params: ReadableMap, promise: Promise) {
    Thread {
      try {
        val method = if (params.hasKey("method")) params.getString("method") ?: "inpaint_telea" else "inpaint_telea"
        val originalUri = if (params.hasKey("originalUri")) params.getString("originalUri") ?: "" else ""
        val radius = if (params.hasKey("radius")) params.getInt("radius") else 5
        val kernelSize = if (params.hasKey("kernelSize")) params.getInt("kernelSize") else 3

        val resultUri = nativeProcessImage(imageUri, maskBase64, method, originalUri, radius, kernelSize)
        promise.resolve(resultUri)
      } catch (e: Exception) {
        promise.reject("OPENCV_ERROR", e.message, e)
      }
    }.start()
  }

  private external fun nativeProcessImage(
    imageUri: String,
    maskBase64: String,
    method: String,
    originalUri: String,
    radius: Int,
    kernelSize: Int
  ): String

  companion object {
    const val NAME = "OpenCVProcessor"
  }
}
