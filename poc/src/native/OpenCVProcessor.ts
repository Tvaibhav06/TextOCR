/**
 * OpenCVProcessor — Native TurboModule (JSI) TypeScript Interface
 *
 * This file documents the contract that the C++ native module must satisfy.
 * It is imported via NativeModules.OpenCVProcessor in PostProcessor.ts.
 *
 * Platform implementations:
 *   Android: src/native/OpenCVProcessor.cpp  (JNI / JSI via CMakeLists.txt)
 *   iOS:     src/native/ios/OpenCVProcessor.mm (Obj-C++ / RCTBridgeModule)
 *
 * ─── Module Operations ────────────────────────────────────────────────────────
 *
 *  processImage(imageUri, maskBase64, params) → Promise<string>
 *    Main entry point. Dispatches to the correct OpenCV operation based on
 *    params.method. Returns the local file URI of the output image.
 *
 *    params.method options:
 *      'blend'          → cv::seamlessClone (NORMAL_CLONE) at mask boundary
 *      'dilate'         → cv::dilate with disk kernel of params.kernelSize px
 *      'inpaint_telea'  → cv::inpaint with INPAINT_TELEA
 *      'inpaint_ns'     → cv::inpaint with INPAINT_NS
 *
 *    All operations write their result to the app's temp directory and return
 *    a file:// URI. The caller is responsible for cleaning up temp files.
 *
 * ─── Data format notes ────────────────────────────────────────────────────────
 *
 *  imageUri:   local file:// URI, e.g. "file:///data/.../image.jpg"
 *  maskBase64: base64-encoded RGBA PNG where R-channel = mask value (0 or 255)
 *              Encoding: btoa() — Hermes-safe, no Buffer dependency
 *
 * ─── Track 2 Stub ─────────────────────────────────────────────────────────────
 *
 *  The C++ stub (Track 2) returns imageUri unchanged for all methods.
 *  Real OpenCV dispatch is activated in Track 3 (pipeline wiring sprint).
 */

export interface OpenCVProcessorParams {
  method: 'blend' | 'dilate' | 'inpaint_telea' | 'inpaint_ns';
  /** For 'blend': URI of the original (pre-inpaint) image */
  originalUri?: string;
  /** For 'inpaint_telea' | 'inpaint_ns': neighbourhood radius (default 5) */
  radius?: number;
  /** For 'dilate': kernel size in pixels (default 3) */
  kernelSize?: number;
}

export interface OpenCVProcessorModule {
  /**
   * Dispatch an OpenCV image-processing operation.
   * @param imageUri   - Input image URI (local file://)
   * @param maskBase64 - Base64-encoded RGBA PNG mask
   * @param params     - Operation parameters
   * @returns          - Output image URI (local file://)
   */
  processImage(
    imageUri: string,
    maskBase64: string,
    params: OpenCVProcessorParams,
  ): Promise<string>;
}

/**
 * Type-safe accessor for the native module.
 * Usage:
 *   import { NativeModules } from 'react-native';
 *   const mod = NativeModules.OpenCVProcessor as OpenCVProcessorModule;
 */
export type { OpenCVProcessorModule as default };
