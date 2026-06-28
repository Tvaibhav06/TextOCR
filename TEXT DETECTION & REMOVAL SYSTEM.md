**TEXT DETECTION & REMOVAL SYSTEM**

_Mobile Application Edition_

Product Requirements Document (PRD) + Technical Requirements Document (TRD)

Version 2.0 | April 2026 | React Native / Expo

**PART A - PRODUCT REQUIREMENTS DOCUMENT (PRD)**

# **1\. Executive Summary**

This document defines the product and technical requirements for the mobile edition of the Text Detection and Removal System (TDRS Mobile). Building on the architecture established in the server-side TDRS v1.0, this version targets iOS and Android mobile devices via a React Native / Expo application. The system allows any user with a smartphone to photograph, import, or record media containing unwanted text - watermarks, burnt-in subtitles, licence plates, brand overlays - and receive a clean, text-free output directly on their device.

Where the original TDRS v1.0 relied on heavyweight server libraries (DBNet++, LaMa, FFmpegKit full build), this mobile edition is re-architected around a tiered on-device-first strategy using lightweight, open-source libraries. Simple cases are resolved entirely on-device in under two seconds; only large or complex masks are escalated to a thin FastAPI backend.

# **2\. Problem Statement**

Users across media production, legal, e-commerce, and social content creation increasingly need a fast, private, and accessible tool to remove text from images and short video clips on their mobile device - without uploading sensitive media to opaque third-party services or waiting for a desktop workflow. Existing mobile tools either produce visible artifacts, support only a narrow range of scripts, or require manual selection that defeats the purpose on a small touchscreen.

A language-agnostic, fully automated, privacy-first pipeline that runs primarily on-device - escalating to a server only when the hardware cannot cope - addresses this gap.

# **3\. Goals & Non-Goals**

## **3.1 Goals**

- Detect text in any script on a mobile device: Latin, Devanagari, Arabic, CJK, Hebrew, Cyrillic, and all scripts supported by ML Kit Text Recognition v2.
- Generate pixel-accurate binary masks for all detected regions using React Native Skia.
- Reconstruct background on-device using a tiered inpainting strategy (OpenCV Telea → PConv TFLite → server LaMa) matched to mask complexity.
- Support static images (JPG, PNG, WEBP, HEIC) and short video clips (MP4, MOV up to 60 seconds).
- Deliver a React Native / Expo application for iOS 15+ and Android 11+ (API level 30+).
- Process a 1080p image end-to-end in under 3 seconds on a mid-range device (e.g. iPhone 12, Pixel 6) for masks covering less than 15% of image area.
- Keep on-device model bundle under 40 MB total.
- No persistent storage of user media - ephemeral processing only.

## **3.2 Non-Goals**

- Real-time live-camera text erasure (v2 roadmap item).
- Video clips longer than 60 seconds in v1.
- OCR / text transcription - the system detects and erases, not reads.
- Handwriting-specific fine-tuning.
- Offline operation for large-mask cases (server connection required for LaMa fallback).

# **4\. User Stories**

| **ID** | **User**                | **Requirement**                                                              |
| ------ | ----------------------- | ---------------------------------------------------------------------------- |
| US-01  | Social media creator    | Remove watermarks from reference images before posting mood boards           |
| US-02  | Legal / HR professional | Redact licence plates and personal text from mobile-captured evidence photos |
| US-03  | Traveller               | Erase foreign-language overlays from screenshots to share cleanly            |
| US-04  | E-commerce seller       | Strip competitor watermarks from product reference images on the go          |
| US-05  | Video editor (field)    | Remove burnt-in date/time stamps from short phone-recorded clips             |
| US-06  | Privacy-conscious user  | Process everything on-device without media leaving the phone                 |
| US-07  | Developer               | Call the REST API programmatically from another mobile app                   |

# **5\. Functional Requirements**

## **5.1 Input Handling**

- FR-1.1: Accept images via device camera, photo library, or share extension: JPG, PNG, WEBP, HEIC up to 24 MP.
- FR-1.2: Accept short video clips from library or camera: MP4 and MOV up to 60 seconds, up to 4K resolution.
- FR-1.3: Accept URL input (share-sheet URL) pointing to publicly accessible image media.
- FR-1.4: Accept in-app camera capture via Vision Camera Frame Processor for single-shot processing.

## **5.2 Detection**

- FR-2.1: Detect horizontal, vertical, diagonal, and curved text.
- FR-2.2: Detect text at arbitrary scale (minimum 8 px height to full-frame banners).
- FR-2.3: Detect text in low-contrast, noisy, and compressed images.
- FR-2.4: Produce bounding polygons (N-point, N ≥ 4) for each text instance.
- FR-2.5: Assign a confidence score \[0, 1\] to each detected region.
- FR-2.6: Support 100+ language scripts via the ML Kit + TFLite EAST ensemble.

## **5.3 Mask Generation**

- FR-3.1: Produce a binary pixel mask (255 = text, 0 = background) at input resolution via Skia Path.
- FR-3.2: Apply configurable Gaussian blur dilation as a soft-edge proxy (default sigma = 3 px).
- FR-3.3: Optionally expose mask as a downloadable PNG alongside the cleaned image.

## **5.4 Inpainting**

- FR-4.1: Apply the tiered inpainting strategy - Telea for small masks, PConv for medium masks, server LaMa for large masks.
- FR-4.2: Preserve textures, gradients, and lighting in reconstructed regions.
- FR-4.3: Handle masks up to 25% of image area on-device; escalate larger masks to server.

## **5.5 Output**

- FR-5.1: Return cleaned image in the same format and resolution as input.
- FR-5.2: Optionally return binary mask PNG, bounding polygon JSON, and confidence scores.
- FR-5.3: For video: return cleaned clip with original audio preserved, same codec.
- FR-5.4: Share output via native iOS/Android share sheet.

## **5.6 Configuration**

- FR-6.1: Adjustable detection confidence threshold (default 0.5; range 0.1-1.0) via in-app slider.
- FR-6.2: Toggle to generate mask only without inpainting (for power users / developers).
- FR-6.3: Tiering override: force on-device or force server for any request.
- FR-6.4: Optional API key entry for server-side LaMa tier.

# **6\. Non-Functional Requirements**

| **Category**  | **Requirement**                                | **Target**                       |
| ------------- | ---------------------------------------------- | -------------------------------- |
| Performance   | 1080p image, mask < 15% area, on-device        | < 3 sec on mid-range device      |
| Performance   | 1080p image, mask 15-25% area, on-device PConv | < 6 sec on mid-range device      |
| Performance   | 1080p image, mask > 25%, server LaMa           | < 8 sec incl. network round-trip |
| Performance   | Video throughput (60-sec 1080p clip)           | < 90 sec on-device (1× realtime) |
| Accuracy      | Text detection F1 (ML Kit, ICDAR 2015)         | \> 85%                           |
| Accuracy      | Inpainting SSIM on background region           | \> 0.90 on-device; > 0.92 server |
| Accuracy      | Inpainting FID score (server tier)             | < 15                             |
| App size      | Total installed size delta                     | < 55 MB (models + native libs)   |
| Battery       | Avg power draw per 1080p process               | < 500 mJ (no sustained thermal)  |
| Privacy       | Media storage post-processing                  | Ephemeral only; zero persistence |
| Reliability   | App crash rate                                 | < 0.5% of sessions               |
| Compatibility | iOS / Android minimum                          | iOS 15+ / Android 11+ (API 30+)  |

# **7\. Success Metrics**

- F1 score > 0.85 on ICDAR 2015 running ML Kit + TFLite EAST ensemble on device.
- SSIM > 0.90 on inpainted regions (on-device PConv); > 0.92 server LaMa, measured on synthetic text removal test set.
- FID < 15 on server-side inpainted outputs (Inception-v3 activations).
- PSNR > 28 dB on clean synthetic test images processed fully on-device.
- 95th-percentile processing time < 5 seconds for 1080p images, mask < 15%.
- Zero media files retained post-session (verified by automated privacy audit).
- App Store / Play Store rating ≥ 4.3 at 1,000 reviews.
- < 0.5% crash rate across iOS and Android (measured via Sentry or equivalent).

**PART B - TECHNICAL REQUIREMENTS DOCUMENT (TRD)**

# **1\. System Architecture Overview**

The mobile TDRS is a five-stage sequential pipeline running inside a React Native / Expo application. Each stage maps to a specific set of mobile-native libraries. The pipeline is modular - each stage is independently versioned and replaceable. GPU-accelerated stages (mask generation, Poisson blending) run on the device GPU via Skia and WebGL; neural stages run via TFLite delegates (NNAPI on Android, Core ML on iOS).

**Pipeline:** INPUT → \[Stage 1: Preprocessing\] → \[Stage 2: Text Detection\] → \[Stage 3: Mask Generation\] → \[Stage 4: Inpainting\] → \[Stage 5: Post-processing\] → OUTPUT

**Key insight:** Inpainting tiering decision: mask area &lt; 5% → OpenCV Telea (on-device, CPU) | 5-25% → PConv TFLite INT8 (on-device, NNAPI/CoreML) | &gt; 25% → FastAPI server (LaMa or SD XL)

# **2\. Stage 1 - Input Preprocessing**

## **2.1 Image Acquisition & Normalisation**

All inputs are acquired via react-native-image-picker (photo library and camera) or the Vision Camera Frame Processor plugin for in-app capture. Images are then normalised to a canonical float32 RGB representation via expo-image-manipulator:

- Strip alpha channel; flatten onto white background for PNG inputs with transparency.
- Convert HEIC to JPEG via expo-image-manipulator compress/format transform before further processing.
- Resize to processing resolution: images larger than 2048 px on any axis are downscaled proportionally. Full resolution is preserved for final output compositing.
- Normalise pixel values to \[0, 1\] float32 for TFLite model inputs.

## **2.2 Tiling Strategy for Large Images**

Images wider or taller than 2048 px (after picker acquisition) are split into overlapping 512 × 512 tiles with a 64 px overlap margin. After per-tile detection and inpainting, tiles are stitched using a Gaussian blend in the overlap zone to prevent seam artifacts, implemented as a WebGL shader in expo-gl:

Blend weight at position x within overlap: w(x) = exp(−x² / 2σ²) where σ = overlap_size / 3 Final pixel value: P = (w₁·P₁ + w₂·P₂) / (w₁ + w₂)

## **2.3 Video Frame Extraction**

For video inputs, FFmpegKit (min build, video demux codecs only, ~12 MB binary) extracts frames at the native frame rate or a configurable stride (default every 2nd frame for clips > 30 fps). Extracted frames are written to the app cache directory as JPEG thumbnails for per-frame processing, then re-muxed with the original audio track on completion.

# **3\. Stage 2 - Text Detection**

## **3.1 Ensemble Strategy**

Text detection runs as an ensemble of three complementary models. Their bounding polygons are merged using Non-Maximum Suppression (NMS) at an IoU threshold of 0.5 before mask generation. The ensemble is executed sequentially on a background thread to avoid blocking the UI:

| **Model**                  | **Library**                  | **Strengths**                              | **Size**          |
| -------------------------- | ---------------------------- | ------------------------------------------ | ----------------- |
| ML Kit Text Recognition v2 | Google ML Kit (on-device)    | 100+ scripts, on-device, fast, no quota    | ~5 MB bundled     |
| EAST (quant. INT8)         | TensorFlow Lite React Native | Arbitrary-angle text, textured backgrounds | ~8 MB model       |
| NanoDet-Plus (optional)    | TensorFlow Lite React Native | Fast CPU inference for dense text regions  | ~5 MB model       |
| MediaPipe Text Detection   | MediaPipe Tasks Vision       | Lightweight; strong on printed text        | ~3 MB task bundle |

## **3.2 ML Kit Text Recognition v2**

ML Kit serves as the primary detector. It runs entirely on-device, requires no API key or network connection, and natively supports Latin, Devanagari, Japanese, Korean, Chinese, Arabic, Hebrew, and Cyrillic scripts. It returns bounding boxes and confidence scores per text block, line, and element.

- Integration: @react-native-google-mlkit/text-recognition
- Output: Array of TextBlock objects with bounding polygon, text content, and confidence.
- Latency: 100-350 ms on a Pixel 6 for a 1080p image.

## **3.3 TFLite EAST / NanoDet**

For cases where ML Kit misses text (e.g. rotated overlays, low-contrast watermarks), a quantised INT8 EAST model provides a second opinion. The TFLite model runs via the react-native-fast-tflite library using NNAPI delegate on Android and Core ML delegate on iOS.

- Model input: 512 × 512 float32 RGB, normalised to \[−1, 1\].
- Model output: score map and geometry map at stride 4; decoded to rotated bounding rectangles.
- NMS post-processing: Locality-Sensitive NMS at IoU threshold 0.3.

EAST score threshold: 0.5 default | NMS IoU: 0.3 | Output: rotated bounding rectangles converted to 4-point polygons before ensemble merge

## **3.4 MediaPipe Text Detection**

MediaPipe Tasks Vision is included as a lightweight supplementary detector, particularly effective on printed and signage text. It uses the @mediapipe/tasks-vision package compiled for React Native via the mediapipe-react-native wrapper. It produces axis-aligned bounding boxes that are converted to 4-point polygons before NMS.

## **3.5 Ensemble NMS**

Polygons from all three detectors are unioned. Duplicates are resolved by computing pairwise polygon IoU and suppressing lower-confidence detections when IoU > 0.5. The surviving set of polygons, each tagged with its source model and confidence score, is passed to Stage 3.

# **4\. Stage 3 - Mask Generation**

## **4.1 Polygon Rasterisation via React Native Skia**

Each detected text polygon is rasterised onto a binary mask using React Native Skia's Path API. Skia renders directly on the GPU, making this stage significantly faster than the OpenCV fillPoly approach used in v1.0. The mask is the same spatial dimensions as the (potentially downscaled) processing image.

const mask = Skia.Surface.Make(width, height); const canvas = mask.getCanvas(); const paint = Skia.Paint(); paint.setColor(Skia.Color('white')); for (const poly of polygons) { const path = Skia.Path.Make(); path.moveTo(poly\[0\].x, poly\[0\].y); poly.slice(1).forEach(p => path.lineTo(p.x, p.y)); path.close(); canvas.drawPath(path, paint); }

## **4.2 Soft Dilation via Gaussian Blur**

Rather than OpenCV morphological dilation, the mobile implementation applies a Skia image filter blur pass to the rasterised mask. A Gaussian blur with sigma proportional to text height serves as a soft-edge dilation proxy, which blends more naturally at mask boundaries and requires no additional native dependencies.

Blur sigma: σ = 3 px for text height < 20 px | σ = 5 px for text height ≥ 20 px Implementation: paint.setImageFilter(Skia.ImageFilter.MakeBlur(σ, σ, TileMode.Decal, null))

## **4.3 Confidence-Weighted Thresholding**

Detections below the configurable threshold θ are discarded before rasterisation. The default θ = 0.5 is exposed as a user-adjustable slider in the app UI. For forensic/legal use cases, θ = 0.3 maximises recall. For precision-sensitive use cases (e.g. clean product photos), θ = 0.7 is recommended.

# **5\. Stage 4 - Inpainting (Tiered Strategy)**

Inpainting is the most computationally and memory-intensive stage. The mobile edition implements a three-tier strategy that keeps the majority of requests fully on-device while gracefully escalating only when hardware constraints require it.

| **Tier**           | **Condition**   | **Method**                    | **Library**              | **Avg latency (1080p)** |
| ------------------ | --------------- | ----------------------------- | ------------------------ | ----------------------- |
| 1 On-device fast   | Mask area < 5%  | OpenCV Telea / NS (classical) | opencv-react-native      | < 0.5 sec               |
| 2 On-device neural | Mask area 5-25% | PConv INT8 TFLite (256×256)   | react-native-fast-tflite | 1-3 sec                 |
| 3 Server           | Mask area > 25% | LaMa (FastAPI) or SD XL       | Fetch API → FastAPI      | 4-8 sec incl. network   |

## **5.1 Tier 1 - OpenCV Telea / Navier-Stokes (on-device)**

For small text regions (watermark text, thin captions) covering less than 5% of image area, the classical OpenCV inpainting algorithms are fast, require no model, and produce acceptable results on uniform or mildly textured backgrounds. The opencv-react-native package provides bindings to cv2.inpaint with both INPAINT_TELEA and INPAINT_NS flag options.

- INPAINT_TELEA: Fast marching method; preferred for most cases. Radius parameter r = 5 px.
- INPAINT_NS: Navier-Stokes fluid model; better on smoother backgrounds. Radius r = 7 px.
- Tiering condition: mask_coverage(mask, width, height) < 0.05.

## **5.2 Tier 2 - PConv TFLite INT8 (on-device)**

For medium masks (5-25% coverage), a Partial Convolutions model pruned and quantised to INT8 is loaded via react-native-fast-tflite. The model is trained at 256 × 256 and operates on tiled crops around each masked region, reducing peak memory usage. The PConv renormalization formula ensures that convolutional output accounts for the proportion of valid (unmasked) pixels:

Partial Conv: x_out = Wᵀ · (X ⊙ M) · (|K²| / sum(M)) + b Mask update: M'\_out(i,j) = 1 if sum(M in receptive field) > 0 else 0 Model size: ~12 MB INT8 | Input: 256×256 float32 RGB + mask | NNAPI delegate (Android) / CoreML (iOS)

## **5.3 Tier 3 - Server LaMa / Stable Diffusion XL (off-device)**

For large or complex masks, the app sends only the masked crop (not the full image) to the FastAPI backend running LaMa or Stable Diffusion XL Inpaint. Sending the crop rather than the full resolution image reduces payload size by 10-20× on typical text removal cases. The server returns the inpainted crop, which is composited back onto the full image locally.

- Endpoint: POST /v1/process/image (same API contract as TDRS v1.0).
- Payload: base64-encoded crop image + binary mask, both at 1024 × 1024 max.
- Authentication: API key in Authorization header; HTTPS / TLS 1.3 required.
- Timeout: 15 seconds client-side; retry once on timeout before surfacing error to user.

# **6\. Stage 5 - Post-processing**

## **6.1 Poisson Blending via expo-gl (WebGL)**

Even high-quality inpainting can leave visible boundary artifacts due to JPEG block effects or minor colour mismatches. A Poisson blending pass dissolves hard mask edges by solving the Poisson equation as a sparse linear system. On mobile, this is implemented as a GLSL fragment shader running via expo-gl's WebGL context, giving GPU performance without a native OpenCV dependency:

Poisson equation: ∇²f = ∇²g inside Ω (mask region) Boundary: f = f\* on ∂Ω where f\* = original image outside mask, g = inpainted region Solver: iterative Jacobi relaxation over 32 GPU passes (sufficient convergence for mobile use case)

## **6.2 Video Temporal Consistency**

For video inputs, independently inpainting each frame causes visible flickering. Temporal consistency is enforced via a lightweight Exponential Moving Average (EMA) blend applied in the Vision Camera Frame Processor plugin, avoiding the full optical flow computation used in v1.0:

- EMA blend: I'\_t = α · I'\_t + (1 − α) · I'\_{t−1}, with α = 0.8.
- The previous inpainted frame is stored in a React Native SharedValue (Reanimated 3) for zero-copy passing to the Frame Processor.
- For clips with significant camera motion, an additional homography-based alignment step (using ML Kit Pose / homography estimation) stabilises the EMA reference frame before blending.

## **6.3 Quality Assessment**

Each output is scored by lightweight on-device metrics computed in JavaScript using raw pixel arrays from the Skia Surface snapshot:

| **Metric**    | **Formula (simplified)**                                       | **On-device target**  |
| ------------- | -------------------------------------------------------------- | --------------------- |
| SSIM          | Structural similarity on masked region vs. surrounding context | \> 0.90               |
| PSNR          | 10 · log₁₀(MAX² / MSE) on synthetic test set                   | \> 28 dB              |
| Mask coverage | sum(mask > 0) / (W × H)                                        | Used for tier routing |

# **7\. Technology Stack**

| **Component**        | **Library / Package**                          | **Rationale**                                        |
| -------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| Framework            | React Native 0.74 + Expo SDK 51                | Cross-platform, OTA updates, managed workflow        |
| Image picker         | react-native-image-picker 7.x                  | Lightweight; camera + library + URL input            |
| Image manipulation   | expo-image-manipulator                         | Resize, crop, convert HEIC; zero native config       |
| Camera (video)       | react-native-vision-camera v4 (min)            | Frame Processor API; video capture only              |
| Video demux          | FFmpegKit (min build, ~12 MB)                  | Frame extraction; stripped to video codecs only      |
| Text detection       | ML Kit Text Recognition v2                     | On-device, 100+ scripts, no quota                    |
| Text detection (aux) | TFLite EAST INT8 via react-native-fast-tflite  | Rotated/low-contrast text fallback                   |
| Text detection (aux) | MediaPipe Tasks Vision                         | Printed/signage text; ~3 MB overhead                 |
| Mask generation      | React Native Skia (@shopify/react-native-skia) | GPU polygon rasterisation; replaces OpenCV for masks |
| Inpainting tier 1    | OpenCV (opencv-react-native, minimal build)    | Classical Telea/NS for small masks                   |
| Inpainting tier 2    | PConv INT8 model via react-native-fast-tflite  | On-device neural; NNAPI / Core ML delegate           |
| Inpainting tier 3    | FastAPI server (LaMa / SD XL) via Fetch API    | Large mask / maximum quality fallback                |
| Post-processing      | expo-gl (WebGL Poisson GLSL shader)            | GPU boundary blending; no OpenCV dependency          |
| Video consistency    | Vision Camera Frame Processor + Reanimated 3   | EMA temporal blend in Frame Processor                |
| State management     | Zustand                                        | Lightweight; replaces Redux for processing state     |
| Navigation           | Expo Router (file-based)                       | Deep link support; share extension routing           |
| Monitoring           | Sentry React Native                            | Crash rate, ANR, performance traces                  |
| API auth             | API key + JWT via expo-secure-store            | Keys stored in device keychain / keystore            |

# **8\. API Design (Server Tier 3)**

The server-side API contract is inherited from TDRS v1.0 with one key addition: mobile clients send cropped regions rather than full images to minimise payload size.

| **Endpoint**           | **Method** | **Description**                                                           |
| ---------------------- | ---------- | ------------------------------------------------------------------------- |
| POST /v1/process/image | REST       | Process a full image or crop; returns cleaned image URL + detections JSON |
| POST /v1/process/crop  | REST (NEW) | Mobile-optimised: accepts base64 crop + mask; returns inpainted crop only |
| POST /v1/process/video | REST       | Async video processing; returns job_id                                    |
| GET /v1/jobs/{job_id}  | REST       | Poll job status: queued \| processing \| done \| failed                   |
| GET /v1/health         | REST       | Liveness check: gpu_available, model_loaded, queue_depth                  |

## **8.1 Mobile Crop Endpoint Request Schema**

POST /v1/process/crop { "crop_base64": "&lt;base64 JPEG, max 1024×1024&gt;", "mask_base64": "&lt;base64 PNG binary mask, same dimensions&gt;", "options": { "model": "lama" | "sd_xl", "dilation_px": 3 } } Response: { "inpainted_crop_base64": "&lt;base64 JPEG&gt;", "processing_time_ms": 2341, "model_used": "lama" }

# **9\. Full Pipeline - End-to-End Pseudocode**

async function processImage(uri, cfg) { // Stage 1: Preprocess const { uri: resized } = await ImageManipulator.manipulateAsync(uri, \[{ resize: { width: Math.min(img.width, 2048) } }\], { format: SaveFormat.JPEG, compress: 0.92 } ); const tiles = tileIfLarge(resized, 512, 64); let allPolygons = \[\]; for (const tile of tiles) { // Stage 2: Detect (ensemble) const mlkitResult = await MLKit.recognizeText(tile.uri); const eastResult = await TFLite.run(eastModel, tile.tensor); const mediaPipeRes = await MediaPipe.detectText(tile.uri); const polys = nmsPolygons(\[ ...mlkitResult.polygons, ...eastResult.polygons, ...mediaPipeRes.polygons \], 0.5); allPolygons.push(...mapToFullImage(polys, tile.offset)); } // Stage 3: Mask const mask = await Skia.renderMask(allPolygons, img.width, img.height, cfg.sigma); if (cfg.maskOnly) return { mask, polygons: allPolygons }; // Stage 4: Inpaint (tiered) const coverage = maskCoverage(mask); let inpainted; if (coverage < 0.05) inpainted = await OpenCV.inpaint(resized, mask, 'telea'); else if (coverage < 0.25) inpainted = await TFLite.run(pconvModel, { image: resized, mask }); else inpainted = await serverInpaint(resized, mask, cfg.apiKey); // Stage 5: Post-process const result = await WebGL.poissonBlend(inpainted, resized, mask); return { result, mask, polygons: allPolygons }; }

# **10\. Performance Benchmarks & Targets**

| **Benchmark**            | **Dataset / Device**                    | **Target** |
| ------------------------ | --------------------------------------- | ---------- |
| Text detection F1        | ICDAR 2015 (ML Kit v2 + EAST ensemble)  | \> 85.0    |
| Text detection F1        | Total-Text curved (EAST INT8)           | \> 78.0    |
| Inpainting SSIM (tier 1) | Synthetic removal, Telea on-device      | \> 0.88    |
| Inpainting SSIM (tier 2) | Synthetic removal, PConv INT8 on-device | \> 0.90    |
| Inpainting SSIM (tier 3) | Synthetic removal, LaMa server          | \> 0.92    |
| Inpainting FID (server)  | Output distribution vs. clean images    | < 15       |
| Latency p50, tier 1      | 1080p, mask < 5%, Pixel 6 / iPhone 13   | < 1.5 sec  |
| Latency p50, tier 2      | 1080p, mask 5-25%, Pixel 6 / iPhone 13  | < 4.0 sec  |
| Latency p95, tier 2      | 1080p, mask 5-25%, Pixel 6 / iPhone 13  | < 6.0 sec  |
| Latency p50, tier 3      | 1080p, mask > 25%, incl. network        | < 7.0 sec  |
| Video processing         | 60-sec 1080p clip, on-device tiers only | < 90 sec   |
| Model bundle size        | All on-device models combined           | < 40 MB    |
| App installed size delta | Incremental size above empty Expo app   | < 55 MB    |

# **11\. Edge Cases & Mitigations**

| **Edge Case**                    | **Challenge**                          | **Mitigation**                                                             |
| -------------------------------- | -------------------------------------- | -------------------------------------------------------------------------- |
| HEIC / HEIF input                | Not universally decodable on Android   | Force-convert to JPEG via expo-image-manipulator before processing         |
| Very small text (< 8 px)         | Below ML Kit minimum resolution        | Super-resolve tile 2× via bicubic before detection; re-scale mask after    |
| Rotated / diagonal text          | ML Kit misses non-horizontal scripts   | EAST ensemble covers arbitrary angles; CRAFT optional future addition      |
| High-density text (> 25%)        | PConv lacks global context             | Escalate to server tier 3 automatically; notify user of server call        |
| Textured background              | Classical Telea leaves visible patches | Auto-escalate to tier 2 if SSIM of tier-1 output < 0.85                    |
| Transparent PNG overlay          | Alpha blended text invisible in RGB    | Flatten onto white before detection; restore alpha after compositing       |
| Offline / no network             | Tier 3 unavailable                     | Surface informative error; offer tier 2 at reduced quality as fallback     |
| Low-light / noisy image          | False-positive text detections         | Apply mild sharpening (Skia ImageFilter) pre-detection; raise θ to 0.6     |
| Video: camera shake              | EMA reference frame misaligned         | Apply homography alignment before EMA blend using MediaPipe features       |
| Memory pressure (low-RAM device) | PConv model OOM on 2 GB devices        | Detect available heap; fall back to tiled Telea or server if < 300 MB free |

# **12\. Model Training & Fine-tuning**

## **12.1 PConv Mobile Model**

The on-device PConv model is a pruned and INT8-quantised derivative of the reference Partial Convolutions implementation. It is trained at 256 × 256 with random binary masks generated from ICDAR polygon annotations, then converted to TFLite using TensorFlow's post-training quantisation pipeline.

| **Parameter**      | **PConv Mobile**                | **EAST Mobile (detection)**       |
| ------------------ | ------------------------------- | --------------------------------- |
| Backbone           | Encoder-decoder, 7 PConv layers | VGG-16 feature extractor (pruned) |
| Input size         | 256 × 256 (tiled crops)         | 512 × 512                         |
| Batch size         | 32                              | 16                                |
| Optimizer          | Adam, lr = 1e-4                 | Adam, lr = 5e-5                   |
| Quantisation       | INT8 post-training (TFLite)     | INT8 post-training (TFLite)       |
| Final model size   | ~12 MB                          | ~8 MB                             |
| Inference delegate | NNAPI (Android) / Core ML (iOS) | NNAPI (Android) / Core ML (iOS)   |

## **12.2 Training Datasets**

- ICDAR 2013 / 2015 / 2019: English scene text with polygon annotations.
- Total-Text: Curved text with word-level polygon annotations.
- MLT-2019: Multilingual scene text (10 scripts) for ML Kit fine-tuning validation.
- TextSeg: Pixel-level text segmentation ground truth for PConv mask training.
- COCO 2017 (backgrounds only): Used as non-text inpainting background diversity set.

# **13\. Deployment Architecture**

## **13.1 Mobile App Distribution**

- iOS: App Store (TestFlight for beta). Minimum iOS 15; built with Xcode 16, Expo EAS Build.
- Android: Google Play Store (Internal Testing Track for beta). Minimum Android 11 (API 30); AAB format via EAS Build.
- OTA updates: Expo Updates for JS bundle hotfixes without full store submission.

## **13.2 Server Infrastructure (Tier 3)**

\# docker-compose.yml (simplified, server tier only) services: api: image: tdrs-api:v2 ports: \['8000:8000'\] deploy: { gpu: 1, nvidia } worker: image: tdrs-worker:v2 command: celery -A tasks worker -Q image,crop redis: image: redis:7-alpine minio: image: minio/minio command: server /data

# **14\. Security & Privacy**

- No persistent storage: all input and output images are deleted from app memory after the result is delivered to the user. No media is written to app-group shared storage.
- API keys stored in iOS Keychain / Android Keystore via expo-secure-store - never in AsyncStorage or plain text.
- TLS 1.3 enforced for all server tier requests; certificate pinning via react-native-ssl-pinning.
- Metadata-only logging: only image dimensions, processing tier, processing time, and job ID are logged. No image content or user identifiers are recorded.
- GDPR: no user data leaves the device in tiers 1 and 2. Tier 3 server is GDPR-compliant; EU-region endpoint available.
- Share extension sandboxing: input images from share extensions are processed in a transient NSTemporaryDirectory / getCacheDir() context and explicitly deleted on completion.

# **15\. Testing Strategy**

## **15.1 Unit & Component Testing (Jest)**

- jest + React Native Testing Library for all React Native components, hooks, and utility functions.
- jest-image-snapshot for visual regression testing of inpainting output: establish baseline renders per tier; fail CI on SSIM regression > 2%.
- Mock TFLite and ML Kit native modules in Jest with recorded polygon fixtures for deterministic unit tests.
- Coverage target: > 80% line coverage on pipeline utility functions (NMS, tiling, mask coverage, EMA blend).

## **15.2 End-to-End Testing (Detox)**

- Detox for full app E2E on real iOS simulator and Android emulator.
- Test matrix: image import → detect → inpaint (each tier) → share output.
- Run on Tier 1 (OpenCV) only in CI to avoid model download latency; Tier 2 and 3 tested in nightly builds.
- Detox device configurations: iPhone 15 Pro simulator (iOS 17), Pixel 7 API 33 emulator.

## **15.3 Server-Side Testing (pytest)**

- pytest for FastAPI endpoint unit tests and integration tests.
- pytest-benchmark for latency regression: fail if p95 latency increases > 15% vs. baseline on /v1/process/crop.
- Hypothesis for property-based testing of NMS and polygon IoU functions.
- LaMa inpainting quality regression: automated SSIM and FID computation on fixed 100-image test set in nightly CI.

## **15.4 Device Testing Matrix**

| **Device**              | **OS**     | **RAM** | **Tier coverage**                |
| ----------------------- | ---------- | ------- | -------------------------------- |
| iPhone 13               | iOS 15.8   | 4 GB    | Tiers 1 + 2 + 3                  |
| iPhone 15 Pro           | iOS 17.4   | 8 GB    | Tiers 1 + 2 + 3                  |
| Samsung Galaxy A54      | Android 13 | 8 GB    | Tiers 1 + 2 + 3                  |
| Pixel 6                 | Android 13 | 8 GB    | Tiers 1 + 2 + 3                  |
| Redmi Note 12 (low-end) | Android 12 | 4 GB    | Tier 1 + 3 (tier 2 memory guard) |
| iPad Pro M2             | iPadOS 16  | 16 GB   | Tiers 1 + 2 + 3 (4K images)      |

# **16\. Implementation Roadmap**

| **Sprint** | **Duration** | **Deliverable**                                                                                                            |
| ---------- | ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| 1          | Week 1-2     | Expo project scaffold; react-native-image-picker + expo-image-manipulator; ML Kit integration; basic detect-and-display UI |
| 2          | Week 3-4     | TFLite EAST + MediaPipe integration; NMS ensemble; Skia mask rasterisation; confidence threshold slider                    |
| 3          | Week 5-6     | OpenCV Telea/NS inpainting (tier 1); Skia Gaussian dilation; end-to-end image pipeline; Jest unit tests                    |
| 4          | Week 7-8     | PConv INT8 TFLite model (tier 2); tiling strategy; NNAPI/CoreML delegate; quality SSIM gate for tier escalation            |
| 5          | Week 9-10    | FastAPI server tier 3 integration; crop-only payload; retry logic; offline fallback UI                                     |
| 6          | Week 11-12   | expo-gl Poisson blending shader; Vision Camera Frame Processor; video pipeline with FFmpegKit min build                    |
| 7          | Week 13-14   | EMA temporal consistency; video output re-mux; Detox E2E tests; TestFlight / Play Internal beta                            |
| 8          | Week 15-16   | Sentry integration; performance profiling; security audit; App Store / Play Store submission                               |

# **Appendix - Key Libraries & References**

| **Library / Paper**                     | **Key Contribution**                                  | **Use in TDRS Mobile**                  |
| --------------------------------------- | ----------------------------------------------------- | --------------------------------------- |
| ML Kit Text Recognition v2              | Google on-device 100+ script OCR detector             | Primary text detector (Stage 2)         |
| EAST (Zhou et al., 2017)                | Efficient scene text detector (arbitrary orientation) | Auxiliary TFLite INT8 detector          |
| NanoDet-Plus                            | Super-lightweight anchor-free object/text detector    | Optional dense-text auxiliary detector  |
| MediaPipe Tasks Vision                  | Google cross-platform vision task framework           | Printed/signage text detection          |
| React Native Skia                       | GPU Skia canvas for React Native                      | Mask rasterisation + dilation (Stage 3) |
| Partial Convolutions (Liu et al., 2018) | Masked convolution renormalisation for inpainting     | On-device inpainting tier 2             |
| LaMa (Suvorov et al., 2021)             | FFT-based large mask inpainting                       | Server inpainting tier 3                |
| SSIM (Wang, 2004)                       | Structural similarity perceptual metric               | On-device quality assessment            |
| expo-gl / WebGL                         | WebGL context for React Native via Expo               | Poisson blending GLSL shader (Stage 5)  |
| FFmpegKit (min build)                   | Stripped FFmpeg for video frame demux                 | Video frame extraction (Stage 1)        |
| react-native-fast-tflite                | High-performance TFLite bindings for RN               | EAST + PConv model inference            |
| Detox                                   | Grey-box E2E testing for React Native                 | Full pipeline integration testing       |

_End of Document | TDRS Mobile PRD + TRD v2.0 | April 2026_