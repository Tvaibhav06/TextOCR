TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

## **TEXT DETECTION, REMOVAL & TRANSLATION RE-RENDER SYSTEM** 

_Multilingual Scene Text Erasure, Translation & In-Place Rendering_ 

Product Requirements Document (PRD)  |  Technical Requirements Document (TRD) Version 2.0  |  April 2026  |  UPDATED — Translation Re-Render Feature Added 

**v2.0 Change Summary:** This revision adds Stage 5 (OCR), Stage 6 (Translation), and Stage 7 (Font Matching + Re-Render) to the original 4-stage text-removal pipeline. All prior PRD/TRD content is preserved; new and modified sections are marked ★. 

Page 1 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

## **PART A — PRODUCT REQUIREMENTS DOCUMENT (PRD)** 

## **1. Executive Summary** 

This document defines requirements for the Text Detection, Removal & Translation Re-Render System (TDRTS). The system detects and erases all visible text from images or video frames using deeplearning inpainting, then — optionally — reads the original text via OCR, translates it into a userselected target language, and renders the translated text back in the same position, at the same size, in the same colour, and in a visually matched font. The result is a fully translated image that looks as if it was originally authored in the target language. 

## **2. Problem Statement** 

Global content creators, publishers, legal teams, and accessibility tools face a two-part problem: (a) removing source-language text overlaid on visual media, and (b) replacing it with accurate targetlanguage text that fits naturally into the original visual context. Manual workflows require a skilled designer for each image. An automated, language-agnostic pipeline that detects, erases, translates, and re-renders text handles both steps without human intervention. 

## **3. Goals & Non-Goals** 

## **3.1 Goals** 

- Detect and erase text from images and video in any language or script 

- Extract text content from each detected region via on-device OCR 

- Translate extracted text into any of 50+ supported target languages 

- Re-render translated text at the exact original bounding region 

- Match original font size, weight, colour, and style class (serif / sans / mono) 

- Support right-to-left (RTL) target languages: Arabic, Hebrew, Urdu 

- Handle text length expansion (German +30%, Hindi +20%) via auto font scaling 

- Preserve image quality: no visible inpainting seams, no colour mismatch 

## **3.2 Non-Goals (unchanged)** 

- Real-time video processing at > 60 FPS on CPU-only hardware 

- Generative font creation (we match from a bundled font library) 

- Legal or certified translation quality (MT output; human review recommended for legal use) 

## **4. User Stories** 

|**ID**|**User**|**Requirement**|
|---|---|---|
|US-01|Video editor|Remove burnt-in subtitles from archival footage so I can re-subtitle<br>in another language|
|US-02|Legal team|Redact and translate licence plates and personal info from video<br>evidence automatically|



Page 2 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

|US-03|E-commerce|Remove competitor watermarks and localise product labels into<br>French and German|
|---|---|---|
|US-04|Publisher|Strip and retranslate promo text overlays on stock images for 12<br>regional editions|
|US-05|Developer|POST an image with target_lang=hi and receive a fully Hindi-<br>rendered cleaned image|
|**US-06**<br>**★**|ML researcher|Batch-process 10,000 images overnight, replacing English<br>captions with Mandarin|
|**US-07**<br>**★**|Accessibility|Translate on-screen text in screenshots for users who are non-<br>native speakers|
|**US-08**<br>**★**|Game localiser|Replace in-game UI text captured in screenshots with localised<br>strings in 15 languages|



## **5. Functional Requirements** 

## **5.1 – 5.6 (unchanged from v1.0)** 

All original functional requirements for input handling, detection, mask generation, inpainting, output, and configuration remain fully in effect. Refer to v1.0 for those sections. 

## **★ 5.7 OCR — Text Extraction (NEW)** 

- **NEW** FR-7.1: For each inpainted text region, extract the original text string using on-device OCR. 

- **NEW** FR-7.2: Report detected script/language family per region (Latin, Devanagari, CJK, Arabic, etc.). 

- **NEW** FR-7.3: Extract per-character bounding boxes when available (for precise re-render alignment). 

- **NEW** FR-7.4: Provide confidence score for OCR result; skip re-render if confidence < 0.6. 

- **NEW** FR-7.5: Support OCR for all scripts supported by ML Kit Text Recognition v2 (50+ scripts). 

## **★ 5.8 Translation (NEW)** 

**▶ NEW** FR-8.1: Translate extracted text into any of the 55 supported target languages (see §5.8.1). 

**▶ NEW** FR-8.2: Support on-device translation (ML Kit Translation API, 58 models, ~30 MB each) for offline use. 

**▶ NEW** FR-8.3: Support cloud translation (Google Cloud Translation / DeepL / LibreTranslate) as quality upgrade. 

- **NEW** FR-8.4: Preserve markdown-style formatting tokens (line breaks, ALL CAPS, title case) in translation. 

- **NEW** FR-8.5: Allow per-region language override via API (e.g., force OCR language = 'ja' for a region). 

- **NEW** FR-8.6: Return original + translated text pair in API response alongside the rendered image. 

## **★ 5.8.1 Supported Target Languages (NEW)** 

|**Group**|**Languages**|
|---|---|
|European|French, German, Spanish, Italian, Portuguese, Dutch, Polish, Swedish, Danish,<br>Norwegian, Finnish, Czech, Romanian, Hungarian, Greek|



Page 3 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

|South Asian|Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi,<br>Urdu, Sinhala, Nepali|
|---|---|
|East Asian|Mandarin (Simplified), Mandarin (Traditional), Japanese, Korean, Vietnamese,<br>Thai, Indonesian, Malay|
|Middle East / RTL|Arabic, Hebrew, Persian (Farsi), Pashto|
|African / Other|Swahili, Afrikaans, Amharic, Turkish, Russian, Ukrainian, Bulgarian|



## **★ 5.9 Font Matching & Re-Render (NEW)** 

**▶ NEW** FR-9.1: Detect original text font class: serif, sans-serif, monospace, display/decorative, handwritten. **▶ NEW** FR-9.2: Detect original font weight: thin/light/regular/medium/bold/black (mapped from stroke width). **▶ NEW** FR-9.3: Sample the dominant text colour from the original pixel region (median of text pixels). **▶ NEW** FR-9.4: Sample the background colour from the inpainted region (used for text shadow/outline if needed). 

**▶ NEW** FR-9.5: Select the best-matching font from a bundled font atlas covering all supported scripts. 

**▶ NEW** FR-9.6: Compute rendered font size as: font_size = bbox_height × scale_factor, then scale down if text overflows width. 

**▶ NEW** FR-9.7: For RTL languages, mirror text alignment and rendering direction within the bounding box. **▶ NEW** FR-9.8: Render translated text over the inpainted background at the original bounding polygon location. 

**▶ NEW** FR-9.9: Apply a thin outline (1–2 px) or drop shadow using background colour when contrast ratio < 3:1. 

**▶ NEW** FR-9.10: Offer three text-fit strategies: (a) scale-to-fit, (b) wrap-to-box, (c) truncate-with-ellipsis. 

## **6. Non-Functional Requirements (updated)** 

|**Category**|**Requirement**|**Target**|
|---|---|---|
|**Performance**<br>**★**|1080p image end-to-end<br>(detect+erase+OCR+translate+render)|< 6 sec on A100 GPU|
|**Performance**<br>**★**|Mobile 720p (on-device translate, OpenCV<br>inpaint)|< 5 sec on ARM CPU|
|**Performance**<br>**★**|Translation API latency (cloud)|< 800 ms per region|
|Accuracy|Text detection F1 (ICDAR 2015)|> 87%|
|Accuracy|OCR word error rate (Latin scripts)|< 5%|
|Accuracy|OCR word error rate (CJK/Indic scripts)|< 12%|
|**Accuracy**<br>**★**|Translation BLEU score (EN→FR, EN→DE<br>benchmark)|> 40|
|Accuracy|Inpainting SSIM on background region|> 0.92|
|**Accuracy**<br>**★**|Colour match: ΔE between original and re-<br>rendered text colour|< 3.0|
|Scalability|Concurrent API requests|> 50 with job queue|
|Reliability|API uptime|99.5% SLA|



Page 4 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

|**Security**<br>**★**|No input media stored after processing|Ephemeral processing|
|---|---|---|
|**Offline**<br>**★**|Full pipeline operable without internet (on-<br>device models)|Yes (degraded quality)|



## **7. Success Metrics (updated)** 

- F1 > 0.87 on ICDAR 2015 (detection — unchanged) 

- SSIM > 0.92 on inpainted regions (unchanged) 

- OCR word accuracy > 95% on Latin and > 88% on CJK/Indic test sets 

- Translation BLEU > 40 on standard EN→FR and EN→DE benchmarks 

- Colour fidelity ΔE₀₀ < 3.0 between original and re-rendered text colour on 100-image test set 

- Font class match rate > 90% (serif/sans/mono correctly identified) 

- Text fits bounding box without overflow in > 95% of re-renders 

- User satisfaction rating ≥ 4.0 / 5.0 on rendered output quality in user study (n=50) 

Page 5 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

Page 6 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

## **PART B — TECHNICAL REQUIREMENTS DOCUMENT (TRD)** 

## **1. Updated System Architecture** 

The pipeline now has 7 sequential stages. Stages 1–4 are unchanged from v1.0. Stages 5–7 are new. 

**Full Pipeline:** INPUT → [1: Preprocess] → [2: Detect] → [3: Mask] → [4: Inpaint] → [5: OCR] → [6: Translate] → [7: Font Match + Re-Render] → OUTPUT 

**Erase-only mode:** Stages 1–4 only (original behaviour, selectable via API flag translate=false) 

## **2. Stages 1–4 — Unchanged** 

All specifications for preprocessing (tiling, normalisation), text detection (DBNet++ + PaddleOCR ensemble), mask generation (morphological dilation), inpainting (LaMa / Stable Diffusion / OpenCV Telea), and post-processing (Poisson blending) remain as documented in v1.0. These stages now run identically whether or not translation is enabled. 

## **★ 3. Stage 5 — OCR (Text Extraction)** 

After inpainting erases the text from the image, the original image (pre-inpainting) is used to extract text content from each detected bounding region. Using the pre-inpainting image preserves the original text pixels for OCR; the inpainted image is used for background sampling. 

## **3.1 OCR Engine Selection** 

|**Engine**|**Environment**|**Strengths**|**Limitations**|
|---|---|---|---|
|Google ML Kit Text<br>Rec. v2|On-device<br>(Android/iOS)|50+ scripts, free, fast<br>(~80ms)|No confidence per<br>word on all scripts|
|Apple Vision<br>VNRecognizeText|iOS only|Best Latin/CJK accuracy on<br>Apple SoC|iOS-only|
|PaddleOCR (Server)|Server / Python|SOTA accuracy, 80+<br>languages, polygon-level|Requires Python<br>runtime|
|Tesseract 5 (LSTM)|Server / fallback|Open-source, 100+<br>languages|Slower, lower<br>accuracy on complex<br>backgrounds|
|Google Cloud Vision<br>OCR|Cloud API|Highest accuracy, 150+<br>languages|Requires internet,<br>cost per call|



Recommended: ML Kit on mobile (Stages 1–7 on-device), PaddleOCR on server backend. 

Page 7 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

## **3.2 OCR Per-Region Cropping** 

Each detected bounding polygon is cropped from the original image with a 2px padding, then deskewed using the polygon rotation angle before OCR: 

Given polygon P = {(x₁,y₁), (x₂,y₂), (x₃,y₃), (x₄,y₄)} Rotation angle:  θ = atan2(y₂-y₁, x₂-x₁) De-skew transform:  M = [cos θ  -sin θ | sin θ   cos θ] Cropped image:  crop = warpAffine(original, M, (bbox_w+4, bbox_h+4)) 

## **3.3 OCR Confidence Gating** 

OCR results below a confidence threshold are flagged and skipped for translation — the text region is kept erased but no translated text is rendered. This avoids rendering garbled translations from misread text. 

Include region R  for translation  iff  conf_ocr(R ) ≥ θ_ocrᵢ ᵢ Default θ_ocr = 0.60 Flagged regions: returned in API response as { status: 'erased_only', reason: 'low_ocr_confidence' } 

## **★ 4. Stage 6 — Translation** 

## **4.1 Translation Architecture** 

Translation follows a quality-tiered model identical in structure to the inpainting tiers: 

|**Tier**|**Engine**|**Latency**|**Quality**|**Requir**<br>**es Net**|
|---|---|---|---|---|
|1 — On-<br>device|ML Kit Translation (58 language<br>pairs, ~30 MB/model)|< 100 ms|Good for common<br>pairs|No|
|2 — Edge<br>server|LibreTranslate (self-hosted, Argos<br>Translate backend)|< 400 ms|Better, BLEU +8<br>vs. Tier 1|LAN<br>only|
|3 — Cloud<br>API|Google Cloud Translation v3 / DeepL<br>API|< 800 ms|SOTA, 100+<br>languages|Yes|



## **4.2 Translation Pre/Post-Processing** 

Raw OCR output often contains artefacts. A normalisation pass runs before translation, and a formatting pass runs after to preserve display properties: 

```
def prepare_for_translation(ocr_text: str) -> str:
    text = ocr_text.strip()
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'[|l1](?=[A-Z])', 'I', text) # OCR I/l confusion fix
    return text
```

```
def apply_formatting(translated: str, original: str) -> str:
    if original.isupper():  return translated.upper()
    if original.istitle():  return translated.title()
    # Preserve leading/trailing newlines for multi-line boxes
```

Page 8 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

```
    leading  = len(original) - len(original.lstrip('\n'))
    trailing = len(original) - len(original.rstrip('\n'))
    return '\n'*leading + translated + '\n'*trailing
```

## **4.3 Text Expansion Handling** 

Translating from compact scripts (e.g., English, Chinese) to verbose scripts (German, Hindi) can increase text length by 20–40%. The pipeline handles this at the render stage via adaptive font scaling: 

max_chars_per_line = floor(bbox_width / (font_size × avg_char_width_ratio)) where avg_char_width_ratio ≈ 0.55 for Latin,  0.60 for Devanagari,  0.95 for CJK 

If rendered_width(text, font_size) > bbox_width: font_size_new = font_size × (bbox_width / rendered_width(text, font_size)) font_size_new = max(font_size_new, min_font_size)   (floor: 8px) 

Multi-line wrap trigger:  font_size_new < 0.65 × font_size_original → split translated text into word-wrapped lines within bbox_height 

## **★ 5. Stage 7 — Font Matching & Text Re-Render** 

This stage composites translated text back onto the inpainted image at the exact original bounding region, matching the visual appearance of the original text as closely as possible. 

## **5.1 Font Attribute Extraction** 

Font attributes are inferred from the original text region pixels using a combination of classical image processing and a lightweight CNN classifier: 

|**Attribute**|**Method**|**Output**|
|---|---|---|
|Font size (px)|bbox_height × 0.72 (cap-height to em<br>ratio)|Integer px|
|Font weight|Mean stroke width via Distance<br>Transform; map to<br>thin/regular/bold/black|Weight class 100–900|
|Font class|ResNet-18 classifier (serif / sans /<br>mono / display / handwritten) — 5-<br>class|Class + confidence|
|Text colour|K-means (k=2) on text-region pixels;<br>dominant non-background cluster<br>centroid|RGB tuple|
|Background colour|Median of inpainted region pixels<br>(5px border strip around bbox)|RGB tuple|
|Contrast ratio|WCAG: (L1+0.05)/(L2+0.05) where L<br>= relative luminance|Float ≥ 1.0|
|Italic / oblique|Measure average character slant|Boolean + angle °|



Page 9 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

angle via horizontal projection profile 

## **5.2 Font Weight from Stroke Width** 

Distance Transform: DT(p) = min distance from foreground pixel p to background 

Mean stroke width:  SW = 2 × mean(DT(p))  ∀p ∈ text region 

Weight mapping: SW < 1.5 px  → Thin (100)     SW 1.5–2.5 → Light (300) SW 2.5–3.5   → Regular (400)  SW 3.5–4.5 → Medium (500) SW 4.5–6.0   → Bold (700)     SW > 6.0   → Black (900) 

## **5.3 Colour Extraction — K-means on Text Region** 

Input: pixel set X = { p ∈ bbox | edge_dilated_mask(p) = 1 } K-means (k=2) minimises:  Σ  Σᵢ ₚ∈C  ‖p - μ ‖ᵢ ᵢ² Dominant non-background cluster = argmax cluster_size excluding background cluster (Background cluster identified as cluster whose centroid is closest to inpainted region median) 

Colour distance check:  ΔE₀₀(original_colour, rendered_colour) < 3.0 ΔE₀₀ = CIEDE2000 formula (IEC 61966-2-1), computed in L*a*b* space 

## **5.4 Font Library & Script Coverage** 

A bundled font atlas covers all 55 supported target languages. Fonts are grouped by script family and selected by font-class match: 

|**Script Family**|**Serif Font**|**Sans-Serif Font**|**Mono Font**|
|---|---|---|---|
|Latin (EU languages)|Noto Serif|Noto Sans|Noto Sans Mono|
|Devanagari (Hindi, Marathi,<br>Nepali)|Noto Serif<br>Devanagari|Noto Sans<br>Devanagari|—|
|Bengali|Noto Serif Bengali|Noto Sans Bengali|—|
|Tamil / Telugu / Kannada /<br>Malayalam|Noto Serif Tamil etc.|Noto Sans Tamil etc.|—|
|Arabic / Urdu / Persian<br>(RTL)|Noto Naskh Arabic|Noto Kufi Arabic|—|
|Hebrew (RTL)|Frank Ruhl Libre|Noto Sans Hebrew|—|
|CJK Simplified (Mandarin)|Noto Serif CJK SC|Noto Sans CJK SC|—|
|CJK Traditional / Japanese /<br>Korean|Noto Serif CJK<br>TC/JP/KR|Noto Sans CJK<br>TC/JP/KR|—|



Page 10 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

|Thai / Vietnamese /<br>Indonesian|Noto Serif Thai etc.|Noto Sans Thai etc.|—|
|---|---|---|---|
|Cyrillic (Russian, Ukrainian,<br>Bulgarian)|Noto Serif|Noto Sans|Noto Sans Mono|



All Noto fonts are licensed under the SIL Open Font License (OFL). Total atlas size: ~85 MB (subsetted), loaded on-demand per script. 

## **5.5 RTL Rendering** 

Right-to-left scripts (Arabic, Hebrew, Urdu, Persian, Pashto) require mirrored layout: 

- Text alignment: right-aligned within bounding box (reversed from LTR) 

- Bidirectional algorithm: Unicode BiDi (UAX #9) applied before rendering 

- Ligature shaping: HarfBuzz used for Arabic ligature/contextual shaping — required for correct Arabic glyph forms 

- Kerning and glyph advances follow the RTL direction 

- Mixed LTR/RTL (e.g., 'Call +44 123' in Arabic context): BiDi algorithm handles embedding levels 

## **5.6 Text Rendering Pipeline** 

The full text rendering sequence for a single bounding region: 

1. Extract font attributes (§5.1) 

2. Select matching font from atlas (§5.4) 

3. Compute font size: size = bbox_height × 0.72, then apply expansion scaling (§4.3) 

4. Compute layout: wrap text into lines if needed; align LTR or RTL 

5. Measure rendered text width via FreeType glyph advance sum 

6. Apply contrast check: if ΔE₀₀ < 3.0 or contrast ratio < 3:1, add 1px outline in background colour 

7. Render text to a transparent RGBA buffer using FreeType / Skia text API 

8. Composite text buffer over inpainted image at polygon position using alpha blending: 

Composite:  C_out = α · C_text + (1 - α) · C_background where α = text pixel opacity from FreeType anti-aliasing (0.0–1.0) C_text = sampled text colour (RGB) 

C_background = inpainted image pixel at that location 

9. Apply final sharpening pass on the rendered text region to match original crispness 

## **5.7 Font Size Computation — Detailed** 

Step 1 — Base size from bbox: 

font_size = bbox_height × cap_height_ratio 

cap_height_ratio ≈ 0.72 for most Latin fonts (cap height / em height) Adjusted per script: Devanagari = 0.68, CJK = 0.85, Arabic = 0.70 

Step 2 — Overflow check: 

rendered_width = Σ advance_width(glyph_i, font_size) + kerning if rendered_width > bbox_width: 

Page 11 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

scale = bbox_width / rendered_width font_size = font_size × scale Step 3 — Floor guard: font_size = max(font_size, 8)   (minimum legible size) 

Step 4 — Multi-line trigger: if font_size < 0.65 × original_font_size: → activate word-wrap within bbox; reset font_size to original 

## **★ 6. Updated Technology Stack** 

|**Component**|**Technology**|**Stag**<br>**e**|**Rationale**|
|---|---|---|---|
|Runtime|Python 3.11 + CUDA 12|All|Ecosystem compatibility|
|Detection|DBNet++ (mmocr)|2|SOTA scene text detection|
|Multilingual det.|PaddleOCR det|2|80+ language scripts|
|Inpainting|LaMa (simple-lama-<br>inpainting)|4|Best-in-class large masks|
|Diffusion fallback|diffusers (SD XL Inpaint)|4|Maximum quality option|
|Video inpainting|ProPainter + FFmpeg|4|Temporal consistency|
|**OCR — server**<br>**★**|PaddleOCR v3 (PP-OCRv4<br>rec)|5|SOTA multilingual OCR|
|**OCR — mobile**<br>**★**|ML Kit Text Rec. v2 / Apple<br>Vision|5|On-device, zero download|
|**Translation — on-**<br>**★**<br>**device**|ML Kit Translation API (58<br>models)|6|Offline capability|
|**Translation —**<br>**★**<br>**cloud**|Google Cloud Translation v3<br>/ DeepL|6|SOTA quality, 100+ langs|
|**BiDi / shaping**<br>**★**|HarfBuzz + ICU (via python-<br>bidi / uharfbuzz)|7|Correct Arabic/Hebrew shaping|
|**Font rendering**<br>**★**|FreeType (server) / Skia<br>(mobile)|7|Sub-pixel glyph rendering|
|**Font library**<br>**★**|Noto font family (SIL OFL)|7|Universal script coverage|
|**Font**<br>**★**<br>**classification**|ResNet-18 fine-tuned<br>(serif/sans/mono/display)|7|Font class detection|
|**Colour space**<br>**★**|OpenCV LAB + CIEDE2000<br>(python-colormath)|7|Perceptual colour matching|
|API Server|FastAPI + Uvicorn|API|Async, typed, fast|
|Job Queue|Celery + Redis|API|Async batch processing|



Page 12 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

## **★ 7. Updated API Design** 

## **7.1 Updated /v1/process/image Endpoint** 

**==> picture [469 x 377] intentionally omitted <==**

**----- Start of picture text -----**<br>
POST /v1/process/image<br>Body: {<br>  image_url | image_base64,<br>  options: {<br>    threshold:          float,     // detection confidence, default 0.5<br>    dilation_px:        int,       // mask dilation, default 3<br>    mask_only:          bool,      // skip inpaint, default false<br>    translate:          bool,      // enable Stage 5-7, default false  ← NEW<br>    target_language:    string,    // BCP-47 e.g. 'fr', 'hi', 'zh-CN'  ← NEW<br>    translation_tier:   int,       // 1=on-device 2=edge 3=cloud, default 1  ←<br>NEW<br>    font_size_strategy: string,    // 'scale' | 'wrap' | 'truncate', default<br>'scale'  ← NEW<br>    region_overrides: [            // per-region language override  ← NEW<br>      { region_id: int, ocr_lang: string, target_lang: string }<br>    ]<br>  }<br>}<br>Returns: {<br>  cleaned_image_url:    string,   // text erased<br>  translated_image_url: string,   // text erased + translated  ← NEW<br>  mask_url:             string,<br>  detections: [<br>    {<br>      id, polygon, confidence,<br>      ocr_text:        string,    // original extracted text  ← NEW<br>      translated_text: string,    // translated string  ← NEW<br>      source_language: string,    // detected language BCP-47  ← NEW<br>      font_attrs: {              // ← NEW<br>        size_px, weight, class, colour_hex, is_italic, is_rtl<br>      }<br>    }<br>  ]<br>}<br>**----- End of picture text -----**<br>


## **★ 8. Full Pipeline — Updated End-to-End Pseudocode** 

```
def process_image(image_path, cfg):
    # ── Stages 1–4: unchanged ─────────────────────────────────────
    img         = load_and_normalize(image_path)
    detections  = ensemble_detect(img, cfg.threshold)     # Stage 2
    mask        = build_mask(detections, cfg.dilation_px) # Stage 3
    inpainted   = lama_inpaint(img, mask)                 # Stage 4
    result      = poisson_blend(inpainted, img, mask)     # Stage 4 post
    if not cfg.translate:
        return {'result': result, 'mask': mask, 'detections': detections}
    # ── Stage 5: OCR ──────────────────────────────────────────────
    for det in detections:
        crop          = crop_deskew(img, det.polygon)     # original image!
        det.ocr_text  = paddle_ocr(crop)
        det.ocr_conf  = det.ocr_result.confidence
    # ── Stage 6: Translation ──────────────────────────────────────
```

Page 13 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

```
    translator = get_translator(cfg.translation_tier, cfg.target_language)
    for det in detections:
```

```
        if det.ocr_conf < 0.60: continue   # skip low-confidence OCR
        normed            = normalize_text(det.ocr_text)
        det.translated    = translator.translate(normed, cfg.target_language)
        det.translated    = apply_formatting(det.translated, det.ocr_text)
    # ── Stage 7: Font Match + Re-Render ───────────────────────────
    canvas = result.copy()  # start from inpainted background
    for det in detections:
```

```
        if not det.translated: continue
        attrs   = extract_font_attrs(img, det)  # size, weight, class, colour
        font    = select_font(attrs, cfg.target_language)  # from Noto atlas
        size    = compute_font_size(attrs, det.bbox, det.translated, font)
        colour  = attrs.text_colour
        outline = compute_outline(colour, attrs.bg_colour)  # if contrast <
3:1
        canvas  = render_text(canvas, det.translated, font, size,
                              colour, det.polygon, outline, attrs.is_rtl)
    return {'result': result, 'translated': canvas,
            'mask': mask, 'detections': detections}
```

## **9. Performance Benchmarks — Updated Targets** 

|**Benchmark**|**Dataset / Condition**|**Target**|
|---|---|---|
|Text detection F1|ICDAR 2015|> 87.0|
|Text detection F1|Total-Text (curved)|> 82.0|
|**OCR word accuracy**<br>**★**|Latin scripts (en, fr, de, es)|> 95%|
|**OCR word accuracy**<br>**★**|CJK / Indic scripts|> 88%|
|**Translation BLEU**<br>**★**|WMT EN→FR benchmark|> 40|
|**Translation BLEU**<br>**★**|WMT EN→DE benchmark|> 38|
|**Colour fidelity**<br>**★**|ΔE<br>on 100-image test set<br>₀₀|< 3.0|
|**Font class accuracy**<br>**★**|serif/sans/mono/display 4-class|> 90%|
|**Text fit rate**<br>**★**|No bbox overflow on 1000<br>regions|> 95%|
|Inpainting SSIM|Synthetic text removal set|> 0.92|
|Inpainting PSNR|Synthetic text removal set|> 30 dB|
|**E2E latency p50**<br>**★**|1080p translate, A100 GPU|< 6.0 sec|
|**E2E latency p95**<br>**★**|1080p translate, A100 GPU|< 10 sec|
|**Mobile E2E (720p)**<br>**★**|On-device, ARM CPU, Tier 1<br>translate|< 5 sec|



## **★ 10. Edge Cases — Translation & Re-Render** 

**Edge Case Challenge Mitigation** 

Page 14 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

|**Text too short for**<br>**★**<br>**OCR**|< 3 characters → high OCR error|Skip translation; keep erased|
|---|---|---|
|**Mixed script in one**<br>**★**<br>**region**|e.g., '50% OFF' — numbers +<br>Latin|Translate only the text tokens;<br>preserve numerals|
|**Text length**<br>**★**<br>**expansion > 50%**|Translated text > 1.5× original<br>width|Multi-line wrap or scale-to-fit (user<br>configurable)|
|**RTL in LTR image**<br>**★**<br>**context**|Arabic text in English image|Full BiDi algorithm + HarfBuzz<br>shaping|
|Low OCR confidence (<<br>0.60)|Garbled text pollutes translation|Skip re-render; region stays erased;<br>flag in API|
|**Handwritten /**<br>**★**<br>**decorative font**|No exact match in Noto atlas|Fall back to regular weight Noto<br>Sans for that script|
|Very small text (< 12px<br>height)|OCR misreads; font unreadable<br>after render|Skip translation for regions < 12px<br>height|
|Colour on gradient<br>background|Text colour varies across region|Sample colour at text centroid; use<br>median of top-5 pixels|
|**Logo / stylised text**<br>**★**|Not natural language; translation<br>meaningless|Detect via text confidence + word-<br>shape heuristic; skip|
|**CJK → Latin**<br>**★**<br>**expansion**|Japanese kanji expands 300–<br>500% in English|Wrap + scale; offer user toggle to<br>keep Japanese|
|**Overlapping text**<br>**★**<br>**regions**|Two regions whose polygons<br>intersect|Render in z-order: background<br>region first, then foreground|



## **★ 11. Updated Implementation Roadmap** 

Weeks 1–12 remain as in v1.0 (detection, masking, inpainting, API, video, server). New phases: 

10. Week 13–14: OCR integration (PaddleOCR server, ML Kit mobile); per-region crop + deskew; confidence gating 

11. Week 15–16: ML Kit Translation on-device; LibreTranslate edge server; normalisation + formatting preserve 

12. Week 17–18: Font attribute extraction (stroke-width weight, ResNet-18 class classifier); Noto atlas integration 

13. Week 19–20: FreeType / Skia text rendering; font size computation; RTL BiDi + HarfBuzz; composite pipeline 

14. Week 21–22: Contrast check + outline fallback; multi-line wrap; text-fit strategies 

15. Week 23–24: API v2 endpoints (translate flag, target_language, per-region override); updated response schema 

16. Week 25–26: Quality benchmarking (BLEU, ΔE 

   - ₀₀, font class accuracy, fit rate); user study (n=50) 

17. Week 27–28: Edge case hardening (RTL, expansion, handwritten); load testing at 50 concurrent requests 

**Design Principle:** The re-render stage must be visually undetectable to a casual viewer — the translated text should look as if the image was originally authored in the target language. When in doubt, err toward legibility (use a clean Noto Sans) rather than forcing a match to a decorative font that may render poorly for the target script. 

Page 15 of 16 

TDRTS — Text Detection, Removal & Translation Re-Render System  |  PRD + TRD v2.0 

## **Appendix — Key References** 

|**Reference**|**Contribution**|**Stage**|
|---|---|---|
|DBNet++ (2022)|Differentiable binarization text<br>detection|2|
|LaMa (2021)|Fourier convolution inpainting|4|
|PaddleOCR v4 (2023)|PP-OCRv4: SOTA multilingual<br>OCR|5|
|ML Kit Text Rec. v2|On-device 50+ script OCR|5|
|**Google Cloud Trans. v3**<br>**★**|Neural MT, 133 languages|6|
|**DeepL API (2023)**<br>**★**|Highest BLEU for EU language<br>pairs|6|
|**HarfBuzz (2023)**<br>**★**|Text shaping for complex scripts|7|
|**Unicode BiDi UAX #9**<br>**★**|Bidirectional text algorithm|7|
|**Noto Fonts (Google)**<br>**★**|Universal script font coverage|7|
|**CIEDE2000 (Sharma 2005)**<br>**★**|Perceptual colour difference<br>formula|7|
|**Stroke Width Transform**<br>**★**|Font weight estimation from images|7|
|SSIM (Wang 2004)|Structural similarity quality metric|QA|



Page 16 of 16 

