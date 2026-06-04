/**
 * src/services/BackendService.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Phase 2: Backend URL resolution with priority-ordered strategy:
 *
 *  1. Last-known working URL from MMKV cache  (fastest — reuse across sessions)
 *  2. Android emulator alias  http://10.0.2.2:8000              (dev shortcut)
 *  3. mDNS / Zeroconf LAN scan for _textremover._tcp             (real devices)
 *  4. Manual URL configured by the user in Settings               (always wins)
 *
 * After any URL is validated via GET /api/health → 200, it is stored in MMKV
 * so subsequent launches can skip discovery.
 *
 * Usage:
 *   const url = await backendService.resolveUrl();
 *   if (!url) { ... no server found ... }
 */

import { Platform } from 'react-native';
import { MMKV } from 'react-native-mmkv';
import * as ImageManipulator from 'expo-image-manipulator';

// ─── Constants ────────────────────────────────────────────────────────────────

const MMKV_KEY_BACKEND_URL    = 'backend_url_v2';
const MMKV_KEY_MANUAL_URL     = 'backend_url_manual_v2';
const EMULATOR_URL             = 'http://10.0.2.2:8000';
const DEFAULT_PORT             = 8000;
const HEALTH_PATH              = '/api/health';
const HEALTH_TIMEOUT_MS        = 3000;   // short — fail fast, don't freeze UI
const MDNS_SCAN_TIMEOUT_MS     = 5000;   // how long to wait for mDNS results

// Max dimension sent to backend.
// LaMa processing scales with image area — halving the longest side = 4× faster.
// We cap at 1200px (not 800px) to retain 87% of original pixels:
//   1920×1080 → 1200×675  (~2.6× faster, ~30s on CPU, minimal quality loss)
//   2560×1440 → 1200×675  (~4.5× faster)
// Going below 1000px starts to visibly reduce the LaMa output quality.
const UPLOAD_MAX_DIM           = 1200;   // px — best quality/speed tradeoff
const UPLOAD_TIMEOUT_MS        = 180_000; // 3 minutes — headroom for large images on CPU

// ─── MMKV storage (process-level singleton) ───────────────────────────────────

const storage = new MMKV({ id: 'textremover-backend' });

// ─── Types ────────────────────────────────────────────────────────────────────

export interface BackendHealthInfo {
  status: string;
  version: string;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Probe a URL by calling GET /api/health with a short timeout.
 * Returns true if the server responds with HTTP 200.
 */
async function probeHealth(baseUrl: string): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS);
  try {
    const url = baseUrl.replace(/\/$/, '') + HEALTH_PATH;
    const resp = await fetch(url, {
      method: 'GET',
      signal: controller.signal,
    });
    return resp.status === 200;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Attempt mDNS discovery for _textremover._tcp using react-native-zeroconf.
 * Resolves with the first responding URL, or null on timeout / library unavailable.
 *
 * NOTE: react-native-zeroconf must be installed and linked.
 *       Skips gracefully if the library is not available.
 */
async function discoverViaMdns(): Promise<string | null> {
  try {
    // Dynamic import — won't crash if library is absent
    const ZeroconfModule = await import('react-native-zeroconf').catch(() => null);
    if (!ZeroconfModule) {
      console.log('[BackendService] react-native-zeroconf not available, skipping mDNS.');
      return null;
    }

    const Zeroconf = ZeroconfModule.default ?? ZeroconfModule.Zeroconf;
    if (!Zeroconf) return null;

    const zeroconf = new Zeroconf();

    return new Promise<string | null>((resolve) => {
      let settled = false;
      const candidates: string[] = [];

      const timer = setTimeout(async () => {
        zeroconf.stop();
        if (settled) return;
        settled = true;

        // Probe all discovered candidates, resolve with first healthy one
        for (const url of candidates) {
          if (await probeHealth(url)) {
            console.log('[BackendService] mDNS: validated', url);
            resolve(url);
            return;
          }
        }
        console.log('[BackendService] mDNS: no healthy server found in scan window.');
        resolve(null);
      }, MDNS_SCAN_TIMEOUT_MS);

      zeroconf.on('resolved', (service: any) => {
        try {
          const host = service.host || service.addresses?.[0];
          const port = service.port ?? DEFAULT_PORT;
          if (host) {
            const url = `http://${host}:${port}`;
            console.log('[BackendService] mDNS resolved:', url);
            candidates.push(url);
          }
        } catch (e) {
          console.warn('[BackendService] mDNS service parse error:', e);
        }
      });

      zeroconf.on('error', (err: any) => {
        console.warn('[BackendService] mDNS error:', err);
      });

      // Start scanning for _textremover._tcp
      zeroconf.scan('_textremover', '_tcp', 'local.');
    });
  } catch (err) {
    console.warn('[BackendService] mDNS discovery exception:', err);
    return null;
  }
}

// ─── BackendService ───────────────────────────────────────────────────────────

class BackendService {
  private _resolvedUrl: string | null = null;

  /**
   * Read the user-configured manual URL (from Settings screen).
   * Returns null if not set.
   */
  getManualUrl(): string | null {
    return storage.getString(MMKV_KEY_MANUAL_URL) ?? null;
  }

  /**
   * Save a manual URL from the Settings screen.
   * Immediately marks it as the resolved URL too.
   */
  setManualUrl(url: string): void {
    const normalized = url.trim().replace(/\/$/, '');
    storage.set(MMKV_KEY_MANUAL_URL, normalized);
    // Also update the cached working URL
    storage.set(MMKV_KEY_BACKEND_URL, normalized);
    this._resolvedUrl = normalized;
    console.log('[BackendService] Manual URL saved:', normalized);
  }

  clearManualUrl(): void {
    storage.delete(MMKV_KEY_MANUAL_URL);
    console.log('[BackendService] Manual URL cleared.');
  }

  /** The cached last-known working URL (persists across launches). */
  getCachedUrl(): string | null {
    return storage.getString(MMKV_KEY_BACKEND_URL) ?? null;
  }

  private _saveWorkingUrl(url: string): void {
    storage.set(MMKV_KEY_BACKEND_URL, url);
    this._resolvedUrl = url;
    console.log('[BackendService] Cached working URL:', url);
  }

  /**
   * Core resolution logic — returns a validated base URL or null.
   *
   * Priority:
   *   1. In-memory resolved URL (already validated this session)
   *   2. Manual URL override (from Settings)
   *   3. Cached URL (MMKV — validated in a previous session)
   *   4. Emulator shortcut (Android only)
   *   5. mDNS LAN discovery (physical devices)
   */
  async resolveUrl(forceRediscover = false): Promise<string | null> {
    // Fast path: already resolved this session
    if (this._resolvedUrl && !forceRediscover) {
      const ok = await probeHealth(this._resolvedUrl);
      if (ok) return this._resolvedUrl;
      console.log('[BackendService] In-memory URL no longer responding, re-resolving…');
      this._resolvedUrl = null;
    }

    // ── Priority 1: Manual URL ────────────────────────────────────────────────
    const manualUrl = this.getManualUrl();
    if (manualUrl) {
      console.log('[BackendService] Trying manual URL:', manualUrl);
      if (await probeHealth(manualUrl)) {
        this._saveWorkingUrl(manualUrl);
        return manualUrl;
      }
      console.log('[BackendService] Manual URL not responding. Trying other sources…');
    }

    // ── Priority 2: MMKV cached URL ───────────────────────────────────────────
    const cachedUrl = this.getCachedUrl();
    if (cachedUrl && cachedUrl !== manualUrl) {
      console.log('[BackendService] Trying cached URL:', cachedUrl);
      if (await probeHealth(cachedUrl)) {
        this._resolvedUrl = cachedUrl;
        return cachedUrl;
      }
      console.log('[BackendService] Cached URL not responding.');
    }

    // ── Priority 3: Android emulator shortcut ─────────────────────────────────
    if (Platform.OS === 'android') {
      console.log('[BackendService] Trying emulator URL:', EMULATOR_URL);
      if (await probeHealth(EMULATOR_URL)) {
        this._saveWorkingUrl(EMULATOR_URL);
        return EMULATOR_URL;
      }
      console.log('[BackendService] Emulator URL not responding.');
    }

    // ── Priority 4: mDNS LAN discovery ───────────────────────────────────────
    console.log('[BackendService] Starting mDNS discovery…');
    const mdnsUrl = await discoverViaMdns();
    if (mdnsUrl) {
      this._saveWorkingUrl(mdnsUrl);
      return mdnsUrl;
    }

    console.log('[BackendService] No backend server found.');
    return null;
  }

  /**
   * Upload the original image to the backend for high-quality LaMa inpainting.
   *
   * @param imageUri   - file:// URI of the ORIGINAL (pre-pipeline) image
   * @param baseUrl    - resolved backend URL (from resolveUrl())
   * @returns          - file:// URI of the saved result image, or null on failure
   */
  async processImage(
    imageUri:   string,
    baseUrl:    string,
    origWidth:  number = 0,
    origHeight: number = 0,
  ): Promise<{
    resultUri:   string;
    ssim:        number;
    psnr:        number;
    coverage:    number;
    inpaintTier: string;
    totalMs:     number;
  } | null> {
    const endpoint = baseUrl.replace(/\/$/, '') + '/api/process';

    // ── Downscale before upload ────────────────────────────────────────────────
    // LaMa processing time scales with image area (O(N²)).
    // Capping at 800px on the longest side makes it 4-6× faster on CPU.
    let uploadUri  = imageUri;
    let uploadW    = origWidth;
    let uploadH    = origHeight;

    try {
      const longestSide = Math.max(origWidth, origHeight);
      if (longestSide > UPLOAD_MAX_DIM) {
        const scale = UPLOAD_MAX_DIM / longestSide;
        uploadW = Math.round(origWidth  * scale);
        uploadH = Math.round(origHeight * scale);
        const resized = await ImageManipulator.manipulateAsync(
          imageUri,
          [{ resize: { width: uploadW, height: uploadH } }],
          { compress: 0.88, format: ImageManipulator.SaveFormat.JPEG },
        );
        uploadUri = resized.uri;
        console.log(
          `[BackendService] Downscaled ${origWidth}×${origHeight} → ${uploadW}×${uploadH} for upload`,
        );
      }
    } catch (resizeErr) {
      console.warn('[BackendService] Resize failed, uploading original size:', resizeErr);
      uploadUri = imageUri;
    }

    console.log('[BackendService] Uploading to:', endpoint, `(${uploadW}×${uploadH})`);

    const formData = new FormData();
    formData.append('image', {
      uri:  uploadUri,
      name: 'image.jpg',
      type: 'image/jpeg',
    } as any);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), UPLOAD_TIMEOUT_MS);

    try {
      const resp = await fetch(endpoint, {
        method:  'POST',
        body:    formData,
        signal:  controller.signal,
      });

      if (!resp.ok) {
        const errText = await resp.text().catch(() => '(no body)');
        console.error(`[BackendService] Server error ${resp.status}: ${errText}`);
        return null;
      }

      const ssim        = parseFloat(resp.headers.get('X-SSIM')       ?? '0');
      const psnr        = parseFloat(resp.headers.get('X-PSNR')       ?? '0');
      const coverage    = parseFloat(resp.headers.get('X-Coverage')   ?? '0');
      const inpaintTier = resp.headers.get('X-InpaintTier')           ?? 'unknown';
      const totalMs     = parseFloat(resp.headers.get('X-TotalMs')    ?? '0');

      const blob = await resp.blob();
      const resultUri = await this._saveBlobToCache(blob);
      if (!resultUri) return null;

      console.log(
        `[BackendService] Result received: SSIM=${ssim}  tier=${inpaintTier}  ${totalMs}ms`,
      );
      return { resultUri, ssim, psnr, coverage, inpaintTier, totalMs };
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        console.error(`[BackendService] Request timed out (${UPLOAD_TIMEOUT_MS / 1000}s).`);
      } else {
        console.error('[BackendService] Upload failed:', err);
      }
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Save a Blob to the device cache directory and return its file:// URI.
   * Uses expo-file-system.
   */
  private async _saveBlobToCache(blob: Blob): Promise<string | null> {
    try {
      // Dynamic import — expo-file-system may not be installed in all builds
      const FileSystem = await import('expo-file-system').catch(() => null);
      if (!FileSystem) {
        console.warn('[BackendService] expo-file-system not available.');
        return null;
      }

      const filename  = `backend_result_${Date.now()}.jpg`;
      const cacheDir  = FileSystem.cacheDirectory ?? 'file:///data/data/com.textremover/cache/';
      const resultUri = cacheDir + filename;

      // Convert blob to base64, then write with FileSystem
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload  = () => {
          const dataUrl = reader.result as string;
          // Strip "data:image/jpeg;base64," prefix
          resolve(dataUrl.split(',')[1] ?? '');
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
      });

      await FileSystem.writeAsStringAsync(resultUri, base64, {
        encoding: FileSystem.EncodingType.Base64,
      });

      console.log('[BackendService] Result saved to cache:', resultUri);
      return resultUri;
    } catch (err) {
      console.error('[BackendService] Failed to save result to cache:', err);
      return null;
    }
  }
}

// ─── Singleton export ─────────────────────────────────────────────────────────

export const backendService = new BackendService();
