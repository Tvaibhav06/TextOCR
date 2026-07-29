/**
 * App.tsx — Phase 2: Hybrid Fallback UI
 * Google Translate Refactored Style
 */

import React, { useEffect, useRef, useState } from 'react'; // Force Fast Refresh reload
import {
  Alert,
  Animated,
  Dimensions,
  Image,
  NativeModules,
  Pressable,
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as MediaLibrary from 'expo-media-library';
import * as ImageManipulator from 'expo-image-manipulator';
import * as FileSystem from 'expo-file-system';

import {
  pipeline,
  CancellationToken,
  type PipelineResult,
  type PipelineProgress,
  type PipelineStage,
} from './src/pipeline/Pipeline';
import { backendService } from './src/services/BackendService';

// Components
import { Theme } from './src/components/Theme';
import { TopBar } from './src/components/TopBar';
import { FloatingActionBar, ActionButton } from './src/components/FloatingActionBar';
import { ProgressCard } from './src/components/ProgressCard';
import { ResultBottomSheet } from './src/components/ResultBottomSheet';
import { TranslationService, type TranslationSession } from './src/services/TranslationService';
import { TranslatedOverlay } from './src/pipeline/translation/SkiaRenderer';
import { SettingsModal } from './src/components/SettingsModal';
import { FEATURES } from './src/config/features';

const DEBUG_MODULES = () => {
  console.log('Available NativeModules:', Object.keys(NativeModules).filter(n => n.toLowerCase().includes('mlkit') || n.toLowerCase().includes('translate')));
};
DEBUG_MODULES();

const { width: SW, height: SH } = Dimensions.get('window');

// App State
type AppPhase = 'idle' | 'editor' | 'processing' | 'result';

interface AppState {
  phase: AppPhase;
  sourceUri: string | null;
  resultUri: string | null;
  imgW: number;
  imgH: number;
  stage: PipelineStage;
  progress: number;
  stageMsg: string;
  result: PipelineResult | null;
  error: string | null;
  showBefore: boolean;
  showSettings: boolean;
  manualUrl: string;

  // Translation State
  translationSession: TranslationSession | null;
  isTranslating: boolean;
  translationStage: string;  // live status message while translating
  targetLang: string;
  /**
   * User-selected source language filter.
   * 'auto' = translate all detected regions (existing behavior).
   * BCP-47 code = only translate regions matching this language; others are restored.
   */
  sourceLang: string;
  /** Stage 7 mobile render coverage — set by onRenderComplete callback. */
  renderCoverage: number | null;
  /**
   * File URI of the flattened snapshot produced by SkiaRenderer offscreen render.
   * Non-null only after a successful mobile translation render.
   * The Save button uses this instead of resultUri on the mobile path so the
   * saved image actually contains the translated text.
   */
  mobileSnapshotUri: string | null;
  isFullscreenPreview: boolean;
  /**
   * S-2: Monotonically-incrementing nonce, incremented at the START of every
   * runTranslation() call. The onSnapshot callback captures this value at render
   * time and rejects any snapshot whose nonce does not match the current value,
   * preventing a stale SkiaRenderer from a prior run from overwriting the current
   * session's resultUri.
   */
  translationSessionId: number;
}

const initialState: AppState = {
  phase: 'idle', sourceUri: null, resultUri: null,
  imgW: 1, imgH: 1, stage: 'idle',
  progress: 0, stageMsg: '', result: null,
  error: null, showBefore: true,
  showSettings: false,
  manualUrl: backendService.getManualUrl() ?? '',
  translationSession: null,
  isTranslating: false,
  translationStage: '',
  targetLang: 'hi',
  sourceLang: 'auto',
  renderCoverage: null,
  mobileSnapshotUri: null,
  isFullscreenPreview: false,
  translationSessionId: 0,
};

export default function App() {
  const [s, setS] = useState<AppState>(initialState);
  const tokenRef = useRef<CancellationToken | null>(null);
  // BUG-6: useRef so onSnapshot always reads the live session ID, never a stale closure.
  const sessionIdRef = useRef(0);
  useEffect(() => { sessionIdRef.current = s.translationSessionId; }, [s.translationSessionId]);
  const progressAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    pipeline.warmUp().catch(() => { });
    return () => { tokenRef.current?.cancel(); };
  }, []);

  useEffect(() => {
    Animated.timing(progressAnim, {
      toValue: s.progress,
      duration: 400,
      useNativeDriver: false,
    }).start();
  }, [s.progress]);

  // Helpers
  const preprocessImage = async (uri: string, w: number, h: number) => {
    const maxDim = 1920;
    if (Math.max(w, h) <= maxDim) return { uri, w, h };
    const scale = maxDim / Math.max(w, h);
    const nw = Math.round(w * scale);
    const nh = Math.round(h * scale);
    const res = await ImageManipulator.manipulateAsync(
      uri,
      [{ resize: { width: nw, height: nh } }],
      { compress: 0.92, format: ImageManipulator.SaveFormat.JPEG },
    );
    return { uri: res.uri, w: nw, h: nh };
  };

  /**
   * F-1: Checks that a cached file URI still exists on disk.
   * Returns the URI if it is still present, or null if it has been GC’d.
   * Used to guard the ‘failed’ translation path where resultUri is not
   * refreshed and may point to a deleted cache file.
   */
  const verifyResultUri = async (uri: string | null): Promise<string | null> => {
    if (!uri) return null;
    try {
      const info = await FileSystem.getInfoAsync(uri);
      return info.exists ? uri : null;
    } catch {
      return null;
    }
  };

  const pickGallery = async () => {
    try {
      const res = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        quality: 1,
        allowsEditing: false,
      });
      if (res.canceled || !res.assets[0]) return;
      const a = res.assets[0];
      const { uri, w, h } = await preprocessImage(a.uri, a.width!, a.height!);
      setS(prev => ({ ...prev, sourceUri: uri, imgW: w, imgH: h, phase: 'editor', error: null }));
    } catch (e: any) {
      Alert.alert('Gallery Error', String(e.message || e));
    }
  };

  const pickCamera = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission required', 'Please allow camera access.');
        return;
      }
      const res = await ImagePicker.launchCameraAsync({ quality: 0.9 });
      if (res.canceled || !res.assets[0]) return;
      const a = res.assets[0];
      const { uri, w, h } = await preprocessImage(a.uri, a.width!, a.height!);
      setS(prev => ({ ...prev, sourceUri: uri, imgW: w, imgH: h, phase: 'editor', error: null }));
    } catch (e: any) {
      Alert.alert('Camera Error', String(e.message || e));
    }
  };

  const runPipeline = async (forceMode?: 'mobile' | 'pc') => {
    if (!s.sourceUri) return;
    tokenRef.current?.cancel();
    const token = new CancellationToken();
    tokenRef.current = token;

    const modeLabel = forceMode === 'mobile' ? '📱 Mobile Only…' : forceMode === 'pc' ? '⚡ PC Server…' : 'Starting…';
    setS(prev => ({
      ...prev, phase: 'processing',
      stage: 'detecting', progress: 0, stageMsg: modeLabel, error: null,
    }));

    const onProgress = (p: PipelineProgress) => {
      if (token.isCancelled) return;
      setS(prev => ({ ...prev, stage: p.stage, progress: p.progress, stageMsg: p.message }));
    };

    try {
      const result = await pipeline.run(
        { imageUri: s.sourceUri, imageWidth: s.imgW, imageHeight: s.imgH, forceMode },
        onProgress,
        token,
      );
      if (token.isCancelled) return;
      setS(prev => ({
        ...prev,
        phase: 'result',
        stage: 'done',
        progress: 100,
        resultUri: result.outputUri,
        result,
        showBefore: false,
      }));
    } catch (e: any) {
      console.error(e);
      Alert.alert('Pipeline Failed', e.message);
      setS(prev => ({ ...prev, phase: 'editor' }));
    }
  };

  const runTranslation = async (preferredPath?: 'mobile' | 'pc') => {
    if (!s.result || s.isTranslating) return;

    // Pass existing mobile OCR results if they exist (to bypass PC detection/OCR)
    const existingOcrResults = s.translationSession?.mobileData?.ocrResults;

    setS(prev => ({ ...prev, isTranslating: true, translationStage: 'Starting…', translationSession: null, renderCoverage: null, mobileSnapshotUri: null,
      // S-2: Increment nonce so any in-flight onSnapshot from the prior session is rejected.
      translationSessionId: prev.translationSessionId + 1 }));

    const onProgress = (stage: string, detail?: string) => {
      setS(prev => ({ ...prev, translationStage: detail ?? stage }));
    };

    try {

      const session = await TranslationService.run(
        s.result,
        s.targetLang,
        onProgress,
        preferredPath,
        existingOcrResults,
        s.sourceLang,
      );



      setS(prev => ({
        ...prev,
        translationSession: session,
        isTranslating: false,
        translationStage: '',
        // For PC path: update resultUri to the server-rendered image
        resultUri: session.path === 'pc' ? session.outputUri : prev.resultUri,
      }));

      // F-1: On the 'failed' path resultUri is not updated, but the cached
      // inpainted file may have been GC’d. Verify it still exists and clear
      // to null if not, so the UI doesn’t silently show a broken image.
      if (session.path === 'failed') {
        const currentUri = s.resultUri;
        verifyResultUri(currentUri).then(verified => {
          if (verified === null && currentUri !== null) {
            console.warn('[App] F-1: resultUri no longer exists on disk — clearing:', currentUri);
            setS(prev => prev.resultUri === currentUri ? { ...prev, resultUri: null } : prev);
          }
        });
      }
    } catch (e: any) {
      console.error('Translation error:', e);
      Alert.alert('Translation Failed', e.message);
      setS(prev => ({ ...prev, isTranslating: false, translationStage: '' }));
    }
  };

  const cancelPipeline = () => {
    tokenRef.current?.cancel();
    setS(prev => ({ ...prev, phase: 'editor', stage: 'idle', progress: 0 }));
  };

  const saveResult = async () => {
    // Priority: flattened mobile snapshot (has text) > resultUri (background or PC-rendered).
    const uriToSave = s.mobileSnapshotUri ?? s.resultUri;
    if (!uriToSave) return;

    const { status } = await MediaLibrary.requestPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission required', 'Please allow media library access.');
      return;
    }

    // Copy to a fresh temp file first.
    // MediaLibrary on Android cannot always read files directly from the
    // app's internal cacheDirectory, so this copy step is mandatory.
    const tmpPath = `${FileSystem.cacheDirectory ?? ''}save_${Date.now()}.jpg`;
    try {
      await FileSystem.copyAsync({ from: uriToSave, to: tmpPath });
      await MediaLibrary.saveToLibraryAsync(tmpPath);
      Alert.alert('Saved!', 'Image saved to your photo gallery.');
    } catch (err: any) {
      console.error('[App] saveResult failed:', err);
      Alert.alert('Save Failed', err?.message ?? 'Could not save image. Please try again.');
    } finally {
      // S-3: Add 200ms grace period before deletion so OEM file-system flushes
      // complete before the temp file is removed.
      await new Promise(r => setTimeout(r, 200));
      FileSystem.deleteAsync(tmpPath, { idempotent: true }).catch(() => {});
    }
  };

  const reset = () => {
    tokenRef.current?.cancel();
    setS(initialState);
  };

  // Phase 2 Settings helpers
  const openSettings = () => setS(prev => ({ ...prev, showSettings: true }));
  const closeSettings = () => setS(prev => ({ ...prev, showSettings: false }));

  const saveManualUrl = () => {
    const url = s.manualUrl.trim();
    if (url) {
      backendService.setManualUrl(url);
      Alert.alert('Saved', `Backend URL set to:\n${url}`);
    } else {
      backendService.clearManualUrl();
      Alert.alert('Cleared', 'Manual backend URL cleared. Auto-discovery will be used.');
    }
    closeSettings();
  };

  const testConnection = async () => {
    const url = s.manualUrl.trim() || s.manualUrl;
    if (!url) { Alert.alert('No URL', 'Enter a URL first.'); return; }
    try {
      const resp = await fetch(url.replace(/\/$/, '') + '/api/health', {
        signal: AbortSignal.timeout(4000),
      });
      if (resp.status === 200) {
        Alert.alert('✅ Connected', 'Server is reachable!');
      } else {
        Alert.alert('⚠️ Error', `Server returned HTTP ${resp.status}`);
      }
    } catch (e: any) {
      Alert.alert('❌ Unreachable', String(e?.message ?? e));
    }
  };

  // Render Helpers
  const renderIdle = () => (
    <View style={styles.idleContainer}>
      <View style={styles.idleTopHalf}>
        <TopBar title="TextRemover" onSettingsPress={openSettings} />

        <View style={styles.idleHeroContent}>
          <Text style={styles.idleHeroTitle}>TextRemover</Text>
          <Text style={styles.idleHeroDesc}>
            On-device ML pipeline with seamless PC server fallback.
          </Text>
        </View>
      </View>

      <View style={styles.idleBottomHalf}>
        <View style={styles.stepsCard}>
          <Text style={styles.stepsTitle}>Pipeline Steps</Text>
          {[
            { n: '1', icon: '🔍', label: 'ML Kit Text Detection' },
            { n: '2', icon: '🎭', label: 'Skia Mask Generation' },
            { n: '3', icon: '🧹', label: 'OpenCV C++ Inpainting' },
            { n: '4', icon: '✨', label: 'Edge Blending + QA' },
          ].map(st => (
            <View key={st.n} style={styles.stepRow}>
              <View style={styles.stepNum}><Text style={styles.stepNumText}>{st.n}</Text></View>
              <Text style={styles.stepIcon}>{st.icon}</Text>
              <Text style={styles.stepLabel}>{st.label}</Text>
            </View>
          ))}
        </View>
      </View>

      <FloatingActionBar>
        <ActionButton id="btn-camera" icon="📷" label="" onPress={pickCamera} primary isCircular />
        <ActionButton id="btn-gallery" icon="🖼️" label="" onPress={pickGallery} primary isCircular />
      </FloatingActionBar>
    </View>
  );

  const renderEditor = () => (
    <View style={styles.editorContainer}>
      {/* Top Floating Pill */}
      <View style={styles.floatingTopBar}>
        <Text style={styles.floatingTopBarText}>Ready to Process</Text>
        <Text style={styles.floatingTopBarSub}>
          {s.imgW} × {s.imgH}
        </Text>
      </View>

      {/* Full screen image preview */}
      <Image
        source={{ uri: s.sourceUri! }}
        style={styles.fullScreenImage}
        resizeMode="contain"
      />

      {s.error && (
        <View style={styles.errorFloat}>
          <Text style={styles.errorText}>⚠️ {s.error}</Text>
        </View>
      )}

      {/* Bottom Floating Bar */}
      <FloatingActionBar>
        <ActionButton id="btn-repick" icon="✕" label="Back" onPress={reset} />
        <ActionButton id="btn-run" icon="✨" label="Remove Text" onPress={() => runPipeline()} primary />
      </FloatingActionBar>
    </View>
  );

  const renderProcessing = () => (
    <View style={styles.editorContainer}>
      <Image
        source={{ uri: s.sourceUri! }}
        style={[styles.fullScreenImage, { opacity: 0.3 }]}
        resizeMode="contain"
      />
      <View style={styles.processingOverlay}>
        <View style={styles.progressContainer}>
          <ProgressCard
            stage={s.stage}
            stageMsg={s.stageMsg}
            progressAnim={progressAnim}
            progressPct={s.progress}
          />
        </View>
      </View>

      <FloatingActionBar>
        <ActionButton id="btn-cancel" icon="✕" label="Cancel" onPress={cancelPipeline} />
      </FloatingActionBar>
    </View>
  );

  const renderResult = () => (
    <View style={styles.resultContainer}>
      {/* Toggle pill — sits right at the top, no floating gap */}
      {!s.isFullscreenPreview && (
        <View style={styles.resultToggleRow}>
          <View style={styles.resultToggle}>
            <Pressable
              id="btn-before"
              style={[styles.toggleBtn, s.showBefore && styles.toggleBtnActive]}
              onPress={() => setS(prev => ({ ...prev, showBefore: true }))}
            >
              <Text style={[styles.toggleBtnText, s.showBefore && styles.toggleBtnTextActive]}>Before</Text>
            </Pressable>
            <View style={styles.toggleDivider} />
            <Pressable
              id="btn-after"
              style={[styles.toggleBtn, !s.showBefore && styles.toggleBtnActive]}
              onPress={() => setS(prev => ({ ...prev, showBefore: false }))}
            >
              <Text style={[styles.toggleBtnText, !s.showBefore && styles.toggleBtnTextActive]}>After</Text>
            </Pressable>
          </View>
        </View>
      )}

      {/* Image fills all remaining space between toggle and bottom sheet */}
      <Pressable 
        style={styles.resultImageContainer}
        onPress={() => setS(prev => ({ ...prev, isFullscreenPreview: !prev.isFullscreenPreview }))}
      >
        {FEATURES.STAGE7_MOBILE_ENABLED
          && s.translationSession?.path === 'mobile'
          && s.translationSession.mobileData
          && !s.showBefore ? (
          // Stage 7 Mobile: SkiaRenderer overlays translated text on the inpainted image
          <TranslatedOverlay
            inpaintedUri={s.translationSession.outputUri}
            originalUri={s.translationSession.originalUri}
            imageWidth={s.translationSession.mobileData.imageWidth}
            imageHeight={s.translationSession.mobileData.imageHeight}
            ocrResults={s.translationSession.mobileData.ocrResults}
            translationResults={s.translationSession.mobileData.translationResults}
            targetLang={s.translationSession.targetLang}
            onRenderComplete={(rendered, total) => {
              console.log(`[App] Stage 7 render complete: ${rendered}/${total} regions`);
              setS(prev => {
                const cov = total > 0 ? rendered / total : 1;
                if (prev.renderCoverage === cov) return prev; // Break the re-render cycle
                return { ...prev, renderCoverage: cov };
              });
            }}
            onSnapshot={(uri) => {
              // BUG-6: Read the live session ID from the ref, not the stale closure.
              // The closure-captured s.translationSessionId can be stale if App
              // re-rendered between when this callback was created and when it fires.
              const capturedSessionId = sessionIdRef.current;
              console.log('[App] Stage 7 snapshot received:', uri);
              setS(prev => {
                if (prev.mobileSnapshotUri === uri) return prev; // idempotent
                if (prev.translationSessionId !== capturedSessionId) {
                  console.warn('[App] BUG-6: Stale snapshot rejected (session changed)');
                  return prev;
                }
                return {
                  ...prev,
                  mobileSnapshotUri: uri,
                  resultUri: uri,
                };
              });
            }}
          />
        ) : (
          // Default: show result image (erase-only, or PC-rendered translation)
          <Image
            source={{ uri: s.showBefore ? s.sourceUri! : s.resultUri! }}
            style={styles.resultImage}
            resizeMode="contain"
          />
        )}
      </Pressable>

      {/* Bottom sheet — fixed at bottom, never overlaps image */}
      {!s.isFullscreenPreview && (
        <ResultBottomSheet
          result={s.result!}
          onSave={saveResult}
          onRetry={reset}
          onOpenSettings={openSettings}
          onForceMobile={() => runPipeline('mobile')}
          onForceServer={() => runPipeline('pc')}
          onTranslate={() => runTranslation()}
          onForceTranslateMobile={() => runTranslation('mobile')}
          onForceTranslateServer={() => runTranslation('pc')}
          isTranslating={s.isTranslating}
          translationStage={s.translationStage}
          targetLang={s.targetLang}
          onTargetLangChange={(lang) => setS(prev => ({
            ...prev,
            targetLang: lang,
            // ST-1: Clear previous translation when target language changes so the
            // UI signals to the user that re-translation is needed.
            translationSession: null,
            mobileSnapshotUri: null,
            renderCoverage: null,
          }))}
          sourceLang={s.sourceLang}
          onSourceLangChange={(lang) => setS(prev => ({
            ...prev,
            sourceLang: lang,
            // Clear previous translation session when source language changes
            // so the UI signals to the user that re-translation is needed.
            translationSession: null,
            mobileSnapshotUri: null,
          }))}
          translationSession={s.translationSession}
          renderCoverage={s.renderCoverage}
          // S-1: Save is only ready when:
          // • Translation is not on the mobile path (no snapshot needed), OR
          // • The mobile snapshot has already been generated
          isSaveReady={
            !(
              FEATURES.TRANSLATION_ENABLED &&
              s.translationSession?.path === 'mobile' &&
              !s.mobileSnapshotUri
            )
          }
        />
      )}
    </View>
  );

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar
        barStyle={s.phase === 'idle' ? 'light-content' : 'dark-content'}
        backgroundColor={s.phase === 'idle' ? Theme.colors.primary : Theme.colors.background}
      />

      {s.phase === 'idle' && renderIdle()}
      {s.phase === 'editor' && renderEditor()}
      {s.phase === 'processing' && renderProcessing()}
      {s.phase === 'result' && renderResult()}

      {/* Settings Modal */}
      <SettingsModal
        visible={s.showSettings}
        manualUrl={s.manualUrl}
        onClose={closeSettings}
        onChangeUrl={v => setS(prev => ({ ...prev, manualUrl: v }))}
        onSave={saveManualUrl}
        onTest={testConnection}
      />
    </SafeAreaView>
  );
}

// Styles
const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: Theme.colors.background,
  },

  // Idle State
  idleContainer: {
    flex: 1,
  },
  idleTopHalf: {
    height: '40%',
    backgroundColor: Theme.colors.primary,
  },
  idleHeroContent: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Theme.spacing.lg,
  },
  idleHeroTitle: {
    ...Theme.typography.h1,
    color: Theme.colors.textInverse,
    marginBottom: Theme.spacing.sm,
  },
  idleHeroDesc: {
    ...Theme.typography.body,
    color: 'rgba(255,255,255,0.8)',
    textAlign: 'center',
  },
  idleBottomHalf: {
    height: '60%',
    backgroundColor: Theme.colors.background,
    borderTopLeftRadius: Theme.radius.lg,
    borderTopRightRadius: Theme.radius.lg,
    marginTop: -Theme.spacing.lg,
    paddingTop: Theme.spacing.xxl,
    paddingHorizontal: Theme.spacing.lg,
  },
  stepsCard: {
    backgroundColor: Theme.colors.surface,
    borderRadius: Theme.radius.md,
    padding: Theme.spacing.lg,
    ...Theme.shadows.sm,
  },
  stepsTitle: {
    ...Theme.typography.subtitle,
    marginBottom: Theme.spacing.md,
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Theme.spacing.md,
  },
  stepNum: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: Theme.colors.surfaceAlt,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: Theme.spacing.sm,
  },
  stepNumText: {
    ...Theme.typography.caption,
    fontWeight: '700',
    color: Theme.colors.textSecondary,
  },
  stepIcon: {
    fontSize: 18,
    marginRight: Theme.spacing.sm,
  },
  stepLabel: {
    ...Theme.typography.body,
  },

  // Editor / Processing / Result State
  editorContainer: {
    flex: 1,
    backgroundColor: Theme.colors.background,
  },
  fullScreenImage: {
    width: '100%',
    height: '100%',
    position: 'absolute',
  },
  floatingTopBar: {
    position: 'absolute',
    top: Theme.spacing.xl,
    alignSelf: 'center',
    backgroundColor: Theme.colors.surface,
    paddingHorizontal: Theme.spacing.xl,
    paddingVertical: Theme.spacing.md,
    borderRadius: Theme.radius.round,
    alignItems: 'center',
    ...Theme.shadows.md,
    zIndex: 10,
  },
  floatingTopBarText: {
    ...Theme.typography.button,
    color: Theme.colors.text,
  },
  floatingTopBarSub: {
    ...Theme.typography.caption,
    color: Theme.colors.textSecondary,
    marginTop: 2,
  },
  errorFloat: {
    position: 'absolute',
    top: 100,
    alignSelf: 'center',
    backgroundColor: Theme.colors.errorBg,
    padding: Theme.spacing.md,
    borderRadius: Theme.radius.sm,
    marginHorizontal: Theme.spacing.lg,
    ...Theme.shadows.sm,
  },
  errorText: {
    color: Theme.colors.error,
    ...Theme.typography.body,
  },

  // Processing specific
  processingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(255,255,255,0.7)',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Theme.spacing.lg,
  },
  progressContainer: {
    width: '100%',
  },

  // Result specific — flex column layout: toggle → image → bottom sheet
  resultContainer: {
    flex: 1,
    flexDirection: 'column',
    backgroundColor: Theme.colors.background,
  },
  resultToggleRow: {
    paddingTop: Theme.spacing.sm,
    paddingBottom: Theme.spacing.sm,
    alignItems: 'center',
    backgroundColor: Theme.colors.background,
  },
  resultToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Theme.colors.surface,
    borderRadius: Theme.radius.round,
    padding: 4,
    ...Theme.shadows.sm,
  },
  resultImageContainer: {
    flex: 1,
    width: '100%',
  },
  resultImage: {
    flex: 1,
    width: '100%',
  },
  floatingTopToggle: {
    // kept for reference but replaced by resultToggle in result screen
    position: 'absolute',
    top: Theme.spacing.xl,
    alignSelf: 'center',
    backgroundColor: Theme.colors.surface,
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: Theme.radius.round,
    padding: 4,
    ...Theme.shadows.md,
    zIndex: 10,
  },
  toggleBtn: {
    paddingHorizontal: Theme.spacing.xl,
    paddingVertical: Theme.spacing.sm,
    borderRadius: Theme.radius.round,
  },
  toggleBtnActive: {
    backgroundColor: Theme.colors.primary,
  },
  toggleBtnText: {
    ...Theme.typography.button,
    color: Theme.colors.textSecondary,
  },
  toggleBtnTextActive: {
    color: Theme.colors.textInverse,
  },
  toggleDivider: {
    width: 1,
    height: 20,
    backgroundColor: Theme.colors.border,
    marginHorizontal: 4,
  },
  bottomSheetContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
  },
});
