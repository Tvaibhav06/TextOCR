/**
 * App.tsx — Phase 1 Track 4: Text Remover POC UI
 *
 * Single-screen, 4-state flow:
 *   A. Idle        → pick image (gallery / camera)
 *   B. Editor      → view image + detection boxes, trigger pipeline
 *   C. Processing  → animated progress bar, live stage label, cancel
 *   D. Result      → before/after toggle, SSIM badge, save / retry
 *
 * No navigator needed for Phase 1 — state machine lives in useTextRemoval hook.
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  Alert,
  Animated,
  Dimensions,
  Image,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
  ActivityIndicator,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as MediaLibrary from 'expo-media-library';
import * as ImageManipulator from 'expo-image-manipulator';

import {
  pipeline,
  CancellationToken,
  type PipelineResult,
  type PipelineProgress,
  type PipelineStage,
} from './src/pipeline/Pipeline';

// ─── Design Tokens ────────────────────────────────────────────────────────────

const T = {
  bg:         '#0A0A0F',
  surface:    '#14141E',
  surfaceAlt: '#1C1C2E',
  border:     '#2A2A3E',
  primary:    '#6366F1',
  primaryDim: '#3730A3',
  success:    '#22C55E',
  warning:    '#F59E0B',
  error:      '#EF4444',
  text:       '#F1F1F1',
  textMuted:  '#9CA3AF',
  textDim:    '#6B7280',
  accent:     '#818CF8',
};

const { width: SW, height: SH } = Dimensions.get('window');

// ─── Stage label map ──────────────────────────────────────────────────────────

const STAGE_LABELS: Record<PipelineStage, string> = {
  idle:           'Ready',
  detecting:      'Stage 1 — Detecting text…',
  masking:        'Stage 2 — Generating mask…',
  inpainting:     'Stage 3 — Erasing text (OpenCV)…',
  postprocessing: 'Stage 4 — Blending edges…',
  done:           'Complete!',
  error:          'Error',
};

// ─── App State ────────────────────────────────────────────────────────────────

type AppPhase = 'idle' | 'editor' | 'processing' | 'result';

interface AppState {
  phase:      AppPhase;
  sourceUri:  string | null;
  resultUri:  string | null;
  imgW:       number;
  imgH:       number;
  stage:      PipelineStage;
  progress:   number;
  stageMsg:   string;
  result:     PipelineResult | null;
  error:      string | null;
  showBefore: boolean;
}

const INIT: AppState = {
  phase: 'idle', sourceUri: null, resultUri: null,
  imgW: 1, imgH: 1, stage: 'idle',
  progress: 0, stageMsg: '', result: null,
  error: null, showBefore: true,
};

// ─── Main Component ───────────────────────────────────────────────────────────

export default function App() {
  const [s, setS] = useState<AppState>(INIT);
  const tokenRef  = useRef<CancellationToken | null>(null);
  const progressAnim = useRef(new Animated.Value(0)).current;

  // Warm-up pipeline models on mount
  useEffect(() => {
    pipeline.warmUp().catch(() => {});
    return () => { tokenRef.current?.cancel(); };
  }, []);

  // Animate progress bar whenever s.progress changes
  useEffect(() => {
    Animated.timing(progressAnim, {
      toValue: s.progress,
      duration: 400,
      useNativeDriver: false,
    }).start();
  }, [s.progress]);

  // ── Helpers ──────────────────────────────────────────────────────────────

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

  const runPipeline = async () => {
    if (!s.sourceUri) return;
    tokenRef.current?.cancel();
    const token = new CancellationToken();
    tokenRef.current = token;

    setS(prev => ({
      ...prev, phase: 'processing',
      stage: 'detecting', progress: 0, stageMsg: 'Starting…', error: null,
    }));

    const onProgress = (p: PipelineProgress) => {
      if (token.isCancelled) return;
      setS(prev => ({ ...prev, stage: p.stage, progress: p.progress, stageMsg: p.message }));
    };

    try {
      const result = await pipeline.run(
        { imageUri: s.sourceUri, imageWidth: s.imgW, imageHeight: s.imgH },
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
    } catch (err: any) {
      if (token.isCancelled) return;
      const errorMessage = err?.message ?? 'Pipeline failed. Check logcat for details.';
      setS(prev => ({
        ...prev,
        phase: 'editor',
        stage: 'idle',
        error: errorMessage,
      }));
    }
  };

  const cancelPipeline = () => {
    tokenRef.current?.cancel();
    setS(prev => ({ ...prev, phase: 'editor', stage: 'idle', progress: 0 }));
  };

  const saveResult = async () => {
    if (!s.resultUri) return;
    const { status } = await MediaLibrary.requestPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission required', 'Please allow media library access.');
      return;
    }
    await MediaLibrary.saveToLibraryAsync(s.resultUri);
    Alert.alert('Saved!', 'Image saved to your photo gallery.');
  };

  const reset = () => {
    tokenRef.current?.cancel();
    setS(INIT);
  };

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" backgroundColor={T.bg} />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>TextRemover</Text>
        <Text style={styles.headerSub}>On-device OpenCV pipeline</Text>
      </View>

      {/* ── State A: Idle ── */}
      {s.phase === 'idle' && (
        <View style={styles.centerFlex}>
          <View style={styles.heroBox}>
            <Text style={styles.heroIcon}>🔍</Text>
            <Text style={styles.heroTitle}>Remove Text from Images</Text>
            <Text style={styles.heroDesc}>
              Powered by ML Kit detection + native OpenCV inpainting.
              All processing happens on-device.
            </Text>
          </View>

          <Btn
            id="btn-gallery"
            icon="🖼️"
            label="Pick from Gallery"
            onPress={pickGallery}
            primary
          />
          <Btn
            id="btn-camera"
            icon="📷"
            label="Use Camera"
            onPress={pickCamera}
          />

          <PipelineStepsCard />
        </View>
      )}

      {/* ── State B: Editor ── */}
      {s.phase === 'editor' && s.sourceUri && (
        <ScrollView style={styles.flex} contentContainerStyle={styles.editorContent}>
          <Text style={styles.sectionLabel}>Selected Image</Text>

          <View style={styles.imageFrame}>
            <Image
              source={{ uri: s.sourceUri }}
              style={styles.imagePreview}
              resizeMode="contain"
            />
            <View style={styles.imageBadge}>
              <Text style={styles.imageBadgeText}>
                {s.imgW} × {s.imgH}
              </Text>
            </View>
          </View>

          {s.error ? (
            <View style={styles.errorBanner}>
              <Text style={styles.errorText}>⚠️ {s.error}</Text>
            </View>
          ) : null}

          <View style={styles.infoCard}>
            <Text style={styles.infoCardText}>
              ℹ️ ML Kit will auto-detect text regions. If no text is found,
              the pipeline returns the original image unchanged.
            </Text>
          </View>

          <Btn
            id="btn-run"
            icon="🗑️"
            label="Remove Text"
            onPress={runPipeline}
            primary
          />
          <Btn
            id="btn-repick"
            icon="↩"
            label="Pick Another Image"
            onPress={reset}
          />
        </ScrollView>
      )}

      {/* ── State C: Processing ── */}
      {s.phase === 'processing' && s.sourceUri && (
        <View style={styles.centerFlex}>
          <View style={styles.processingImageWrap}>
            <Image
              source={{ uri: s.sourceUri }}
              style={styles.processingImage}
              resizeMode="contain"
            />
            <View style={styles.processingOverlay} />
          </View>

          <View style={styles.progressCard}>
            <Text style={styles.progressStageLabel}>
              {STAGE_LABELS[s.stage] ?? s.stageMsg}
            </Text>

            {/* Progress bar */}
            <View style={styles.progressTrack}>
              <Animated.View
                style={[
                  styles.progressFill,
                  {
                    width: progressAnim.interpolate({
                      inputRange: [0, 100],
                      outputRange: ['0%', '100%'],
                    }),
                  },
                ]}
              />
            </View>

            <Text style={styles.progressPct}>{Math.round(s.progress)}%</Text>

            {/* Stage dots */}
            <View style={styles.stageDots}>
              {(['detecting', 'masking', 'inpainting', 'postprocessing'] as PipelineStage[]).map((st, i) => {
                const stageOrder: Record<PipelineStage, number> = {
                  idle: 0, detecting: 1, masking: 2, inpainting: 3,
                  postprocessing: 4, done: 5, error: 0,
                };
                const current = stageOrder[s.stage] ?? 0;
                const thisIdx = i + 1;
                const done    = current > thisIdx;
                const active  = current === thisIdx;
                return (
                  <View key={st} style={styles.stageDotItem}>
                    <View style={[
                      styles.stageDot,
                      done   && styles.stageDotDone,
                      active && styles.stageDotActive,
                    ]}>
                      {active && <ActivityIndicator size="small" color={T.bg} />}
                      {done   && <Text style={styles.stageDotCheck}>✓</Text>}
                      {!active && !done && <Text style={styles.stageDotNum}>{i + 1}</Text>}
                    </View>
                    <Text style={styles.stageDotLabel} numberOfLines={1}>
                      {['Detect', 'Mask', 'Erase', 'Blend'][i]}
                    </Text>
                  </View>
                );
              })}
            </View>
          </View>

          <Btn id="btn-cancel" icon="✕" label="Cancel" onPress={cancelPipeline} />
        </View>
      )}

      {/* ── State D: Result ── */}
      {s.phase === 'result' && s.result && (
        <ScrollView style={styles.flex} contentContainerStyle={styles.editorContent}>
          {/* Quality badge */}
          <View style={styles.qualityRow}>
            <QualityBadge ssim={s.result.quality.ssim} above={s.result.quality.aboveThreshold} />
            <Text style={styles.timingText}>
              {s.result.totalMs}ms  ·  {s.result.detections.length} region{s.result.detections.length !== 1 ? 's' : ''}
            </Text>
          </View>

          {/* Before / After toggle */}
          <View style={styles.toggleRow}>
            <Pressable
              id="btn-before"
              style={[styles.toggleBtn, s.showBefore && styles.toggleBtnActive]}
              onPress={() => setS(prev => ({ ...prev, showBefore: true }))}>
              <Text style={[styles.toggleBtnText, s.showBefore && styles.toggleBtnTextActive]}>
                Before
              </Text>
            </Pressable>
            <Pressable
              id="btn-after"
              style={[styles.toggleBtn, !s.showBefore && styles.toggleBtnActive]}
              onPress={() => setS(prev => ({ ...prev, showBefore: false }))}>
              <Text style={[styles.toggleBtnText, !s.showBefore && styles.toggleBtnTextActive]}>
                After
              </Text>
            </Pressable>
          </View>

          <View style={styles.imageFrame}>
            <Image
              source={{ uri: s.showBefore ? s.sourceUri! : s.resultUri! }}
              style={styles.imagePreview}
              resizeMode="contain"
            />
            <View style={[styles.imageBadge, { backgroundColor: s.showBefore ? '#374151' : T.primaryDim }]}>
              <Text style={styles.imageBadgeText}>
                {s.showBefore ? 'ORIGINAL' : 'PROCESSED'}
              </Text>
            </View>
          </View>

          {/* Server upgrade hint */}
          {s.result.quality.suggestServerUpgrade && (
            <View style={styles.serverHintBanner}>
              <Text style={styles.serverHintTitle}>⚡ Phase 2 Enhancement Available</Text>
              <Text style={styles.serverHintBody}>
                Local quality (SSIM {s.result.quality.ssim.toFixed(3)}) is below target.
                A cloud LaMa pass would improve results — available in Phase 2.
              </Text>
            </View>
          )}

          {/* Timing breakdown */}
          <TimingCard timing={s.result.timing} />

          <Btn id="btn-save"  icon="💾" label="Save to Gallery" onPress={saveResult} primary />
          <Btn id="btn-retry" icon="↩" label="Try Another Image" onPress={reset} />
        </ScrollView>
      )}
    </SafeAreaView>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

interface BtnProps {
  id: string;
  icon: string;
  label: string;
  onPress: () => void;
  primary?: boolean;
}

function Btn({ id, icon, label, onPress, primary }: BtnProps) {
  const [pressed, setPressed] = useState(false);
  return (
    <Pressable
      id={id}
      style={[
        styles.btn,
        primary ? styles.btnPrimary : styles.btnSecondary,
        pressed && styles.btnPressed,
      ]}
      onPressIn={() => setPressed(true)}
      onPressOut={() => setPressed(false)}
      onPress={onPress}>
      <Text style={styles.btnIcon}>{icon}</Text>
      <Text style={[styles.btnLabel, primary && styles.btnLabelPrimary]}>{label}</Text>
    </Pressable>
  );
}

function QualityBadge({ ssim, above }: { ssim: number; above: boolean }) {
  return (
    <View style={[styles.badge, above ? styles.badgeGood : styles.badgeWarn]}>
      <Text style={styles.badgeText}>
        {above ? '✅' : '⚠️'} SSIM {ssim.toFixed(3)} — {above ? 'PASS' : 'LOW'}
      </Text>
    </View>
  );
}

function PipelineStepsCard() {
  const steps = [
    { n: '1', icon: '🔍', label: 'ML Kit Text Detection' },
    { n: '2', icon: '🎭', label: 'Skia Mask Generation' },
    { n: '3', icon: '🧹', label: 'OpenCV C++ Inpainting' },
    { n: '4', icon: '✨', label: 'Edge Blending + QA' },
  ];
  return (
    <View style={styles.stepsCard}>
      <Text style={styles.stepsTitle}>4-Stage On-Device Pipeline</Text>
      {steps.map(st => (
        <View key={st.n} style={styles.stepRow}>
          <View style={styles.stepNum}><Text style={styles.stepNumText}>{st.n}</Text></View>
          <Text style={styles.stepIcon}>{st.icon}</Text>
          <Text style={styles.stepLabel}>{st.label}</Text>
        </View>
      ))}
    </View>
  );
}

function TimingCard({ timing }: { timing: any[] }) {
  if (!timing?.length) return null;
  return (
    <View style={styles.timingCard}>
      <Text style={styles.timingCardTitle}>Stage Timings</Text>
      {timing.map((t, i) => (
        <View key={i} style={styles.timingRow}>
          <Text style={styles.timingStage}>{t.stage}</Text>
          <View style={styles.timingBarWrap}>
            <View style={[styles.timingBar, { width: `${Math.min(100, t.ms / 5)}%` }]} />
          </View>
          <Text style={styles.timingMs}>{t.ms}ms</Text>
        </View>
      ))}
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  safeArea:   { flex: 1, backgroundColor: T.bg },
  flex:       { flex: 1 },
  centerFlex: { flex: 1, alignItems: 'center', justifyContent: 'center', paddingHorizontal: 20 },

  // Header
  header:     { paddingHorizontal: 20, paddingVertical: 14, borderBottomWidth: 1, borderBottomColor: T.border },
  headerTitle:{ fontSize: 20, fontWeight: '700', color: T.text, letterSpacing: 0.5 },
  headerSub:  { fontSize: 12, color: T.textDim, marginTop: 2 },

  // Hero (idle)
  heroBox:    { alignItems: 'center', marginBottom: 36, paddingHorizontal: 16 },
  heroIcon:   { fontSize: 56, marginBottom: 16 },
  heroTitle:  { fontSize: 22, fontWeight: '700', color: T.text, textAlign: 'center', marginBottom: 10 },
  heroDesc:   { fontSize: 14, color: T.textMuted, textAlign: 'center', lineHeight: 22 },

  // Buttons
  btn:        { flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
                width: '100%', paddingVertical: 15, paddingHorizontal: 20, borderRadius: 14,
                marginBottom: 12 },
  btnPrimary: { backgroundColor: T.primary },
  btnSecondary:{ backgroundColor: T.surfaceAlt, borderWidth: 1, borderColor: T.border },
  btnPressed: { opacity: 0.75 },
  btnIcon:    { fontSize: 18, marginRight: 10 },
  btnLabel:   { fontSize: 16, fontWeight: '600', color: T.textMuted },
  btnLabelPrimary: { color: '#FFFFFF' },

  // Editor
  editorContent: { paddingHorizontal: 16, paddingBottom: 32 },
  sectionLabel:  { fontSize: 12, fontWeight: '600', color: T.textDim, letterSpacing: 1,
                   textTransform: 'uppercase', marginTop: 16, marginBottom: 10 },

  imageFrame:    { borderRadius: 16, overflow: 'hidden', backgroundColor: T.surface,
                   borderWidth: 1, borderColor: T.border, marginBottom: 16 },
  imagePreview:  { width: '100%', height: SW * 0.65 },
  imageBadge:    { position: 'absolute', bottom: 10, right: 10, backgroundColor: T.primaryDim,
                   paddingHorizontal: 10, paddingVertical: 4, borderRadius: 8 },
  imageBadgeText:{ fontSize: 11, fontWeight: '700', color: '#FFF', letterSpacing: 0.5 },

  errorBanner:   { backgroundColor: '#1F1010', borderRadius: 10, padding: 14,
                   borderWidth: 1, borderColor: T.error, marginBottom: 12 },
  errorText:     { color: T.error, fontSize: 13, lineHeight: 20 },

  infoCard:      { backgroundColor: T.surfaceAlt, borderRadius: 10, padding: 14,
                   borderWidth: 1, borderColor: T.border, marginBottom: 16 },
  infoCardText:  { fontSize: 13, color: T.textMuted, lineHeight: 20 },

  // Processing
  processingImageWrap: { width: SW - 32, height: SW * 0.5, borderRadius: 16,
                          overflow: 'hidden', marginBottom: 24 },
  processingImage:     { width: '100%', height: '100%' },
  processingOverlay:   { ...StyleSheet.absoluteFillObject, backgroundColor: 'rgba(0,0,0,0.55)' },

  progressCard:    { width: '100%', backgroundColor: T.surface, borderRadius: 16,
                     padding: 20, borderWidth: 1, borderColor: T.border, marginBottom: 20 },
  progressStageLabel:{ fontSize: 14, fontWeight: '600', color: T.accent, marginBottom: 14,
                        textAlign: 'center' },
  progressTrack:   { height: 8, backgroundColor: T.surfaceAlt, borderRadius: 4, overflow: 'hidden' },
  progressFill:    { height: '100%', backgroundColor: T.primary, borderRadius: 4 },
  progressPct:     { fontSize: 13, color: T.textDim, textAlign: 'right', marginTop: 6 },

  stageDots:       { flexDirection: 'row', justifyContent: 'space-around', marginTop: 20 },
  stageDotItem:    { alignItems: 'center', flex: 1 },
  stageDot:        { width: 36, height: 36, borderRadius: 18, backgroundColor: T.surfaceAlt,
                     borderWidth: 1.5, borderColor: T.border, alignItems: 'center',
                     justifyContent: 'center', marginBottom: 6 },
  stageDotActive:  { backgroundColor: T.primary, borderColor: T.primary },
  stageDotDone:    { backgroundColor: T.success, borderColor: T.success },
  stageDotNum:     { fontSize: 13, color: T.textDim, fontWeight: '600' },
  stageDotCheck:   { fontSize: 14, color: '#FFF', fontWeight: '700' },
  stageDotLabel:   { fontSize: 10, color: T.textDim, textAlign: 'center' },

  // Result
  qualityRow:      { flexDirection: 'row', alignItems: 'center',
                     justifyContent: 'space-between', marginTop: 16, marginBottom: 12 },
  timingText:      { fontSize: 12, color: T.textDim },
  toggleRow:       { flexDirection: 'row', backgroundColor: T.surfaceAlt, borderRadius: 12,
                     padding: 4, marginBottom: 12, borderWidth: 1, borderColor: T.border },
  toggleBtn:       { flex: 1, paddingVertical: 8, alignItems: 'center', borderRadius: 9 },
  toggleBtnActive: { backgroundColor: T.primary },
  toggleBtnText:   { fontSize: 14, fontWeight: '600', color: T.textDim },
  toggleBtnTextActive: { color: '#FFF' },

  badge:      { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20 },
  badgeGood:  { backgroundColor: 'rgba(34,197,94,0.15)', borderWidth: 1, borderColor: T.success },
  badgeWarn:  { backgroundColor: 'rgba(245,158,11,0.15)', borderWidth: 1, borderColor: T.warning },
  badgeText:  { fontSize: 12, fontWeight: '700', color: T.text },

  serverHintBanner: { backgroundColor: 'rgba(99,102,241,0.1)', borderWidth: 1,
                       borderColor: T.primary, borderRadius: 12, padding: 14, marginBottom: 12 },
  serverHintTitle:  { fontSize: 13, fontWeight: '700', color: T.accent, marginBottom: 4 },
  serverHintBody:   { fontSize: 12, color: T.textMuted, lineHeight: 18 },

  // Steps card (idle)
  stepsCard:     { width: '100%', backgroundColor: T.surface, borderRadius: 16,
                   padding: 16, borderWidth: 1, borderColor: T.border, marginTop: 8 },
  stepsTitle:    { fontSize: 13, fontWeight: '700', color: T.textMuted, marginBottom: 14,
                   textTransform: 'uppercase', letterSpacing: 0.8 },
  stepRow:       { flexDirection: 'row', alignItems: 'center', marginBottom: 10 },
  stepNum:       { width: 26, height: 26, borderRadius: 13, backgroundColor: T.primaryDim,
                   alignItems: 'center', justifyContent: 'center', marginRight: 10 },
  stepNumText:   { fontSize: 12, fontWeight: '700', color: T.accent },
  stepIcon:      { fontSize: 16, marginRight: 8 },
  stepLabel:     { fontSize: 13, color: T.textMuted },

  // Timing card (result)
  timingCard:    { backgroundColor: T.surface, borderRadius: 12, padding: 14,
                   borderWidth: 1, borderColor: T.border, marginBottom: 16 },
  timingCardTitle:{ fontSize: 12, fontWeight: '700', color: T.textDim, marginBottom: 10,
                    textTransform: 'uppercase', letterSpacing: 0.8 },
  timingRow:     { flexDirection: 'row', alignItems: 'center', marginBottom: 8 },
  timingStage:   { width: 110, fontSize: 11, color: T.textMuted },
  timingBarWrap: { flex: 1, height: 6, backgroundColor: T.surfaceAlt,
                   borderRadius: 3, overflow: 'hidden', marginHorizontal: 8 },
  timingBar:     { height: '100%', backgroundColor: T.primaryDim, borderRadius: 3 },
  timingMs:      { width: 50, fontSize: 11, color: T.textDim, textAlign: 'right' },
});
