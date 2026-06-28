import React, { useState } from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { Theme } from './Theme';
import { PipelineResult } from '../pipeline/Pipeline';
import { ActionButton } from './FloatingActionBar';
import { FEATURES } from '../config/features';
import { AVAILABLE_LANGUAGES } from '../pipeline/translation/MobileTranslate';
import { Modal, ScrollView, TouchableOpacity } from 'react-native';
import type { TranslationSession } from '../services/TranslationService';

interface ResultBottomSheetProps {
  result: PipelineResult;
  onSave: () => void;
  onRetry: () => void;
  onOpenSettings: () => void;
  /**
   * Manual compute override handlers.
   * Easy to remove: just delete these two props and the Override row below.
   */
  onForceMobile: () => void;
  onForceServer: () => void;
  /** Translation Stage action */
  onTranslate?: () => void;
  onForceTranslateMobile?: () => void;
  onForceTranslateServer?: () => void;
  isTranslating?: boolean;
  /** Current translation progress stage message (shown while translating). */
  translationStage?: string;
  targetLang?: string;
  onTargetLangChange?: (lang: string) => void;
  /** Set after a translation completes — drives the translation status banner. */
  translationSession?: TranslationSession | null;
  /** Stage 7 mobile render coverage: 0–1. null = not yet rendered. */
  renderCoverage?: number | null;
}

export function ResultBottomSheet({ 
  result, 
  onSave, 
  onRetry, 
  onOpenSettings, 
  onForceMobile, 
  onForceServer,
  onTranslate,
  onForceTranslateMobile,
  onForceTranslateServer,
  isTranslating = false,
  translationStage,
  targetLang,
  onTargetLangChange,
  translationSession,
  renderCoverage,
}: ResultBottomSheetProps) {
  const [showTimings, setShowTimings] = useState(false);
  const [showLangPicker, setShowLangPicker] = useState(false);

  const { quality, backend, timing, detections } = result;

  const srcLangs = new Set<string>();
  if (translationSession?.path === 'mobile' && translationSession.mobileData?.translationResults) {
    translationSession.mobileData.translationResults.forEach(r => {
      if (r.srcLang && r.srcLang !== 'und' && r.srcLang !== 'auto') {
        const name = AVAILABLE_LANGUAGES.find(l => l.code === r.srcLang)?.name || r.srcLang;
        srcLangs.add(name);
      }
    });
  } else if (translationSession?.path === 'pc' && translationSession.pcSourceLangs) {
    translationSession.pcSourceLangs.forEach(langCode => {
      const name = AVAILABLE_LANGUAGES.find(l => l.code === langCode)?.name || langCode;
      srcLangs.add(name);
    });
  }
  const srcLangsArray = Array.from(srcLangs);

  return (
    <View style={styles.sheet}>
      <View style={styles.dragHandle} />
      
      {/* Quality & Metadata Row */}
      <View style={styles.headerRow}>
        <View style={[styles.badge, quality.aboveThreshold ? styles.badgeGood : styles.badgeWarn]}>
          <Text style={styles.badgeText}>
            {quality.aboveThreshold ? '✅' : '⚠️'} SSIM {quality.ssim.toFixed(3)}
          </Text>
        </View>
        <Pressable onPress={() => setShowTimings(!showTimings)}>
          <Text style={styles.metaText}>
            {result.totalMs}ms • {detections.length} region{detections.length !== 1 ? 's' : ''}
          </Text>
        </Pressable>
      </View>

      {/* Erase Status Banners (Stage 1–4) */}
      {backend?.used ? (
        <View style={styles.bannerSuccess}>
          <Text style={styles.bannerTitle}>⚡ Enhanced by PC Server (LaMa)</Text>
          <Text style={styles.bannerBody}>
            Server SSIM {backend.ssim.toFixed(3)} • Tier: {backend.inpaintTier} • {backend.totalMs.toFixed(0)}ms
          </Text>
        </View>
      ) : quality.suggestServerUpgrade ? (
        <Pressable style={styles.bannerHint} onPress={onOpenSettings}>
          <Text style={styles.bannerTitleHint}>⚡ Server Enhancement Available</Text>
          <Text style={styles.bannerBody}>
            Local quality is below target. Tap to configure server URL and try again for LaMa quality.
          </Text>
        </Pressable>
      ) : null}

      {/* Translation Status Banner (Stage 5–7) */}
      {isTranslating && (
        <View style={styles.bannerTranslating}>
          <Text style={styles.bannerTitle}>⏳ Translating…</Text>
          {translationStage ? (
            <Text style={styles.bannerBody}>{translationStage}</Text>
          ) : null}
        </View>
      )}
      {!isTranslating && translationSession?.path === 'mobile' && (
        <View style={styles.bannerTranslateSuccess}>
          <Text style={styles.bannerTitle}>
            📱 Translated on Device • {translationSession.translatedCount}/{translationSession.detectionCount} regions • {translationSession.totalMs}ms
          </Text>
        </View>
      )}
      {!isTranslating && translationSession?.path === 'pc' && (
        <View style={styles.bannerTranslatePc}>
          <Text style={styles.bannerTitle}>
            ⚡ PC Server Translation • {translationSession.translatedCount}/{translationSession.detectionCount} regions • {translationSession.totalMs}ms
          </Text>
          {translationSession.fallbackDetail ? (
            <Text style={styles.bannerBody}>Mobile fallback: {translationSession.fallbackDetail}</Text>
          ) : null}
        </View>
      )}
      {!isTranslating && (translationSession?.path === 'mobile' || translationSession?.path === 'pc') && srcLangsArray.length > 0 && (
        <View style={styles.sourceLangContainer}>
          <Text style={styles.sourceLangText}>
            Source language{srcLangsArray.length > 1 ? 's' : ''} detected as {srcLangsArray.join(', ')}
          </Text>
        </View>
      )}
      {!isTranslating && translationSession?.path === 'failed' && (
        <View style={styles.bannerTranslateFail}>
          <Text style={styles.bannerTitle}>❌ Translation Failed</Text>
          {translationSession.fallbackDetail ? (
            <Text style={styles.bannerBody}>{translationSession.fallbackDetail}</Text>
          ) : null}
        </View>
      )}

      {/* Stage 7 Mobile Render Coverage Warning */}
      {!isTranslating
        && translationSession?.path === 'mobile'
        && renderCoverage !== null
        && renderCoverage !== undefined
        && renderCoverage < 0.5 && (
        <View style={styles.bannerWarnRender}>
          <Text style={styles.bannerTitle}>⚠️ Partial Render</Text>
          <Text style={styles.bannerBody}>
            {Math.round(renderCoverage * 100)}% of regions rendered. Some text could not be drawn on device.
          </Text>
        </View>
      )}

      {/* Expandable Timings */}
      {showTimings && timing?.length > 0 && (
        <View style={styles.timingCard}>
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
      )}

      {/* Actions */}
      <View style={styles.actionColumn}>
        {FEATURES.TRANSLATION_ENABLED && onTranslate && targetLang && onTargetLangChange && (
          <View style={styles.actionRow}>
            <Pressable 
              style={styles.langSelectorBtn} 
              onPress={() => setShowLangPicker(true)}
            >
              <Text style={styles.langSelectorText}>
                {AVAILABLE_LANGUAGES.find(l => l.code === targetLang)?.name || 'Language'} ▼
              </Text>
            </Pressable>
            <ActionButton 
              id="btn-translate" 
              icon={isTranslating ? "⏳" : "🌐"} 
              label={isTranslating ? "Translating..." : "Translate"} 
              onPress={onTranslate} 
              primary 
            />
          </View>
        )}
        <View style={styles.actionRow}>
          <ActionButton id="btn-save" icon="💾" label="Save" onPress={onSave} primary={!FEATURES.TRANSLATION_ENABLED} />
          <ActionButton id="btn-retry" icon="✕" label="Close" onPress={onRetry} />
        </View>
      </View>

      {/* Language Picker Modal */}
      <Modal visible={showLangPicker} transparent animationType="fade">
        <Pressable style={styles.modalBackdrop} onPress={() => setShowLangPicker(false)}>
          <View style={styles.pickerContainer}>
            <Text style={styles.pickerTitle}>Select Language</Text>
            <ScrollView style={styles.pickerScroll}>
              {AVAILABLE_LANGUAGES.map(lang => (
                <TouchableOpacity 
                  key={lang.code} 
                  style={[styles.pickerItem, targetLang === lang.code && styles.pickerItemActive]}
                  onPress={() => {
                    onTargetLangChange?.(lang.code);
                    setShowLangPicker(false);
                  }}
                >
                  <Text style={[styles.pickerItemText, targetLang === lang.code && styles.pickerItemTextActive]}>
                    {lang.name}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
            <TouchableOpacity style={styles.pickerCloseBtn} onPress={() => setShowLangPicker(false)}>
              <Text style={styles.pickerCloseText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </Pressable>
      </Modal>

      {/* ── Manual Compute Override Row ────────────────────────────────────────
         Easy to remove: delete this entire block + overrideRow / overrideBtn styles.
         Shown when: result is already available, allowing user to switch compute path. */}
      <View style={styles.overrideRow}>
        {backend?.used ? (
          // Result came from PC server → offer mobile re-run
          <Pressable
            id="btn-force-mobile"
            style={styles.overrideBtn}
            onPress={onForceMobile}
          >
            <Text style={styles.overrideBtnText}>📱 Recompute Locally (Mobile)</Text>
          </Pressable>
        ) : (
          // Result came from mobile → offer PC server re-run
          <Pressable
            id="btn-force-server"
            style={styles.overrideBtn}
            onPress={onForceServer}
          >
            <Text style={styles.overrideBtnText}>⚡ Enhance on Server (PC)</Text>
          </Pressable>
        )}
      </View>
      {/* ── End Manual Compute Override Row (Inpainting) ────────────────── */}

      {/* ── Manual Compute Override Row (Translation) ────────────────────────
         Shown when translation is already available, allowing user to switch compute path. */}
      {FEATURES.TRANSLATION_ENABLED && translationSession && (
        <View style={styles.overrideRow}>
          {translationSession.path === 'pc' ? (
            // Result came from PC server → offer to prefer mobile
            <Pressable
              id="btn-force-translate-mobile"
              style={styles.overrideBtn}
              onPress={onForceTranslateMobile}
            >
              <Text style={styles.overrideBtnText}>📱 Prefer Mobile Translation</Text>
            </Pressable>
          ) : (
            // Result came from mobile → offer to prefer PC server
            <Pressable
              id="btn-force-translate-server"
              style={styles.overrideBtn}
              onPress={onForceTranslateServer}
            >
              <Text style={styles.overrideBtnText}>⚡ Prefer PC Translation</Text>
            </Pressable>
          )}
        </View>
      )}
      {/* ── End Manual Compute Override Row (Translation) ────────────────── */}
    </View>
  );
}

const styles = StyleSheet.create({
  sheet: {
    backgroundColor: Theme.colors.surface,
    borderTopLeftRadius: Theme.radius.lg,
    borderTopRightRadius: Theme.radius.lg,
    padding: Theme.spacing.lg,
    paddingBottom: Theme.spacing.lg,
    width: '100%',
    ...Theme.shadows.lg,
  },
  dragHandle: {
    width: 40,
    height: 4,
    backgroundColor: Theme.colors.border,
    borderRadius: 2,
    alignSelf: 'center',
    marginBottom: Theme.spacing.md,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: Theme.spacing.lg,
  },
  badge: {
    paddingHorizontal: Theme.spacing.sm,
    paddingVertical: Theme.spacing.xs,
    borderRadius: Theme.radius.md,
    backgroundColor: Theme.colors.surfaceAlt,
  },
  badgeGood: {
    backgroundColor: Theme.colors.successBg,
  },
  badgeWarn: {
    backgroundColor: Theme.colors.warningBg,
  },
  badgeText: {
    ...Theme.typography.button,
    color: Theme.colors.text,
  },
  metaText: {
    ...Theme.typography.caption,
    textDecorationLine: 'underline',
  },
  bannerSuccess: {
    backgroundColor: Theme.colors.successBg,
    borderRadius: Theme.radius.sm,
    padding: Theme.spacing.md,
    marginBottom: Theme.spacing.lg,
  },
  bannerHint: {
    backgroundColor: 'rgba(26,115,232,0.08)',
    borderRadius: Theme.radius.sm,
    padding: Theme.spacing.md,
    marginBottom: Theme.spacing.lg,
    borderWidth: 1,
    borderColor: 'rgba(26,115,232,0.2)',
  },
  bannerTitle: {
    ...Theme.typography.subtitle,
    color: Theme.colors.success,
    marginBottom: Theme.spacing.xs,
  },
  bannerTitleHint: {
    ...Theme.typography.subtitle,
    color: Theme.colors.primary,
    marginBottom: Theme.spacing.xs,
  },
  bannerBody: {
    ...Theme.typography.bodySecondary,
    fontSize: 13,
  },
  actionColumn: {
    gap: Theme.spacing.md,
  },
  actionRow: {
    flexDirection: 'row',
    gap: Theme.spacing.md,
  },
  timingCard: {
    backgroundColor: Theme.colors.background,
    borderRadius: Theme.radius.sm,
    padding: Theme.spacing.md,
    marginBottom: Theme.spacing.lg,
  },
  timingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Theme.spacing.xs,
  },
  timingStage: {
    width: 100,
    ...Theme.typography.caption,
  },
  timingBarWrap: {
    flex: 1,
    height: 4,
    backgroundColor: Theme.colors.border,
    borderRadius: 2,
    marginHorizontal: Theme.spacing.sm,
  },
  timingBar: {
    height: '100%',
    backgroundColor: Theme.colors.primary,
    borderRadius: 2,
  },
  timingMs: {
    width: 45,
    ...Theme.typography.caption,
    textAlign: 'right',
  },
  // ── Manual Compute Override Row styles (easy to remove together with the UI block above) ──
  overrideRow: {
    marginTop: Theme.spacing.sm,
    alignItems: 'center',
  },
  overrideBtn: {
    paddingVertical: Theme.spacing.sm,
    paddingHorizontal: Theme.spacing.md,
  },
  overrideBtnText: {
    ...Theme.typography.caption,
    color: Theme.colors.primary,
    textDecorationLine: 'underline',
  },
  langSelectorBtn: {
    backgroundColor: Theme.colors.surfaceAlt,
    paddingHorizontal: Theme.spacing.md,
    paddingVertical: 14,
    borderRadius: Theme.radius.round,
    justifyContent: 'center',
    alignItems: 'center',
    flex: 1,
  },
  langSelectorText: {
    ...Theme.typography.button,
    color: Theme.colors.text,
  },
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  pickerContainer: {
    width: '80%',
    maxHeight: '60%',
    backgroundColor: Theme.colors.surface,
    borderRadius: Theme.radius.md,
    padding: Theme.spacing.lg,
    ...Theme.shadows.md,
  },
  pickerTitle: {
    ...Theme.typography.subtitle,
    marginBottom: Theme.spacing.md,
  },
  pickerScroll: {
    flexGrow: 0,
  },
  pickerItem: {
    paddingVertical: Theme.spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Theme.colors.surfaceAlt,
  },
  pickerItemActive: {
    backgroundColor: Theme.colors.primary + '20',
  },
  pickerItemText: {
    ...Theme.typography.body,
    color: Theme.colors.text,
  },
  pickerItemTextActive: {
    color: Theme.colors.primary,
    fontWeight: '700',
  },
  pickerCloseBtn: {
    marginTop: Theme.spacing.lg,
    alignItems: 'center',
  },
  pickerCloseText: {
    ...Theme.typography.button,
    color: Theme.colors.primary,
  },
  // ── Translation Status Banners ──────────────────────────────────────────────
  bannerTranslating: {
    backgroundColor: '#E8F0FE',
    borderRadius: Theme.radius.sm,
    padding: Theme.spacing.md,
    marginBottom: Theme.spacing.sm,
    borderLeftWidth: 3,
    borderLeftColor: Theme.colors.primary,
  },
  bannerTranslateSuccess: {
    backgroundColor: Theme.colors.successBg,
    borderRadius: Theme.radius.sm,
    padding: Theme.spacing.md,
    marginBottom: Theme.spacing.sm,
    borderLeftWidth: 3,
    borderLeftColor: Theme.colors.success,
  },
  bannerTranslatePc: {
    backgroundColor: '#FEF7E0',
    borderRadius: Theme.radius.sm,
    padding: Theme.spacing.md,
    marginBottom: Theme.spacing.sm,
    borderLeftWidth: 3,
    borderLeftColor: Theme.colors.warning,
  },
  bannerTranslateFail: {
    backgroundColor: Theme.colors.errorBg,
    borderRadius: Theme.radius.sm,
    padding: Theme.spacing.md,
    marginBottom: Theme.spacing.sm,
    borderLeftWidth: 3,
    borderLeftColor: Theme.colors.error,
  },
  bannerWarnRender: {
    backgroundColor: Theme.colors.warningBg,
    borderRadius: Theme.radius.sm,
    padding: Theme.spacing.md,
    marginBottom: Theme.spacing.sm,
    borderLeftWidth: 3,
    borderLeftColor: Theme.colors.warning,
  },
  sourceLangContainer: {
    marginTop: -Theme.spacing.xs,
    marginBottom: Theme.spacing.sm,
    paddingHorizontal: Theme.spacing.sm,
  },
  sourceLangText: {
    ...Theme.typography.caption,
    color: Theme.colors.textSecondary,
    fontStyle: 'italic',
  },
});

