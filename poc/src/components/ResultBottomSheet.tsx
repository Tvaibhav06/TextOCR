import React, { useState } from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { Theme } from './Theme';
import { PipelineResult } from '../pipeline/Pipeline';
import { ActionButton } from './FloatingActionBar';

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
}

export function ResultBottomSheet({ result, onSave, onRetry, onOpenSettings, onForceMobile, onForceServer }: ResultBottomSheetProps) {
  const [showTimings, setShowTimings] = useState(false);

  const { quality, backend, timing, detections } = result;

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

      {/* Server Status Banners */}
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
      <View style={styles.actionRow}>
        <ActionButton id="btn-save" icon="💾" label="Save" onPress={onSave} primary />
        <ActionButton id="btn-retry" icon="✕" label="Close" onPress={onRetry} />
      </View>

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
      {/* ── End Manual Compute Override Row ────────────────────────────────── */}
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
});
