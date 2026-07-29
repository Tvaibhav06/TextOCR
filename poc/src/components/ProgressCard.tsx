import React from 'react';
import { View, Text, StyleSheet, Animated, ActivityIndicator } from 'react-native';
import { Theme } from './Theme';
import { PipelineStage } from '../pipeline/Pipeline';

interface ProgressCardProps {
  stage: PipelineStage;
  stageMsg: string;
  progressAnim: Animated.Value;
  progressPct: number;
}

const STAGES = [
  { key: 'detecting', label: 'Detect', uploadStage: false },
  { key: 'masking', label: 'Mask', uploadStage: false },
  { key: 'inpainting', label: 'Erase', uploadStage: false },
  { key: 'postprocessing', label: 'Blend', uploadStage: false },
  { key: 'uploading', label: 'Server', uploadStage: true },
] as const;

export function ProgressCard({ stage, stageMsg, progressAnim, progressPct }: ProgressCardProps) {
  const stageOrder: Record<PipelineStage, number> = {
    idle: 0, detecting: 1, masking: 2, inpainting: 3,
    postprocessing: 4, uploading: 5, done: 6, error: 0,
  };
  const current = stageOrder[stage] ?? 0;

  return (
    <View style={styles.card}>
      <Text style={styles.stageMsg}>{stageMsg}</Text>

      {/* Progress bar */}
      <View style={styles.progressTrack}>
        <Animated.View
          style={[
            styles.progressFill,
            stage === 'uploading' && styles.progressFillUpload,
            {
              width: progressAnim.interpolate({
                inputRange: [0, 100],
                outputRange: ['0%', '100%'],
              }),
            },
          ]}
        />
      </View>
      <Text style={styles.progressPct}>{Math.round(progressPct)}%</Text>

      {/* Stage dots */}
      <View style={styles.stageDots}>
        {STAGES.map((item, i) => {
          const thisIdx = i + 1;
          const done = current > thisIdx;
          const active = current === thisIdx;

          return (
            <View key={item.key} style={styles.stageDotItem}>
              <View
                style={[
                  styles.stageDot,
                  active && styles.stageDotActive,
                  done && styles.stageDotDone,
                  item.uploadStage && (active || done) && styles.stageDotUpload,
                ]}
              >
                {active && <ActivityIndicator size="small" color={Theme.colors.surface} />}
                {done && <Text style={styles.stageDotCheck}>✓</Text>}
                {!active && !done && <Text style={styles.stageDotNum}>{i + 1}</Text>}
              </View>
              <Text
                style={[
                  styles.stageDotLabel,
                  (active || done) && styles.stageDotLabelActive,
                  item.uploadStage && (active || done) && styles.stageDotLabelUpload,
                ]}
                numberOfLines={1}
              >
                {item.label}
              </Text>
            </View>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: Theme.colors.surface,
    borderRadius: Theme.radius.lg,
    padding: Theme.spacing.lg,
    width: '100%',
    ...Theme.shadows.md,
  },
  stageMsg: {
    ...Theme.typography.h2,
    color: Theme.colors.primary,
    marginBottom: Theme.spacing.lg,
    textAlign: 'center',
  },
  progressTrack: {
    height: 8,
    backgroundColor: Theme.colors.surfaceAlt,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: Theme.colors.primary,
    borderRadius: 4,
  },
  progressFillUpload: {
    backgroundColor: Theme.colors.warning,
  },
  progressPct: {
    ...Theme.typography.caption,
    textAlign: 'right',
    marginTop: Theme.spacing.xs,
    color: Theme.colors.textSecondary,
  },
  stageDots: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: Theme.spacing.lg,
  },
  stageDotItem: {
    alignItems: 'center',
    flex: 1,
  },
  stageDot: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: Theme.colors.surfaceAlt,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Theme.spacing.xs,
  },
  stageDotActive: {
    backgroundColor: Theme.colors.primary,
  },
  stageDotDone: {
    backgroundColor: Theme.colors.primary,
  },
  stageDotUpload: {
    backgroundColor: Theme.colors.warning,
  },
  stageDotNum: {
    ...Theme.typography.caption,
    color: Theme.colors.textHint,
    fontWeight: '600',
  },
  stageDotCheck: {
    fontSize: 14,
    color: Theme.colors.surface,
    fontWeight: '700',
  },
  stageDotLabel: {
    ...Theme.typography.caption,
    color: Theme.colors.textHint,
  },
  stageDotLabelActive: {
    color: Theme.colors.primary,
    fontWeight: '600',
  },
  stageDotLabelUpload: {
    color: Theme.colors.warning,
  },
});
