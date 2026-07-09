import React from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Theme } from './Theme';

interface ActionButtonProps {
  id: string;
  icon: string;
  label: string;
  onPress: () => void;
  primary?: boolean;
  isCircular?: boolean;
  /** When true, the button is visually dimmed and onPress is suppressed. */
  disabled?: boolean;
}

export function ActionButton({ id, icon, label, onPress, primary, isCircular, disabled }: ActionButtonProps) {
  return (
    <Pressable
      id={id}
      onPress={disabled ? undefined : onPress}
      style={({ pressed }) => [
        styles.btn,
        primary ? styles.btnPrimary : styles.btnSecondary,
        isCircular ? styles.btnCircular : styles.btnPill,
        pressed && !disabled && styles.btnPressed,
        disabled && styles.btnDisabled,
      ]}
    >
      <Text style={[styles.icon, primary && styles.iconPrimary, isCircular && styles.iconCircular]}>{icon}</Text>
      {!isCircular && (
        <Text style={[styles.label, primary && styles.labelPrimary]}>{label}</Text>
      )}
    </Pressable>
  );
}

interface FloatingActionBarProps {
  children: React.ReactNode;
}

export function FloatingActionBar({ children }: FloatingActionBarProps) {
  return (
    <View style={styles.container}>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingBottom: Theme.spacing.xxl, // Safe area padding
    paddingTop: Theme.spacing.lg,
    paddingHorizontal: Theme.spacing.lg,
    gap: Theme.spacing.lg,
  },
  btn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    ...Theme.shadows.md,
  },
  btnCircular: {
    width: 56,
    height: 56,
    borderRadius: 28,
  },
  btnPill: {
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: Theme.radius.round,
    flex: 1, // Let it stretch if it's the only one, or share space
    maxWidth: 250,
  },
  btnPrimary: {
    backgroundColor: Theme.colors.primary,
  },
  btnSecondary: {
    backgroundColor: Theme.colors.surface,
    borderWidth: 1,
    borderColor: Theme.colors.border,
  },
  btnPressed: {
    opacity: 0.85,
    transform: [{ scale: 0.98 }],
  },
  btnDisabled: {
    opacity: 0.45,
  },
  icon: {
    fontSize: 20,
    marginRight: 8,
  },
  iconCircular: {
    marginRight: 0,
    fontSize: 24,
  },
  iconPrimary: {
    color: Theme.colors.textInverse, // for emoji icons, color might not apply, but good for icon fonts
  },
  label: {
    ...Theme.typography.button,
    color: Theme.colors.primary,
  },
  labelPrimary: {
    color: Theme.colors.textInverse,
  },
});
