import React from 'react';
import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Theme } from './Theme';

interface TopBarProps {
  title: string;
  onSettingsPress: () => void;
  isDark?: boolean;
}

export function TopBar({ title, onSettingsPress, isDark = false }: TopBarProps) {
  return (
    <View style={[styles.container, isDark && styles.containerDark]}>
      <Text style={[styles.title, isDark && styles.titleDark]}>{title}</Text>
      <Pressable
        id="btn-settings"
        onPress={onSettingsPress}
        style={({ pressed }) => [styles.btn, pressed && styles.btnPressed]}
      >
        <Text style={styles.icon}>⚙️</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Theme.spacing.lg,
    paddingTop: Theme.spacing.lg,
    paddingBottom: Theme.spacing.md,
    backgroundColor: Theme.colors.primary,
  },
  containerDark: {
    backgroundColor: 'transparent',
  },
  title: {
    ...Theme.typography.h2,
    color: Theme.colors.textInverse,
    letterSpacing: 0.5,
  },
  titleDark: {
    color: Theme.colors.text,
  },
  btn: {
    padding: Theme.spacing.sm,
    borderRadius: Theme.radius.round,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
  },
  btnPressed: {
    opacity: 0.7,
  },
  icon: {
    fontSize: 20,
  },
});
