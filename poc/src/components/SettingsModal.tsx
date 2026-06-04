import React, { useState } from 'react';
import { View, Text, TextInput, Modal, Pressable, StyleSheet, Keyboard } from 'react-native';
import { Theme } from './Theme';
import { ActionButton } from './FloatingActionBar';

interface SettingsModalProps {
  visible: boolean;
  manualUrl: string;
  onClose: () => void;
  onChangeUrl: (url: string) => void;
  onSave: () => void;
  onTest: () => void;
}

export function SettingsModal({
  visible,
  manualUrl,
  onClose,
  onChangeUrl,
  onSave,
  onTest,
}: SettingsModalProps) {
  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <Pressable style={styles.backdrop} onPress={onClose}>
        <Pressable style={styles.card} onPress={Keyboard.dismiss}>
          <Text style={styles.title}>Backend Settings</Text>
          <Text style={styles.desc}>
            The app automatically discovers the server via mDNS, or uses http://10.0.2.2:8000 on the Android emulator. Enter a manual URL to override this.
          </Text>

          <TextInput
            id="input-backend-url"
            style={styles.input}
            value={manualUrl}
            onChangeText={onChangeUrl}
            placeholder="http://192.168.1.X:8000"
            placeholderTextColor={Theme.colors.textHint}
            autoCapitalize="none"
            autoCorrect={false}
            keyboardType="url"
            selectionColor={Theme.colors.primary}
          />

          <View style={styles.btnRow}>
            <ActionButton id="btn-test-url" icon="📶" label="Test" onPress={onTest} />
            <ActionButton id="btn-save-url" icon="💾" label="Save" onPress={onSave} primary />
          </View>

          <Text style={styles.hint}>
            Leave blank to use auto-discovery.
          </Text>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: Theme.colors.overlay,
    justifyContent: 'center',
    padding: Theme.spacing.lg,
  },
  card: {
    backgroundColor: Theme.colors.surface,
    borderRadius: Theme.radius.lg,
    padding: Theme.spacing.xl,
    ...Theme.shadows.lg,
  },
  title: {
    ...Theme.typography.h2,
    marginBottom: Theme.spacing.sm,
  },
  desc: {
    ...Theme.typography.bodySecondary,
    marginBottom: Theme.spacing.lg,
    lineHeight: 20,
  },
  input: {
    backgroundColor: Theme.colors.background,
    borderWidth: 1,
    borderColor: Theme.colors.border,
    borderRadius: Theme.radius.sm,
    paddingHorizontal: Theme.spacing.md,
    paddingVertical: Theme.spacing.md,
    color: Theme.colors.text,
    fontSize: 16,
    fontFamily: 'monospace',
    marginBottom: Theme.spacing.lg,
  },
  btnRow: {
    flexDirection: 'row',
    gap: Theme.spacing.md,
    marginBottom: Theme.spacing.md,
  },
  hint: {
    ...Theme.typography.caption,
    textAlign: 'center',
  },
});
