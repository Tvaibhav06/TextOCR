/**
 * useTextRemoval — React hook
 *
 * Manages the full text-removal pipeline lifecycle:
 *  - Image selection (expo-image-picker / react-native-image-picker)
 *  - Pipeline execution with progress tracking
 *  - Result state + error handling
 *  - Cancellation on component unmount
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import * as ImagePicker from 'expo-image-picker';
import * as ImageManipulator from 'expo-image-manipulator';
import * as MediaLibrary from 'expo-media-library';

import { pipeline, CancellationToken, type PipelineResult, type PipelineProgress, type PipelineStage } from '../pipeline/Pipeline';

// ─── State Shape ──────────────────────────────────────────────────────────────

export interface TextRemovalState {
  status:       'idle' | 'picking' | 'processing' | 'done' | 'error';
  stage:        PipelineStage;
  progress:     number;
  progressMsg:  string;
  sourceUri:    string | null;
  resultUri:    string | null;
  result:       PipelineResult | null;
  error:        string | null;
}

export interface TextRemovalActions {
  pickImage:   () => Promise<void>;
  pickCamera:  () => Promise<void>;
  process:     (uri: string, width: number, height: number) => Promise<void>;
  cancel:      () => void;
  saveResult:  () => Promise<boolean>;
  reset:       () => void;
}

const INITIAL_STATE: TextRemovalState = {
  status: 'idle', stage: 'idle', progress: 0, progressMsg: '',
  sourceUri: null, resultUri: null, result: null, error: null,
};

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useTextRemoval(): [TextRemovalState, TextRemovalActions] {
  const [state, setState] = useState<TextRemovalState>(INITIAL_STATE);
  const tokenRef = useRef<CancellationToken | null>(null);

  // Warm up models on mount
  useEffect(() => {
    pipeline.warmUp().catch(console.warn);
    return () => tokenRef.current?.cancel();
  }, []);

  // ── Image picker helpers ──

  const requestPermissions = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    return status === 'granted';
  };

  const preprocessImage = async (uri: string, width: number, height: number) => {
    // Downscale very large images to max 2048px on longest side for mobile perf
    const maxDim = 2048;
    if (Math.max(width, height) <= maxDim) return { uri, width, height };

    const scale  = maxDim / Math.max(width, height);
    const newW   = Math.round(width  * scale);
    const newH   = Math.round(height * scale);

    const resized = await ImageManipulator.manipulateAsync(
      uri,
      [{ resize: { width: newW, height: newH } }],
      { compress: 0.92, format: ImageManipulator.SaveFormat.JPEG },
    );
    return { uri: resized.uri, width: newW, height: newH };
  };

  const pickImage = useCallback(async () => {
    const granted = await requestPermissions();
    if (!granted) {
      setState(s => ({ ...s, error: 'Photo library permission denied' }));
      return;
    }
    setState(s => ({ ...s, status: 'picking' }));

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality:    1,
      allowsEditing: false,
    });

    if (result.canceled || !result.assets[0]) {
      setState(s => ({ ...s, status: 'idle' }));
      return;
    }

    const asset  = result.assets[0];
    const prepped = await preprocessImage(asset.uri, asset.width!, asset.height!);

    setState(s => ({ ...s, sourceUri: prepped.uri, status: 'idle' }));
    await process(prepped.uri, prepped.width, prepped.height);
  }, []);

  const pickCamera = useCallback(async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      setState(s => ({ ...s, error: 'Camera permission denied' }));
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ quality: 0.9 });
    if (result.canceled || !result.assets[0]) return;

    const asset  = result.assets[0];
    const prepped = await preprocessImage(asset.uri, asset.width!, asset.height!);

    setState(s => ({ ...s, sourceUri: prepped.uri }));
    await process(prepped.uri, prepped.width, prepped.height);
  }, []);

  // ── Pipeline runner ──

  const process = useCallback(async (uri: string, width: number, height: number) => {
    tokenRef.current?.cancel();
    const token = new CancellationToken();
    tokenRef.current = token;

    setState(s => ({ ...s, status: 'processing', stage: 'detecting', progress: 0, error: null, resultUri: null }));

    const onProgress = (p: PipelineProgress) => {
      if (token.isCancelled) return;
      setState(s => ({ ...s, stage: p.stage, progress: p.progress, progressMsg: p.message }));
    };

    try {
      const result = await pipeline.run({ imageUri: uri, imageWidth: width, imageHeight: height }, onProgress, token);
      if (token.isCancelled) return;

      setState(s => ({
        ...s,
        status:    'done',
        stage:     'done',
        progress:  100,
        resultUri: result.outputUri,
        result,
      }));
    } catch (err: any) {
      if (token.isCancelled) return;
      setState(s => ({
        ...s, status: 'error', stage: 'error', error: err.message ?? 'Unknown error',
      }));
    }
  }, []);

  const cancel = useCallback(() => {
    tokenRef.current?.cancel();
    setState(s => ({ ...s, status: 'idle', stage: 'idle', progress: 0 }));
  }, []);

  const saveResult = useCallback(async (): Promise<boolean> => {
    if (!state.resultUri) return false;
    const { status } = await MediaLibrary.requestPermissionsAsync();
    if (status !== 'granted') return false;
    await MediaLibrary.saveToLibraryAsync(state.resultUri);
    return true;
  }, [state.resultUri]);

  const reset = useCallback(() => {
    tokenRef.current?.cancel();
    setState(INITIAL_STATE);
  }, []);

  const actions: TextRemovalActions = { pickImage, pickCamera, process, cancel, saveResult, reset };
  return [state, actions];
}
