import { TranslationRouter } from '../../src/services/TranslationRouter';
import { backendService } from '../../src/services/BackendService';
import { mobileTranslator } from '../../src/pipeline/translation/MobileTranslate';
import * as MobileOCR from '../../src/pipeline/translation/MobileOCR';

// Mock dependencies
jest.mock('../../src/services/BackendService', () => ({
  backendService: {
    resolveUrl: jest.fn(),
    processTranslate: jest.fn(),
  },
}));

jest.mock('../../src/pipeline/translation/MobileTranslate', () => ({
  mobileTranslator: {
    canTranslate: jest.fn(),
    translateAll: jest.fn(),
  },
}));

jest.mock('../../src/pipeline/translation/MobileOCR', () => ({
  extractTextFromRegions: jest.fn(),
}));

describe('TranslationRouter Bidirectional Fallback', () => {
  let router: TranslationRouter;
  
  const mockBaseResult = {
    originalUri: 'file://orig.jpg',
    outputUri: 'file://inpainted.jpg',
    width: 1000,
    height: 1000,
    detections: [
      { x: 0, y: 0, w: 100, h: 50 },
      { x: 0, y: 100, w: 100, h: 50 },
    ],
  };

  beforeEach(() => {
    jest.clearAllMocks();
    router = new TranslationRouter();

    // Default mobile success setup
    (mobileTranslator.canTranslate as jest.Mock).mockResolvedValue(true);
    (MobileOCR.extractTextFromRegions as jest.Mock).mockResolvedValue([
      { text: 'Hello', confidence: 0.9, textQuality: 0.9, skip: false, originalRegion: mockBaseResult.detections[0] },
      { text: 'World', confidence: 0.9, textQuality: 0.9, skip: false, originalRegion: mockBaseResult.detections[1] },
    ]);
    (mobileTranslator.translateAll as jest.Mock).mockResolvedValue([
      { engine: 'mlkit', text: 'Hola', durationMs: 10 },
      { engine: 'mlkit', text: 'Mundo', durationMs: 10 },
    ]);

    // Default PC success setup
    (backendService.resolveUrl as jest.Mock).mockResolvedValue('http://localhost:8000');
    (backendService.processTranslate as jest.Mock).mockResolvedValue({
      resultUri: 'file://pc-translated.jpg',
      translations: 2,
      detections: 2,
      totalMs: 1500,
    });
  });

  // 1. Mobile success
  it('1. Mobile success: uses mobile path when preferredPath is mobile', async () => {
    const res = await router.run(mockBaseResult as any, 'es', undefined, 'mobile');
    expect(res.path).toBe('mobile');
    expect(backendService.resolveUrl).not.toHaveBeenCalled();
  });

  // 2. Mobile fail -> PC success
  it('2. Mobile fail -> PC success: falls back to PC when mobile fails entirely', async () => {
    (mobileTranslator.canTranslate as jest.Mock).mockResolvedValue(false); // Mobile unavailable
    
    const res = await router.run(mockBaseResult as any, 'es', undefined, 'mobile');
    
    expect(res.path).toBe('pc');
    expect(mobileTranslator.canTranslate).toHaveBeenCalled();
    expect(backendService.processTranslate).toHaveBeenCalled();
  });

  // 3. Mobile fail -> PC fail
  it('3. Mobile fail -> PC fail: returns failed path if both fail when preferring mobile', async () => {
    (mobileTranslator.canTranslate as jest.Mock).mockResolvedValue(false);
    (backendService.processTranslate as jest.Mock).mockResolvedValue(null); // PC fails
    
    const res = await router.run(mockBaseResult as any, 'es', undefined, 'mobile');
    
    expect(res.path).toBe('failed');
    expect(res.fallbackReason).toBe('pc_translate_error');
  });

  // 4. PC success
  it('4. PC success: uses PC path when preferredPath is pc', async () => {
    const res = await router.run(mockBaseResult as any, 'es', undefined, 'pc');
    expect(res.path).toBe('pc');
    expect(backendService.processTranslate).toHaveBeenCalled();
    expect(MobileOCR.extractTextFromRegions).not.toHaveBeenCalled();
  });

  // 5. PC fail -> Mobile success
  it('5. PC fail -> Mobile success: falls back to mobile when PC fails', async () => {
    (backendService.processTranslate as jest.Mock).mockResolvedValue(null); // PC fails
    
    const res = await router.run(mockBaseResult as any, 'es', undefined, 'pc');
    
    expect(res.path).toBe('mobile');
    expect(backendService.processTranslate).toHaveBeenCalled();
    expect(MobileOCR.extractTextFromRegions).toHaveBeenCalled();
  });

  // 6. PC fail -> Mobile fail
  it('6. PC fail -> Mobile fail: returns original PC failure if mobile fallback also fails', async () => {
    (backendService.processTranslate as jest.Mock).mockResolvedValue(null); // PC fails
    (mobileTranslator.canTranslate as jest.Mock).mockResolvedValue(false); // Mobile fallback also fails
    
    const res = await router.run(mockBaseResult as any, 'es', undefined, 'pc');
    
    // Should return the PC failure state, not the mobile failure state, because PC was preferred
    expect(res.path).toBe('failed');
    expect(res.fallbackReason).toBe('pc_translate_error');
  });

  // 7. Quality-gate-triggered PC escalation
  it('7. Quality-gate-triggered PC escalation: falls back to PC if mobile OCR confidence is too low', async () => {
    (MobileOCR.extractTextFromRegions as jest.Mock).mockResolvedValue([
      { text: 'Blurry', confidence: 0.2, textQuality: 0.2, skip: false, originalRegion: mockBaseResult.detections[0] },
      { text: 'Mess', confidence: 0.3, textQuality: 0.3, skip: false, originalRegion: mockBaseResult.detections[1] },
    ]);
    
    const res = await router.run(mockBaseResult as any, 'es', undefined, 'mobile');
    
    expect(res.path).toBe('pc'); // Escalated to PC
    expect(backendService.processTranslate).toHaveBeenCalled();
  });

  // 8. No unnecessary escalation when mobile output is acceptable
  it('8. No unnecessary escalation: stays on mobile if confidence is acceptable', async () => {
    (MobileOCR.extractTextFromRegions as jest.Mock).mockResolvedValue([
      { text: 'Okay', confidence: 0.65, textQuality: 0.65, skip: false, originalRegion: mockBaseResult.detections[0] },
      { text: 'Text', confidence: 0.70, textQuality: 0.70, skip: false, originalRegion: mockBaseResult.detections[1] },
    ]); // Averages 0.675 > MIN_OCR_CONFIDENCE (0.60)
    
    const res = await router.run(mockBaseResult as any, 'es', undefined, 'mobile');
    
    expect(res.path).toBe('mobile');
    expect(backendService.processTranslate).not.toHaveBeenCalled();
  });
});
