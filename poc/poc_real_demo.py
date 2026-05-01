import sys, os, io, cv2, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-backend'))
from pipeline.core import TextRemovalPipeline, QualityAssessor

# The realistic image we just generated
SOURCE_IMG = r"C:\Users\Vaibhav Tandon\.gemini\antigravity\brain\19e6628f-ef59-429b-ac46-61020424fa08\poc_real_background_1776591286752.png"
OUT_DIR = os.path.join(os.path.dirname(__file__), 'poc_output', 'real')
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("Loading generated real-world image...")
    img_clean = cv2.imread(SOURCE_IMG)
    if img_clean is None:
        print(f"Error: Could not load image at {SOURCE_IMG}")
        return
        
    # Resize to 720p for standard processing
    img_clean = cv2.resize(img_clean, (1280, 720))
    
    # Add realistic overlays
    img_with_text = img_clean.copy()
    
    # 1. A semi-transparent watermark in the center
    overlay = img_with_text.copy()
    cv2.putText(overlay, "shutterstock", (300, 360), cv2.FONT_HERSHEY_DUPLEX, 3.5, (255,255,255), 5)
    cv2.addWeighted(overlay, 0.4, img_with_text, 0.6, 0, img_with_text)
    
    # 2. A crisp caption at the bottom
    cv2.putText(img_with_text, "@nature_photos_2026", (400, 680), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # 3. A date stamp at the top right
    cv2.putText(img_with_text, "19/04/2026 15:00", (900, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imwrite(os.path.join(OUT_DIR, '01_original.png'), img_with_text)
    
    print("Running pipeline on real-world image...")
    pipeline = TextRemovalPipeline()
    t_start = time.perf_counter()
    
    # On a real image with complex background, we drop threshold and increase radius slightly
    result = pipeline.run(
        image=img_with_text,
        threshold=0.5,
        dilation_px=5, 
        inpaint_method='telea',
        inpaint_radius=7
    )
    t_ms = (time.perf_counter() - t_start) * 1000
    print(f"Completed in {t_ms:.1f}ms")
    
    mask_vis = cv2.cvtColor(result.mask.data, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(OUT_DIR, '02_mask.png'), mask_vis)
    cv2.imwrite(os.path.join(OUT_DIR, '03_result.png'), result.output)
    
    # Side by side: Original -> Result
    sep = np.full((720, 6, 3), 60, dtype=np.uint8)
    comparison = np.hstack([img_with_text, sep, result.output])
    cv2.imwrite(os.path.join(OUT_DIR, '04_comparison.png'), comparison)
    
    print(f"Images saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
