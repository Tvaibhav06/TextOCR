import sys, os, io, cv2, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-backend'))
from pipeline.core import TextRemovalPipeline

SOURCE_IMG = r"C:\Users\Vaibhav Tandon\.gemini\antigravity\brain\19e6628f-ef59-429b-ac46-61020424fa08\poc_real_background_1776591286752.png"
OUT_DIR = os.path.join(os.path.dirname(__file__), 'poc_output', 'video')
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("Loading background for video generation...")
    img_clean = cv2.imread(SOURCE_IMG)
    if img_clean is None:
        return
    img_clean = cv2.resize(img_clean, (640, 360)) # Lower res for fast video rendering
    
    h, w, _ = img_clean.shape
    
    out_orig_path = os.path.join(OUT_DIR, 'original_with_text.mp4')
    out_clean_path = os.path.join(OUT_DIR, 'cleaned_output.mp4')
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    duration = 3 # 3 seconds
    total_frames = fps * duration
    
    writer_orig = cv2.VideoWriter(out_orig_path, fourcc, fps, (w, h))
    writer_clean = cv2.VideoWriter(out_clean_path, fourcc, fps, (w, h))
    
    pipeline = TextRemovalPipeline()
    
    print(f"Generating and processing {total_frames} frames of video...")
    t_start = time.perf_counter()
    
    for i in range(total_frames):
        frame = img_clean.copy()
        
        # Animate moving text across the screen
        x_pos = int((i / total_frames) * w) - 50
        y_pos = int(h / 2) + int(np.sin(i * 0.2) * 50)
        
        # Add dynamic moving watermark
        cv2.putText(frame, "MOVING WATERMARK", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
        # Add a static timestamp that updates
        cv2.putText(frame, f"REC 00:00:0{i//30}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        writer_orig.write(frame)
        
        # Process frame
        result = pipeline.run(
            image=frame,
            threshold=0.5,
            dilation_px=4,
            inpaint_method='telea',
            inpaint_radius=5
        )
        
        writer_clean.write(result.output)
        
        if (i+1) % 15 == 0:
            print(f"Processed frame {i+1}/{total_frames}...")
            
    writer_orig.release()
    writer_clean.release()
    print(f"Video synthesis complete in {time.perf_counter() - t_start:.1f}s")
    print(f"Saved: {out_orig_path}")
    print(f"Saved: {out_clean_path}")

if __name__ == "__main__":
    main()
