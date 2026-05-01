import argparse
import sys
import os
import cv2

# Ensure the python-backend package can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python-backend'))
from pipeline.core import TextRemovalPipeline

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="TDRS POC - Process an arbitrary image to remove text.")
    parser.add_argument("input_image", help="Path to the input image (e.g., 'my_photo.jpg')")
    parser.add_argument("-o", "--output", default="cleaned_output.png", help="Path to save the output image (default: 'cleaned_output.png')")
    
    args = parser.parse_args()
    
    # 1. Validate file exists
    if not os.path.exists(args.input_image):
        print(f"[Error] Could not find the file: '{args.input_image}'")
        return
        
    print(f"[*] Loading '{args.input_image}'...")
    
    # 2. Load the image using OpenCV
    img = cv2.imread(args.input_image)
    if img is None:
        print("[Error] Failed to read image. Ensure it is a valid format like PNG or JPG.")
        return
        
    print("[*] Processing image through Text Removal Pipeline...")
    
    # 3. Process the image (using the defaults we just tuned!)
    pipeline = TextRemovalPipeline()
    result = pipeline.run(image=img)
    
    # 4. Print quick stats
    print(f"    - Found {len(result.detections)} text regions to remove.")
    print(f"    - Processing completed in {result.total_ms:.1f}ms")
    
    # 5. Save the output
    success = cv2.imwrite(args.output, result.output)
    if success:
        print(f"[*] Success! Cleaned image saved as: '{args.output}'")
    else:
        print(f"[Error] Failed to save the image (check your folder permissions).")

if __name__ == "__main__":
    main()
