from PIL import Image
import numpy as np
import sys
import os

def process_image(input_path, output_path, M=800):
    
    # Load image and convert to grayscale
    img = Image.open(input_path).convert("L")
    
    # Resize to M x M
    img = img.resize((M, M))
    
    # Convert to numpy array
    pixels = np.array(img, dtype=np.uint32)
    
    # Save as raw binary
    pixels.tofile(output_path)
    print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    process_image(input_path, output_path)