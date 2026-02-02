from PIL import Image
import numpy as np
import sys

def visualize_raw(input_path, output_path, M=800):
    data = np.fromfile(input_path, dtype=np.uint32)
    data = data.reshape((M, M))

    # Normalize to 0–255 for display
    if data.max() > data.min():
        data = (data - data.min()) * 255 / (data.max() - data.min())
    else:
        data = np.zeros_like(data)
    
    data = data.astype(np.uint8)

    Image.fromarray(data).save(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    visualize_raw(input_path, output_path)