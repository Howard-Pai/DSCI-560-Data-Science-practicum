#!/bin/bash

# Automated Image Processing Pipeline
# This script automates the entire image processing workflow for ALL images

# Specify kernel type (default : edge)
KERNEL_TYPE="edge"
if [ $# -ge 1 ]; then
    KERNEL_TYPE="$1"
fi

echo "========================================="
echo "Image Processing Pipeline"
echo "Kernel: $KERNEL_TYPE"
echo "========================================="

# Define directories
ORIGINAL_DIR="./Image_original"
RAW_DIR="./Image_raw"
RESULT_DIR="./Image_result"

# Check if directories exist
for dir in "$ORIGINAL_DIR" "$RAW_DIR" "$RESULT_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

# Find all image files in Image_original (compatible method)
IMAGE_COUNT=0
for img in "$ORIGINAL_DIR"/*.{png,PNG,jpg,JPG,jpeg,JPEG}; do
    if [ -f "$img" ]; then
        IMAGE_COUNT=$((IMAGE_COUNT + 1))
    fi
done

if [ $IMAGE_COUNT -eq 0 ]; then
    echo "Error: No images found in $ORIGINAL_DIR"
    exit 1
fi

echo "Found $IMAGE_COUNT image(s) to process"
echo ""

# Compile the convolution program once
echo "Compiling convolution program..."
gcc -o convolution_c convolution.c -O2
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed"
    exit 1
fi
echo "Compilation complete!"
echo ""

# Process each image
PROCESSED_COUNT=0
FAILED_COUNT=0
CURRENT=0

for INPUT_IMAGE in "$ORIGINAL_DIR"/*.png "$ORIGINAL_DIR"/*.PNG "$ORIGINAL_DIR"/*.jpg "$ORIGINAL_DIR"/*.JPG "$ORIGINAL_DIR"/*.jpeg "$ORIGINAL_DIR"/*.JPEG; do
    # Skip if file doesn't exist (happens when no files match a pattern)
    if [ ! -f "$INPUT_IMAGE" ]; then
        continue
    fi
    
    CURRENT=$((CURRENT + 1))
    echo "========================================="
    echo "Processing image $CURRENT/$IMAGE_COUNT"
    echo "========================================="
    echo "Input: $INPUT_IMAGE"
    
    BASENAME=$(basename "$INPUT_IMAGE" | sed 's/\.[^.]*$//')
    
    # Define file paths with kernel type in filename
    RAW_FILE="$RAW_DIR/${BASENAME}.raw"
    OUTPUT_RAW="$RAW_DIR/${BASENAME}_${KERNEL_TYPE}.raw"
    RESULT_IMAGE="$RESULT_DIR/${BASENAME}_${KERNEL_TYPE}.jpeg"
    
    # Process with error handling for individual images
    if python3 image_processing.py "$INPUT_IMAGE" "$RAW_FILE" && \
       ./convolution_c "$RAW_FILE" "$OUTPUT_RAW" "$KERNEL_TYPE" && \
       python3 visualize_raw_file.py "$OUTPUT_RAW" "$RESULT_IMAGE"; then
        echo "✓ Success: $RESULT_IMAGE"
        PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    else
        echo "✗ Failed to process: $INPUT_IMAGE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    echo ""
done

echo "========================================="
echo "Pipeline Summary"
echo "========================================="
echo "Kernel type:            $KERNEL_TYPE"
echo "Total images:           $IMAGE_COUNT"
echo "Successfully processed: $PROCESSED_COUNT"
echo "Failed:                 $FAILED_COUNT"
echo ""
echo "Results saved in: $RESULT_DIR"
echo "========================================="