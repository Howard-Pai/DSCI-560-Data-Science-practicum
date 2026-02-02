#!/bin/bash

# =========================================
# Automated CUDA Image Processing Pipeline
# =========================================

# Kernel type (default: edge)
KERNEL_TYPE="edge"
if [ $# -ge 1 ]; then
    KERNEL_TYPE="$1"
fi

echo "========================================="
echo "CUDA Image Processing Pipeline"
echo "Kernel: $KERNEL_TYPE"
echo "========================================="

# Directories
ORIGINAL_DIR="./Image_original"
RAW_DIR="./Image_raw"
RESULT_DIR="./Image_result"

# Create directories if missing
for dir in "$ORIGINAL_DIR" "$RAW_DIR" "$RESULT_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

# Count images
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

echo "Found $IMAGE_COUNT image(s)"
echo ""

# -----------------------------------------
# Compile CUDA convolution code
# -----------------------------------------
echo "Compiling CUDA convolution program..."
nvcc -O2 convolution.cu -o convolution_cuda
if [ $? -ne 0 ]; then
    echo "Error: CUDA compilation failed"
    exit 1
fi
echo "CUDA compilation complete!"
echo ""

# -----------------------------------------
# Process each image
# -----------------------------------------
PROCESSED_COUNT=0
FAILED_COUNT=0
CURRENT=0

for INPUT_IMAGE in "$ORIGINAL_DIR"/*.png "$ORIGINAL_DIR"/*.PNG \
                   "$ORIGINAL_DIR"/*.jpg "$ORIGINAL_DIR"/*.JPG \
                   "$ORIGINAL_DIR"/*.jpeg "$ORIGINAL_DIR"/*.JPEG; do

    if [ ! -f "$INPUT_IMAGE" ]; then
        continue
    fi

    CURRENT=$((CURRENT + 1))
    echo "========================================="
    echo "Processing image $CURRENT/$IMAGE_COUNT"
    echo "========================================="
    echo "Input: $INPUT_IMAGE"

    BASENAME=$(basename "$INPUT_IMAGE" | sed 's/\.[^.]*$//')

    RAW_FILE="$RAW_DIR/${BASENAME}.raw"
    OUTPUT_RAW="$RAW_DIR/${BASENAME}_${KERNEL_TYPE}_cuda.raw"
    RESULT_IMAGE="$RESULT_DIR/${BASENAME}_${KERNEL_TYPE}_cuda.jpeg"

    if python3 image_processing.py "$INPUT_IMAGE" "$RAW_FILE" && \
       ./convolution_cuda "$RAW_FILE" "$OUTPUT_RAW" "$KERNEL_TYPE" && \
       python3 visualize_raw_file.py "$OUTPUT_RAW" "$RESULT_IMAGE"; then
        echo "Success: $RESULT_IMAGE"
        PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    else
        echo "Failed: $INPUT_IMAGE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi

    echo ""
done

# -----------------------------------------
# Summary
# -----------------------------------------
echo "========================================="
echo "CUDA Pipeline Summary"
echo "========================================="
echo "Kernel type:            $KERNEL_TYPE"
echo "Total images:           $IMAGE_COUNT"
echo "Successfully processed: $PROCESSED_COUNT"
echo "Failed:                 $FAILED_COUNT"
echo "Results saved in: $RESULT_DIR"
echo "========================================="
