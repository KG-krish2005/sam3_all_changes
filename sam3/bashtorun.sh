#!/bin/bash

# --- CONFIGURATION ---
INPUT_ROOT="/home/krish/data/2026-01-19_15-04-24-266718.mp4"   # <--- CHANGE THIS
# INPUT_ROOT="/home/krish/abc/sam3/small_testcase/10.1/10.1.mp4"
OUTPUT_ROOT="./outputs_inference_change4_long_video"
FINAL_REPORT="SAM3_Tracking_Report_long_video_logging.csv"
PYTHON_SCRIPT="sam3_piyush.py"         # Name of the python file above

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Initialize the Master CSV with headers (We create a dummy header first)
# We will overwrite this with the first valid CSV found later
echo "clip_name,unique_people_id,start_frame,end_frame,total_frames,fps" > "$FINAL_REPORT"

# Find all mp4 files, sort them naturally (version sort handles 1, 2, 10 correctly)
# Using 'find' handles nested folders recursively
find "$INPUT_ROOT" -type f -name "*.mp4" | sort -V | while read -r VIDEO_PATH; do
    
    FILENAME=$(basename "$VIDEO_PATH")
    DIRNAME=$(dirname "$VIDEO_PATH")
    CLIP_NAME=$(basename "$DIRNAME") # Assumes folder name is clip name (e.g., 1.1)
    
    # If video is in root, use filename without extension as clip name
    if [ "$CLIP_NAME" == "." ] || [ "$CLIP_NAME" == "$INPUT_ROOT" ]; then
        CLIP_NAME="${FILENAME%.*}"
    fi

    echo "=================================================="
    echo "Processing Video: $FILENAME (Clip: $CLIP_NAME)"
    echo "=================================================="

    # Define paths for this specific run
    VIDEO_OUT_DIR="$OUTPUT_ROOT/processed_${FILENAME%.*}"
    # Ensure the directory exists (optional, if you want folders per video)
    # Or just save to root output dir:
    OUTPUT_VIDEO="$OUTPUT_ROOT/processed_$FILENAME"
    FRAGMENT_CSV="$OUTPUT_ROOT/temp_${FILENAME%.*}.csv"

    # Call Python Script
    # The python process starts, uses memory, finishes, and DIES (freeing everything)
    python "$PYTHON_SCRIPT" \
        --input_video "$VIDEO_PATH" \
        --output_video "$OUTPUT_VIDEO" \
        --output_csv_fragment "$FRAGMENT_CSV" \
        --clip_name "$CLIP_NAME"

    # Check if Python succeeded
    if [ $? -eq 0 ]; then
        # If the fragment exists and is not empty
        if [ -s "$FRAGMENT_CSV" ]; then
            # Append data to Master CSV, skipping the header row
            tail -n +2 "$FRAGMENT_CSV" >> "$FINAL_REPORT"
            echo "-> Data merged into $FINAL_REPORT"
            
            # Optional: Delete the fragment to keep folder clean
            rm "$FRAGMENT_CSV"
        else
            echo "-> No objects found in video."
            rm "$FRAGMENT_CSV" 2>/dev/null
        fi
    else
        echo "[ERROR] Python script failed for $FILENAME"
    fi
    
    echo ""
done

echo "Batch processing complete. Full report: $FINAL_REPORT"