#!/bin/bash

# Export CUDA device
export CUDA_VISIBLE_DEVICES=1

# Find all jsonl files in the specified path
# it should include something like this: (each is one line), you can get this by ls log/*/*/*.jsonl
# log/MATH-500/ko/Qwen3-30B-A3B__thinking_prefill-ప్రారంభించడానికి.jsonl
# log/MATH-500/ko/Qwen3-30B-A3B__thinking_prefill-嗯.jsonl
# log/MATH-500/ko/Qwen3-30B-A3B__thinking_prefill-Primero.jsonl
input_file="log/MATH-500/inference_batch.txt"

while IFS= read -r filepath || [ -n "$filepath" ]; do
    # Check if the file exists
    if [ -f "$filepath" ]; then
        # Count the number of lines in the file
        line_count=$(wc -l < "$filepath")
        echo "Processing file: $filepath - Contains $line_count lines"
    else
        echo -e "\033[33mWarning: File '$filepath' does not exist.\033[0m"
    fi
    
    # Create output directory and filename
    output_dir=$(dirname "$filepath")/segmentation_results
    mkdir -p "$output_dir"
    
    # Extract filename without path
    filename=$(basename "$filepath")
    output_file="$output_dir/$filename"
    
    # # Run the segmentation script
    echo "Saving to $output_file"
    python segment_reasoning.py \
        --model_path "appier-ai-research/reasoning-segmentation-model-v0" \
        --input_jsonl "$filepath" \
        --output_jsonl "$output_file"
done < "$input_file"

echo "All files processed."
