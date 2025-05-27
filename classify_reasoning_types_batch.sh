# Find all jsonl files in the specified path
input_file="log/MATH-500/inference_batch_remaining.baseline.txt"

while IFS= read -r filepath || [ -n "$filepath" ]; do
    # Create output directory and filename
    segment_dir=$(dirname "$filepath")/segmentation_results

    # Extract filename without path
    filename=$(basename "$filepath")
    segment_file="$segment_dir/$filename"

    # Check if the file exists
    if [ -f "$segment_file" ]; then
        # Count the number of lines in the file
        line_count=$(wc -l < "$segment_file")
        echo "Processing file: $segment_file - Contains $line_count lines"
    else
        echo -e "\033[33mWarning: File '$segment_file' does not exist.\033[0m"
    fi

    python classify_reasoning_types_batch.py \
        --input_jsonl "$segment_file" \
        --gcs_upload_prefix "gs://<Your GCS project>/multilingual-reasoning/cotscope/reasoning_kind_classification/batch_prediction_input" \
        --gcs_download_prefix "gs://<Your GCS project>/multilingual-reasoning/cotscope/reasoning_kind_classification/batch_prediction_output" \
        --save_root_dir $(dirname "$filepath")/reasoning_kind_results \
        --reasoning_kind_def_json "reasoning_kind_definitions/four_habits_of_highly_effective_stars.json" &

done < "$input_file"

wait

echo "All files processed."
