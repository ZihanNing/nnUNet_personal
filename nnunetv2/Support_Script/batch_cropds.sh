#!/bin/bash

# =========================
# Batch preprocessing script for dental MRI NIfTI files
# =========================

# ---- Paths (EDIT ONLY IF NEEDED) ----
INPUT_DIR="/home/zn23/nnUNet/ddMRI_raw/auto_landmark_training/all_nifti"
OUTPUT_DIR="/home/zn23/nnUNet/ddMRI_raw/auto_landmark_training/all_nifti_cropds"
SCRIPT_PATH="/home/zn23/nnUNet/nnunetv2/Support_Script/ddMRI_cropds_recover.py"

# ---- Create output directory if it does not exist ----
mkdir -p "${OUTPUT_DIR}"

# ---- Loop over .nii and .nii.gz files ----
for INFILE in "${INPUT_DIR}"/*.nii "${INPUT_DIR}"/*.nii.gz; do

    # Skip if no files match (safety)
    [ -e "$INFILE" ] || continue

    # Extract base filename without extensions
    FILENAME=$(basename "$INFILE")
    BASENAME="${FILENAME%.nii.gz}"
    BASENAME="${BASENAME%.nii}"

    # Define output filename
    OUTFILE="${OUTPUT_DIR}/${BASENAME}.nii.gz"

    echo "============================================"
    echo "Processing: $INFILE"
    echo "Output:     $OUTFILE"
    echo "============================================"

    # ---- Call preprocessing script ----
    python "$SCRIPT_PATH" \
        "$INFILE" \
        "$OUTFILE" \
        --mode generic

done

echo "All files processed."

