#!/bin/bash

INPUT_DIR="/home/zn23/nnUNet/ddMRI_raw/auto_landmark_training/all_nifti_cropds"
OUTPUT_DIR="/home/zn23/nnUNet/ddMRI_raw/auto_landmark_training/all_nifti_cropds_norm"
SCRIPT_PATH="/home/zn23/nnUNet/nnunetv2/Support_Script/normalize_intensity_robust.py"

mkdir -p "${OUTPUT_DIR}"

for f in "${INPUT_DIR}"/*.nii.gz; do
    base=$(basename "$f")
    out="${OUTPUT_DIR}/${base}"

    echo "[INFO] Normalizing ${base}"
    python "$SCRIPT_PATH" \
        "$f" \
        "$out" \
        --plow 0.5 \
        --phigh 99.5
done

echo "[OK] All files normalized."
