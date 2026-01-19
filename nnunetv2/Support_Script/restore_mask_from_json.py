"""
Restore a preprocessed (cropped + downsampled + final-cropped) label mask
back to the original image grid using the saved preprocess JSON log.

Assumptions:
- The input mask is a LABEL mask (0/1/2/...) produced on the preprocessed image.
- Cropped-out regions are filled with zeros in the restored mask.
- Nearest-neighbour interpolation is used for all resizing to preserve labels.

by Zihan Ning <zihan.1.ning@kcl.ac.uk>
@King's College London
2026-01-19
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom


def load_json(json_path: str | Path) -> dict:
    json_path = Path(json_path)
    with json_path.open("r") as f:
        return json.load(f)


def ensure_shape_by_crop_or_pad(vol: np.ndarray, target_shape: np.ndarray) -> np.ndarray:
    """
    Force vol to exactly target_shape by centre-cropping or zero-padding.
    This is used after zoom() because rounding can cause off-by-1 differences.
    """
    target_shape = np.array(target_shape, dtype=int)
    src_shape = np.array(vol.shape, dtype=int)

    out = np.zeros(target_shape, dtype=vol.dtype)

    common = np.minimum(src_shape, target_shape)

    src_start = (src_shape - common) // 2
    dst_start = (target_shape - common) // 2

    src_end = src_start + common
    dst_end = dst_start + common

    out[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = vol[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]
    return out


def undo_final_centre_crop_or_pad(
    final_mask: np.ndarray,
    resampled_shape: np.ndarray,
    src_start: np.ndarray,
) -> np.ndarray:
    """
    Inverse of centre_crop_or_pad() used in preprocess:

    Forward:
      crop_min = min(res_shape, tgt_shape)
      src_start = (res_shape - crop_min)//2
      dst_start = (tgt_shape - crop_min)//2
      out[tgt] = res[src]

    Inverse:
      res[src] = final[tgt]
    """
    resampled_shape = np.array(resampled_shape, dtype=int)
    tgt_shape = np.array(final_mask.shape, dtype=int)

    crop_min = np.minimum(resampled_shape, tgt_shape)
    src_start = np.array(src_start, dtype=int)
    src_end = src_start + crop_min

    dst_start = (tgt_shape - crop_min) // 2
    dst_end = dst_start + crop_min

    res = np.zeros(resampled_shape, dtype=final_mask.dtype)

    res[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ] = final_mask[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ]
    return res


def undo_resample_nn(
    resampled_mask: np.ndarray,
    cropped_shape: np.ndarray,
) -> np.ndarray:
    """
    Undo the resampling step using nearest neighbour.

    We resize from resampled_shape -> cropped_shape by factor = cropped_shape / resampled_shape.
    Then enforce exact shape (off-by-1 safe).
    """
    cropped_shape = np.array(cropped_shape, dtype=int)
    res_shape = np.array(resampled_mask.shape, dtype=int)

    factors = cropped_shape.astype(float) / np.maximum(res_shape.astype(float), 1.0)
    out = zoom(resampled_mask, factors, order=0)  # NN for labels
    out = ensure_shape_by_crop_or_pad(out, cropped_shape)
    return out


def undo_crop1_to_original(
    cropped_mask: np.ndarray,
    original_shape: np.ndarray,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
) -> np.ndarray:
    """
    Put the cropped_mask back into an all-zero original volume at [start:end].
    """
    original_shape = np.array(original_shape, dtype=int)
    start_xyz = np.array(start_xyz, dtype=int)
    end_xyz = np.array(end_xyz, dtype=int)

    out = np.zeros(original_shape, dtype=cropped_mask.dtype)
    out[
        start_xyz[0]:end_xyz[0],
        start_xyz[1]:end_xyz[1],
        start_xyz[2]:end_xyz[2],
    ] = cropped_mask
    return out


def restore_mask(mask_path: str | Path, json_path: str | Path, out_path: str | Path) -> None:
    """
    Restore a single preprocessed mask using its json log.
    """
    mask_path = Path(mask_path)
    json_path = Path(json_path)
    out_path = Path(out_path)

    log = load_json(json_path)

    # ---- load final (preprocessed-space) mask
    mask_nii = nib.load(str(mask_path))
    final_mask = mask_nii.get_fdata(dtype=np.float32)
    final_mask = np.rint(final_mask).astype(np.uint8)  # robust for labels saved as float

    if final_mask.ndim != 3:
        raise ValueError(f"Mask must be 3D. Got shape={final_mask.shape}")

    # ---- read needed metadata from json
    original_shape = np.array(log["original"]["shape"], dtype=int)
    original_affine = np.array(log["original"]["affine"], dtype=float)

    crop_start = np.array(log["crop1"]["start"], dtype=int)
    crop_end = np.array(log["crop1"]["end"], dtype=int)
    cropped_shape = np.array(log["crop1"]["shape"], dtype=int)

    resampled_shape = np.array(log["resample"]["shape"], dtype=int)

    src_start = np.array(log["final_crop"]["start"], dtype=int)

    # ---- A) undo final crop/pad (final -> resampled)
    res_mask = undo_final_centre_crop_or_pad(
        final_mask=final_mask,
        resampled_shape=resampled_shape,
        src_start=src_start,
    )

    # ---- B) undo resample (resampled -> cropped)
    cropped_mask = undo_resample_nn(
        resampled_mask=res_mask,
        cropped_shape=cropped_shape,
    )

    # ---- C) undo crop1 (cropped -> original)
    restored = undo_crop1_to_original(
        cropped_mask=cropped_mask,
        original_shape=original_shape,
        start_xyz=crop_start,
        end_xyz=crop_end,
    )

    from nibabel.orientations import (
        axcodes2ornt,
        ornt_transform,
        apply_orientation,
    )

    # ---- reorient from canonical space back to RAW space
    from nibabel.orientations import aff2axcodes

    canonical_affine = np.array(log["original"]["affine"], dtype=float)
    canonical_axcodes = aff2axcodes(canonical_affine)

    raw_axcodes = tuple(log["raw"]["axcodes"])

    ornt_from = axcodes2ornt(canonical_axcodes)
    ornt_to = axcodes2ornt(raw_axcodes)
    transform = ornt_transform(ornt_from, ornt_to)

    restored_raw = apply_orientation(restored, transform)

    raw_affine = np.array(log["raw"]["affine"], dtype=float)

    out_img = nib.Nifti1Image(restored_raw.astype(np.uint8), raw_affine)
    out_img.set_qform(raw_affine, code=1)
    out_img.set_sform(raw_affine, code=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, help="Preprocessed mask NIfTI (e.g., *_mask_preproc.nii.gz)")
    ap.add_argument("--json", required=True, help="Preprocess JSON log (same base name as mask)")
    ap.add_argument("--out", required=True, help="Output restored mask NIfTI (original grid)")
    args = ap.parse_args()

    restore_mask(args.mask, args.json, args.out)


if __name__ == "__main__":
    main()
