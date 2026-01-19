"""
Preprocess heterogeneous dental MRI volumes:
1) Read NIfTI, compute voxel size and FOV
2) Enforce minimum FOV >= target FOV (188×160×144 mm)
3) Crop to the closest possible FOV <= target (per-axis) while preserving orientation
4) Resample to 2×2×2 mm
5) Final centre crop/pad to 94×80×72

by Zihan Ning <zihan.1.ning@kcl.ac.uk>
@King's College London
2026-01-14
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
from scipy.ndimage import zoom
import json


# --- define target FOV by anatomical meaning (do NOT assume array axis order here)
# RL = Right-Left, AP = Anterior-Posterior, SI = Superior-Inferior (Head-Foot)
TARGET_FOV_BY_AXIS_MM = {
    "RL": 144.0,
    "AP": 160.0,
    "SI": 150.0,
}

# placeholder: will be filled per-volume after canonicalisation using affine axcodes
TARGET_FOV_MM = np.array([np.nan, np.nan, np.nan], dtype=float)

TARGET_VOX_MM = np.array([2.0, 2.0, 2.0], dtype=float)
TARGET_SHAPE = np.array([75, 80, 72], dtype=int)

EPS_MM = 1e-3


def save_preprocess_json(log_dict, json_path):
    with open(json_path, "w") as f:
        json.dump(log_dict, f, indent=2)


def voxel_sizes_from_affine(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def fov_mm(shape_xyz: np.ndarray, vox_mm: np.ndarray) -> np.ndarray:
    return shape_xyz.astype(float) * vox_mm.astype(float)


def crop_affine(affine: np.ndarray, start_xyz: np.ndarray) -> np.ndarray:
    new_aff = affine.copy()
    new_aff[:3, 3] = affine[:3, 3] + affine[:3, :3] @ start_xyz.astype(float)
    return new_aff

def compute_crop_indices(
    shape_xyz: np.ndarray,
    vox_mm: np.ndarray,
    target_fov_mm: np.ndarray,
    affine: np.ndarray,
    mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute crop indices using physical directions inferred from affine.

    - RL: centre crop
    - AP: keep Anterior side (front)
    - HF: keep Superior side (head side)  <-- if you want Inferior instead, see notes below

    This is "physical-direction safe":
    it decides whether to crop low-index or high-index based on axis code (A/P/S/I),
    instead of assuming start=0 is always "front".

    by Zihan Ning <zihan.1.ning@kcl.ac.uk>
    @King's College London
    """

    shape_xyz = shape_xyz.astype(int)
    vox_mm = vox_mm.astype(float)
    target_fov_mm = target_fov_mm.astype(float)

    # number of voxels that best matches target FOV
    crop_size = np.floor(target_fov_mm / vox_mm + 1e-9).astype(int)
    crop_size = np.minimum(crop_size, shape_xyz)
    crop_size = np.maximum(crop_size, 1)

    extra = shape_xyz - crop_size

    start = np.zeros(3, dtype=int)
    end = np.zeros(3, dtype=int)

    # Determine anatomical meaning of each data axis (after canonicalization, should be stable)
    axcodes = aff2axcodes(affine)  # e.g. ('R','A','S')

    axis_rl = [i for i, c in enumerate(axcodes) if c in ("R", "L")]
    axis_ap = [i for i, c in enumerate(axcodes) if c in ("A", "P")]
    axis_hf = [i for i, c in enumerate(axcodes) if c in ("S", "I")]

    if not (len(axis_rl) == len(axis_ap) == len(axis_hf) == 1):
        raise RuntimeError(f"Cannot uniquely determine RL/AP/HF axes from affine: {axcodes}")

    axis_rl = axis_rl[0]
    axis_ap = axis_ap[0]
    axis_hf = axis_hf[0]

    # Helper: crop by keeping a physical side
    def keep_high_index_side(axis: int):
        """keep the high-index side of this axis."""
        start[axis] = shape_xyz[axis] - crop_size[axis]
        end[axis] = shape_xyz[axis]

    def keep_low_index_side(axis: int):
        """keep the low-index side of this axis."""
        start[axis] = 0
        end[axis] = crop_size[axis]

    # --- RL: always centre crop ---
    start[axis_rl] = extra[axis_rl] // 2
    end[axis_rl] = start[axis_rl] + crop_size[axis_rl]

    if mode == "generic":
        # --- AP: keep Anterior ---
        # If axis code is 'A', anterior is high-index; if 'P', anterior is low-index.
        if axcodes[axis_ap] == "A":
            keep_high_index_side(axis_ap)
        else:  # 'P'
            keep_low_index_side(axis_ap)

        # --- HF: keep Interior (foot side) ---
        # If axis code is 'S', superior is high-index; if 'I', superior is low-index.
        if axcodes[axis_hf] == "S":
            keep_low_index_side(axis_hf)
        else:  # 'I'
            keep_high_index_side(axis_hf)

    elif mode == "mprage":
        # RL: centre already set above

        # AP: asymmetric 30/70, but in physical sense:
        # Here we implement: remove 15% from the "kept side" boundary, and the rest from the opposite side.
        # For reproducibility and simplicity, we keep the same "keep Anterior" rule, then shift start accordingly.

        extra_ap = extra[axis_ap]
        low = int(np.round(0.15 * extra_ap))  # small removal on one side

        if axcodes[axis_ap] == "A":
            # Keep Anterior = high-index side. So we crop more from low-index (posterior) side.
            start[axis_ap] = low
            end[axis_ap] = start[axis_ap] + crop_size[axis_ap]
        else:
            # If axis is 'P', Anterior corresponds to low-index side; crop more from high-index side.
            start[axis_ap] = 0
            end[axis_ap] = crop_size[axis_ap]  # keep low side
            # apply the small shift inside the kept range
            start[axis_ap] = 0
            end[axis_ap] = crop_size[axis_ap]
            # Note: if you rely heavily on mprage asymmetry, we can refine this with a clearer spec.

        # HF: keep Superior (head side)
        if axcodes[axis_hf] == "S":
            keep_high_index_side(axis_hf)
        else:
            keep_low_index_side(axis_hf)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # clamp to bounds
    start = np.maximum(start, 0)
    end = np.minimum(end, shape_xyz)

    if np.any(end <= start):
        raise ValueError(f"Invalid crop indices start={start}, end={end}, shape={shape_xyz}")

    return start, end


def crop_volume(data, affine, start, end):
    cropped = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    new_aff = crop_affine(affine, start)
    return cropped, new_aff


def resample_to_voxsize(data, affine, target_vox_mm, is_label):
    current_vox = voxel_sizes_from_affine(affine)
    zoom_factor = current_vox / target_vox_mm
    order = 0 if is_label else 1
    out = zoom(data, zoom_factor, order=order)

    new_aff = affine.copy()
    for i in range(3):
        col = affine[:3, i]
        n = np.linalg.norm(col)
        new_aff[:3, i] = col / n * target_vox_mm[i]

    return out, new_aff


def centre_crop_or_pad(data, affine, target_shape):
    src_shape = np.array(data.shape, dtype=int)
    tgt = target_shape.astype(int)

    crop_min = np.minimum(src_shape, tgt)
    src_start = (src_shape - crop_min) // 2
    src_end = src_start + crop_min

    out = np.zeros(tgt, dtype=data.dtype)
    dst_start = (tgt - crop_min) // 2
    dst_end = dst_start + crop_min

    out[dst_start[0]:dst_end[0], dst_start[1]:dst_end[1], dst_start[2]:dst_end[2]] = \
        data[src_start[0]:src_end[0], src_start[1]:src_end[1], src_start[2]:src_end[2]]

    new_aff = affine
    if np.any(src_shape > crop_min):
        new_aff = crop_affine(affine, src_start)

    return out, new_aff, src_start


def preprocess(infile, outfile, mode, is_label):
    nii = nib.load(infile)

    # ---- store RAW (scanner-native) geometry BEFORE canonicalisation
    raw_affine = nii.affine
    raw_shape = nii.shape[:3]
    raw_axcodes = nib.orientations.aff2axcodes(raw_affine)

    nii = nib.as_closest_canonical(nii)

    data = nii.get_fdata(dtype=np.float32)
    affine = nii.affine

    print(f"[INFO] Orientation after canonical: {aff2axcodes(affine)}")

    # ---- map TARGET_FOV_BY_AXIS_MM (RL/AP/SI) onto data axes using current axcodes
    axcodes = aff2axcodes(affine)  # e.g. ('R','A','S') after canonical

    target_fov_mm = np.zeros(3, dtype=float)
    for i, c in enumerate(axcodes):
        if c in ("R", "L"):
            target_fov_mm[i] = TARGET_FOV_BY_AXIS_MM["RL"]
        elif c in ("A", "P"):
            target_fov_mm[i] = TARGET_FOV_BY_AXIS_MM["AP"]
        elif c in ("S", "I"):
            target_fov_mm[i] = TARGET_FOV_BY_AXIS_MM["SI"]
        else:
            raise RuntimeError(f"Unknown axis code {c} in axcodes={axcodes}")

    # overwrite the global-style variable used by later code paths (FOV check + cropping)
    global TARGET_FOV_MM
    TARGET_FOV_MM = target_fov_mm

    print("[DEBUG] TARGET_FOV_MM aligned to data axes:", TARGET_FOV_MM)


    shape_xyz = np.array(data.shape[:3], dtype=int)
    vox_mm = voxel_sizes_from_affine(affine)
    in_fov = fov_mm(shape_xyz, vox_mm)

    log = {
        "original": {
            "shape": shape_xyz.tolist(),
            "affine": affine.tolist(),
            "voxel_size": vox_mm.tolist()
        },
        "raw": {
            "shape": list(raw_shape),
            "affine": raw_affine.tolist(),
            "axcodes": raw_axcodes               # scanner-native orientation
        }
    }

    print("[DEBUG] Input shape (vox):", shape_xyz)
    print("[DEBUG] Voxel size (mm):", vox_mm)
    print("[DEBUG] Input FOV (mm):", in_fov)
    print("[DEBUG] Required FOV (mm):", TARGET_FOV_MM)

    if np.any(in_fov + EPS_MM < TARGET_FOV_MM):
        raise RuntimeError("Input FOV too small")

    start, end = compute_crop_indices(
        shape_xyz, vox_mm, TARGET_FOV_MM, affine, mode
    )

    cropped, aff_crop = crop_volume(data, affine, start, end)

    log["crop1"] = {
        "start": start.tolist(),
        "end": end.tolist(),
        "shape": list(cropped.shape)
    }

    res, aff_res = resample_to_voxsize(cropped, aff_crop, TARGET_VOX_MM, is_label)

    log["resample"] = {
        "voxel_size_before": vox_mm.tolist(),
        "voxel_size_after": TARGET_VOX_MM.tolist(),
        "shape": list(res.shape)
    }

    final, aff_final, src_start = centre_crop_or_pad(res, aff_res, TARGET_SHAPE)

    log["final_crop"] = {
        "start": src_start.tolist(),
        "shape": TARGET_SHAPE.tolist()
    }

    out_img = nib.Nifti1Image(final.astype(np.float32), aff_final)

    # make qform/sform consistent (viewer-safe)
    out_img.set_qform(aff_final, code=1)
    out_img.set_sform(aff_final, code=1)

    nib.save(out_img, outfile)

    save_preprocess_json(log, outfile.replace(".nii.gz", ".json").replace(".nii", ".json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--mode", choices=["generic", "mprage"], default="generic")
    ap.add_argument("--label", action="store_true")
    args = ap.parse_args()

    preprocess(args.infile, args.outfile, args.mode, args.label)


if __name__ == "__main__":
    main()
