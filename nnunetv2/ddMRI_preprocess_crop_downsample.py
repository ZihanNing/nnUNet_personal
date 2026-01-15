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
from scipy.ndimage import zoom


TARGET_FOV_MM = np.array([150.0, 160.0, 144.0], dtype=float)  # (X,Y,Z) in "data axes"
TARGET_VOX_MM = np.array([2.0, 2.0, 2.0], dtype=float)
TARGET_SHAPE = np.array([75, 80, 72], dtype=int)

EPS_MM = 1e-3


def voxel_sizes_from_affine(affine: np.ndarray) -> np.ndarray:
    """
    Compute voxel sizes as norms of affine columns.
    by Zihan Ning <zihan.1.ning@kcl.ac.uk>
    @King's College London
    2026-01-14
    """
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def fov_mm(shape_xyz: np.ndarray, vox_mm: np.ndarray) -> np.ndarray:
    """
    FOV in mm along data axes (not world axes) = shape * voxel size.
    by Zihan Ning <zihan.1.ning@kcl.ac.uk>
    @King's College London
    2026-01-14
    """
    return shape_xyz.astype(float) * vox_mm.astype(float)


def crop_affine(affine: np.ndarray, start_xyz: np.ndarray) -> np.ndarray:
    """
    Update affine after voxel-index cropping so orientation stays identical.
    New origin = old origin + A * start
    where A = affine[:3,:3].
    by Zihan Ning <zihan.1.ning@kcl.ac.uk>
    @King's College London
    2026-01-14
    """
    start_xyz = start_xyz.astype(float)
    new_aff = affine.copy()
    new_aff[:3, 3] = affine[:3, 3] + affine[:3, :3] @ start_xyz
    return new_aff


def compute_crop_indices(
    shape_xyz: np.ndarray,
    vox_mm: np.ndarray,
    target_fov_mm: np.ndarray,
    mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute [start, end) indices for cropping to <= target_fov_mm.

    - generic: centre crop (mouth-only scans)
    - mprage: asymmetric crop:
        dim0: remove extra with 30% from low-index(front), 70% from high-index(back)
        dim1: crop from high-index(back) only
        dim2: centre crop

    by Zihan Ning <zihan.1.ning@kcl.ac.uk>
    @King's College London
    2026-01-14
    """
    shape_xyz = shape_xyz.astype(int)
    vox_mm = vox_mm.astype(float)
    target_fov_mm = target_fov_mm.astype(float)

    # number of voxels that best matches target FOV, without exceeding the available shape
    crop_size = np.floor(target_fov_mm / vox_mm + 1e-9).astype(int)
    crop_size = np.minimum(crop_size, shape_xyz)
    crop_size = np.maximum(crop_size, 1)

    extra = shape_xyz - crop_size  # how many voxels to remove per axis

    start = np.zeros(3, dtype=int)
    end = np.zeros(3, dtype=int)

    if mode == "generic":
        # dim0: keep front, crop from back only
        start[0] = 0
        end[0] = crop_size[0]

        # dim1: keep front, crop from back only
        start[1] = 0
        end[1] = crop_size[1]

        # dim2: centre crop
        start[2] = extra[2] // 2
        end[2] = start[2] + crop_size[2]

    elif mode == "mprage":
        # dim0: asymmetric 30/70 split (low/high index)
        extra0 = extra[0]
        low0 = int(np.round(0.30 * extra0))
        high0 = extra0 - low0
        start[0] = low0
        end[0] = shape_xyz[0] - high0

        # dim1: asymmetric 30/70 split (low/high index)
        #start[1] = 0
        #end[1] = shape_xyz[1] - extra[1]
        extra1 = extra[1]
        low1 = int(np.round(0.15 * extra1))
        high1 = extra1 - low1
        start[1] = low1
        end[1] = shape_xyz[1] - high1

        # dim2: symmetric centre crop
        start[2] = extra[2] // 2
        end[2] = start[2] + crop_size[2]

        # sanity: enforce end-start == crop_size
        # (dim0,dim1 may differ by rounding; correct)
        end[0] = start[0] + crop_size[0]
        end[1] = start[1] + crop_size[1]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # clamp to bounds
    start = np.maximum(start, 0)
    end = np.minimum(end, shape_xyz)
    # ensure valid
    if np.any(end <= start):
        raise ValueError(f"Invalid crop indices start={start}, end={end}, shape={shape_xyz}")

    return start, end


def crop_volume(data: np.ndarray, affine: np.ndarray, start: np.ndarray, end: np.ndarray):
    """
    Crop volume and update affine accordingly.
    by Zihan Ning <zihan.1.ning@kcl.ac.uk>
    @King's College London
    2026-01-14
    """
    cropped = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    new_aff = crop_affine(affine, start)
    return cropped, new_aff


def resample_to_voxsize(data: np.ndarray, affine: np.ndarray, target_vox_mm: np.ndarray, is_label: bool):
    """
    Resample with scipy zoom; preserve orientation by scaling affine columns.
    by Zihan Ning <zihan.1.ning@kcl.ac.uk>
    @King's College London
    2026-01-14
    """
    current_vox = voxel_sizes_from_affine(affine)
    zoom_factor = current_vox / target_vox_mm  # >1 means upsample, <1 downsample

    order = 0 if is_label else 1
    out = zoom(data, zoom_factor, order=order)

    # update affine: keep direction, update spacing
    new_aff = affine.copy()
    A = affine[:3, :3]
    # scale columns so their norms become target_vox_mm
    for i in range(3):
        col = A[:, i]
        n = np.linalg.norm(col)
        if n < 1e-12:
            raise ValueError("Degenerate affine column encountered.")
        new_aff[:3, i] = col / n * target_vox_mm[i]

    return out, new_aff


def centre_crop_or_pad(data: np.ndarray, affine: np.ndarray, target_shape: np.ndarray):
    """
    Centre crop/pad to target_shape; update affine if cropping is applied.
    Padding keeps affine unchanged.
    by Zihan Ning <zihan.1.ning@kcl.ac.uk>
    @King's College London
    2026-01-14
    """
    src_shape = np.array(data.shape, dtype=int)
    tgt = target_shape.astype(int)

    # if need crop: compute starts in source
    crop_min = np.minimum(src_shape, tgt)
    src_start = (src_shape - crop_min) // 2
    src_end = src_start + crop_min

    # create output
    out = np.zeros(tgt, dtype=data.dtype)
    dst_start = (tgt - crop_min) // 2
    dst_end = dst_start + crop_min

    out[dst_start[0]:dst_end[0], dst_start[1]:dst_end[1], dst_start[2]:dst_end[2]] = \
        data[src_start[0]:src_end[0], src_start[1]:src_end[1], src_start[2]:src_end[2]]

    # update affine only if we actually cropped in source
    new_aff = affine
    if np.any(src_shape > crop_min):
        new_aff = crop_affine(affine, src_start)

    return out, new_aff


def preprocess(infile: str, outfile: str, mode: str, is_label: bool):
    nii = nib.load(infile)
    data = nii.get_fdata(dtype=np.float32)
    affine = nii.affine

    shape_xyz = np.array(data.shape[:3], dtype=int)
    vox_mm = voxel_sizes_from_affine(affine)
    in_fov = fov_mm(shape_xyz, vox_mm)

    print(f"[INFO] Input: {infile}")
    print(f"[INFO] Shape: {shape_xyz.tolist()}  Vox(mm): {vox_mm.tolist()}")
    print(f"[INFO] FOV(mm): {in_fov.tolist()}")

    # Enforce minimal coverage
    if np.any(in_fov + EPS_MM < TARGET_FOV_MM):
        raise RuntimeError(
            f"Input FOV too small for target crop.\n"
            f"  Input FOV (mm):  {in_fov}\n"
            f"  Target FOV (mm): {TARGET_FOV_MM}\n"
            f"Consider excluding this scan or changing target FOV."
        )

    # Crop to closest possible <= target FOV
    start, end = compute_crop_indices(shape_xyz, vox_mm, TARGET_FOV_MM, mode=mode)
    cropped, aff_crop = crop_volume(data, affine, start, end)

    shape_c = np.array(cropped.shape, dtype=int)
    fov_c = fov_mm(shape_c, vox_mm)  # voxel size unchanged after cropping
    print(f"[INFO] Cropped indices start={start.tolist()} end={end.tolist()}")
    print(f"[INFO] Cropped shape: {shape_c.tolist()}  Cropped FOV(mm): {fov_c.tolist()}")

    # Resample to target voxel size
    res, aff_res = resample_to_voxsize(cropped, aff_crop, TARGET_VOX_MM, is_label=is_label)
    print(f"[INFO] Resampled shape: {list(res.shape)}  Target vox(mm): {TARGET_VOX_MM.tolist()}")

    # Final exact shape
    final, aff_final = centre_crop_or_pad(res, aff_res, TARGET_SHAPE)
    print(f"[INFO] Final shape: {list(final.shape)} (target {TARGET_SHAPE.tolist()})")

    out_img = nib.Nifti1Image(final.astype(np.float32), aff_final)
    nib.save(out_img, outfile)
    print(f"[OK] Saved: {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--mode", choices=["generic", "mprage"], default="generic",
                    help="generic for mouth-only scans; mprage for whole-head MPRAGE")
    ap.add_argument("--label", action="store_true",
                    help="Use nearest-neighbour resampling for label maps")
    args = ap.parse_args()

    preprocess(args.infile, args.outfile, mode=args.mode, is_label=args.label)


if __name__ == "__main__":
    main()
