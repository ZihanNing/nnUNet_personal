"""
Robust intensity normalization for cropped + downsampled dental MRI.

- Percentile-based clipping
- Z-score normalization
- Background (0) ignored

by Zihan Ning <zihan.1.ning@kcl.ac.uk>
@King's College London
2026-01-15
"""

import argparse
import numpy as np
import nibabel as nib

def robust_normalize_to_range(
    img: np.ndarray,
    p_low: float = 0.5,
    p_high: float = 99.5,
    out_max: float = 1000.0,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Robust percentile clipping + minmax scaling to [0, out_max].
    Background (0) is kept at 0.
    """
    mask = img > 0
    if not np.any(mask):
        raise RuntimeError("Image contains no non-zero voxels.")

    vals = img[mask]
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)

    img_clip = np.clip(img, lo, hi)

    out = np.zeros_like(img, dtype=np.float32)
    out[mask] = (img_clip[mask] - lo) / (hi - lo + eps)  # [0,1]
    out[mask] *= out_max                                  # [0,out_max]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_nii", help="Input NIfTI (cropped + downsampled)")
    ap.add_argument("out_nii", help="Output normalized NIfTI")
    ap.add_argument("--plow", type=float, default=0.5)
    ap.add_argument("--phigh", type=float, default=99.5)
    args = ap.parse_args()

    nii = nib.load(args.in_nii)
    img = nii.get_fdata(dtype=np.float32)

    img_norm = robust_normalize_to_range(img, out_max=1000.0, p_low=args.plow, p_high=args.phigh)

    nib.save(
        nib.Nifti1Image(img_norm, nii.affine, nii.header),
        args.out_nii
    )

    print(f"[OK] Saved normalized image: {args.out_nii}")


if __name__ == "__main__":
    main()
