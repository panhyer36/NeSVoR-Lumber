import torch
import numpy as np
import nibabel as nib


def load_nifti(path):
    """
    載入 NIfTI 檔案

    Returns:
        data: (Nx, Ny, Nz) float32 numpy array
        affine: (4, 4) 仿射矩陣
        spacing: (3,) voxel spacing (mm)
    """
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine.copy()
    spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    return data, affine, spacing


def compute_world_bbox(shape, affine):
    """
    計算體積在世界座標中的包圍盒

    Args:
        shape: (Nx, Ny, Nz)
        affine: (4, 4) 仿射矩陣
    Returns:
        bbox_min: (3,) 最小世界座標
        bbox_max: (3,) 最大世界座標
    """
    corners = np.array([
        [i, j, k, 1]
        for i in [0, shape[0] - 1]
        for j in [0, shape[1] - 1]
        for k in [0, shape[2] - 1]
    ])
    world = (affine @ corners.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def compute_intersection_roi(bboxes, margin=2.0):
    """
    計算多個包圍盒的交集 ROI

    Args:
        bboxes: list of (bbox_min, bbox_max) tuples
        margin: ROI 邊界外擴 (mm)
    Returns:
        roi_min: (3,) float32
        roi_max: (3,) float32
    """
    roi_min = np.maximum.reduce([bb[0] for bb in bboxes])
    roi_max = np.minimum.reduce([bb[1] for bb in bboxes])
    if not np.all(roi_max > roi_min):
        raise ValueError("掃描之間無有效交集區域！")
    return (
        (roi_min - margin).astype(np.float32),
        (roi_max + margin).astype(np.float32),
    )


def decompose_affine(affine):
    """
    分解 NIfTI 仿射矩陣

    Returns:
        R: (3, 3) 旋轉矩陣（列向量為各軸單位方向）
        col_norms: (3,) 各列的範數 = voxel spacing
        origin: (3,) 平移原點
    """
    M = affine[:3, :3].astype(np.float64)
    col_norms = np.linalg.norm(M, axis=0)
    R = M / col_norms[np.newaxis, :]
    origin = affine[:3, 3].astype(np.float64)
    return R, col_norms, origin


def rotation_matrix_to_axis_angle(R):
    """
    旋轉矩陣 → 軸角表示 (Rodrigues 逆變換)

    Args:
        R: (3, 3) 旋轉矩陣
    Returns:
        axis_angle: (3,) float32 軸角向量
    """
    R = np.array(R, dtype=np.float64)
    det = np.linalg.det(R)

    # 處理瑕旋轉 (det < 0)：翻轉最後一列使其成為正旋轉
    if det < 0:
        R = R.copy()
        R[:, 2] *= -1

    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)

    # theta ≈ π 的特殊情況
    if np.abs(theta - np.pi) < 1e-6:
        B = R + np.eye(3)
        norms = np.linalg.norm(B, axis=0)
        k = np.argmax(norms)
        axis = B[:, k] / norms[k]
        return (axis * theta).astype(np.float32)

    # 一般情況
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(theta))

    return (axis * theta).astype(np.float32)


def compute_psf_sigma(spacing):
    """
    從 voxel spacing 計算 PSF 標準差

    假設 FWHM ≈ voxel spacing，
    sigma = FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.3548

    Args:
        spacing: (3,) voxel spacing [axis0, axis1, axis2]
    Returns:
        psf_sigma: (3,) float32 PSF sigma (mm)
    """
    return (np.array(spacing, dtype=np.float64) / 2.3548).astype(np.float32)


def normalize_to_roi(coords, roi_min, roi_max):
    """
    將世界座標正規化到 [0, 1] (基於 ROI 範圍)

    Args:
        coords: (..., 3) tensor, 世界座標
        roi_min: (3,) tensor
        roi_max: (3,) tensor
    Returns:
        (..., 3) tensor, [0, 1] 範圍
    """
    extent = roi_max - roi_min
    extent = extent.clamp(min=1e-8)
    normed = (coords - roi_min) / extent
    return normed.clamp(0.0, 1.0)
