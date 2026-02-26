import torch
import numpy as np
from pathlib import Path

from .utils import (
    load_nifti,
    compute_world_bbox,
    compute_intersection_roi,
    decompose_affine,
    rotation_matrix_to_axis_angle,
    compute_psf_sigma,
)


class SpinalMRIDataset:
    """
    脊椎 MRI 資料集

    載入多個方向的 MRI 掃描（冠狀面 / 矢狀面），
    計算世界座標交集 ROI，
    並準備 NeSVoR 訓練所需的全部張量。
    """

    def __init__(
        self,
        nifti_paths,
        intensity_threshold=0.01,
        roi_margin=2.0,
        downsample=1,
    ):
        """
        Args:
            nifti_paths: dict, e.g. {"coronal": "path.nii.gz", "sagittal": "..."}
            intensity_threshold: 排除低於 max*threshold 的背景像素
            roi_margin: ROI 外擴 (mm)
            downsample: 面內降採樣倍率 (1=不降採樣, 2=取每隔2個像素)
        """
        self.downsample = max(1, int(downsample))
        self.scans = {}
        bboxes = []

        # --- 1. 載入所有掃描 ---
        for name, path in nifti_paths.items():
            data, affine, spacing = load_nifti(path)
            bbox = compute_world_bbox(data.shape, affine)
            self.scans[name] = dict(
                data=data, affine=affine, spacing=spacing, shape=data.shape
            )
            bboxes.append(bbox)
            print(
                f"[{name}] Shape: {data.shape}, "
                f"Spacing: ({spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f}) mm, "
                f"Range: [{data.min():.1f}, {data.max():.1f}]"
            )

        # --- 2. 計算交集 ROI ---
        self.roi_min, self.roi_max = compute_intersection_roi(
            bboxes, margin=roi_margin
        )
        self.roi_extent = self.roi_max - self.roi_min
        print(f"\nROI 交集:")
        print(f"  Min : {self.roi_min}")
        print(f"  Max : {self.roi_max}")
        print(f"  範圍: {self.roi_extent} mm")

        # --- 3. 提取切片像素 ---
        self.intensity_threshold = intensity_threshold
        self._prepare_slices()

    # ------------------------------------------------------------------
    def _prepare_slices(self):
        """將所有切片的像素資料提取成 flat tensors."""
        all_intensities = []
        all_positions = []
        all_slice_indices = []
        initial_transforms = []   # (n_slices, 6)
        psf_sigmas = []           # (n_slices, 3)

        slice_counter = 0
        ds = self.downsample

        for scan_name, scan in self.scans.items():
            data = scan["data"]
            affine = scan["affine"]
            Nx, Ny, Nz = scan["shape"]

            # 分解仿射
            R, col_norms, origin = decompose_affine(affine)
            axis_angle = rotation_matrix_to_axis_angle(R)
            psf_sigma = compute_psf_sigma(col_norms)

            threshold = data.max() * self.intensity_threshold

            print(
                f"\n[{scan_name}] {Nz} slices, "
                f"axis-angle: [{axis_angle[0]:.4f}, {axis_angle[1]:.4f}, {axis_angle[2]:.4f}], "
                f"PSF sigma: ({psf_sigma[0]:.4f}, {psf_sigma[1]:.4f}, {psf_sigma[2]:.4f}) mm"
            )

            # 面內像素索引（可降採樣）
            ii_range = np.arange(0, Nx, ds)
            jj_range = np.arange(0, Ny, ds)
            ii_grid, jj_grid = np.meshgrid(ii_range, jj_range, indexing="ij")
            ii_flat = ii_grid.ravel()
            jj_flat = jj_grid.ravel()
            n_pixels_per_slice = len(ii_flat)

            for k in range(Nz):
                # 取得像素強度
                intensities = data[ii_flat, jj_flat, k]

                # 計算世界座標
                vox = np.column_stack([
                    ii_flat.astype(np.float64),
                    jj_flat.astype(np.float64),
                    np.full(n_pixels_per_slice, k, dtype=np.float64),
                    np.ones(n_pixels_per_slice, dtype=np.float64),
                ])
                world = (affine @ vox.T).T[:, :3]  # (N, 3)

                # 過濾條件：ROI 內 且 強度 > 閾值
                in_roi = np.all(
                    (world >= self.roi_min) & (world <= self.roi_max),
                    axis=1,
                )
                above_thresh = intensities > threshold
                valid = in_roi & above_thresh

                n_valid = int(valid.sum())
                if n_valid == 0:
                    continue

                # 本地座標 (mm)：面內方向 × spacing，通過面 = 0
                local_coords = np.zeros((n_valid, 3), dtype=np.float32)
                local_coords[:, 0] = ii_flat[valid] * col_norms[0]
                local_coords[:, 1] = jj_flat[valid] * col_norms[1]
                # local[:, 2] = 0  (切片在本地座標 z=0 平面)

                # 此切片的世界平移：affine @ [0, 0, k, 1]
                t_k = (affine[:3, :3] @ np.array([0.0, 0.0, k]) + origin)
                transform = np.concatenate([
                    axis_angle, t_k.astype(np.float32)
                ])

                all_intensities.append(intensities[valid].astype(np.float32))
                all_positions.append(local_coords)
                all_slice_indices.append(
                    np.full(n_valid, slice_counter, dtype=np.int64)
                )
                initial_transforms.append(transform)
                psf_sigmas.append(psf_sigma)
                slice_counter += 1

        # --- 合併 ---
        self.intensities = torch.from_numpy(
            np.concatenate(all_intensities)
        ).float()
        self.pixel_positions = torch.from_numpy(
            np.concatenate(all_positions)
        ).float()
        self.slice_indices = torch.from_numpy(
            np.concatenate(all_slice_indices)
        ).long()
        self.initial_transforms = torch.from_numpy(
            np.stack(initial_transforms)
        ).float()
        self.psf_sigmas = torch.from_numpy(
            np.stack(psf_sigmas)
        ).float()
        self.n_slices = slice_counter

        # 正規化強度到 [0, 1]
        i_max = self.intensities.max()
        if i_max > 0:
            self.intensities = self.intensities / i_max

        print(f"\n{'='*40}")
        print(f"資料集摘要")
        print(f"  切片總數 : {self.n_slices}")
        print(f"  像素總數 : {len(self.intensities):,}")
        print(f"  ROI Min  : {self.roi_min}")
        print(f"  ROI Max  : {self.roi_max}")
        print(f"{'='*40}")

    # ------------------------------------------------------------------
    def get_training_data(self):
        """
        回傳訓練所需的全部資料

        Returns: dict with keys:
            intensities        (N,)          像素強度 [0,1]
            pixel_positions    (N, 3)        本地座標 (mm)
            slice_indices      (N,)          切片索引
            initial_transforms (n_slices, 6) 初始剛體 [axis_angle(3), translation(3)]
            psf_sigmas         (n_slices, 3) 每切片 PSF sigma (mm)
            n_slices           int
            roi_min            (3,) tensor
            roi_max            (3,) tensor
        """
        return dict(
            intensities=self.intensities,
            pixel_positions=self.pixel_positions,
            slice_indices=self.slice_indices,
            initial_transforms=self.initial_transforms,
            psf_sigmas=self.psf_sigmas,
            n_slices=self.n_slices,
            roi_min=torch.from_numpy(self.roi_min).float(),
            roi_max=torch.from_numpy(self.roi_max).float(),
        )
