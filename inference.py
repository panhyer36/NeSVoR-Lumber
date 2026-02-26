import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from src.model import NeSVoR
from src.config import NeSVoRConfig


@torch.no_grad()
def reconstruct_volume(model, output_spacing=0.8,
                       roi_min=None, roi_max=None,
                       device=None):
    """
    從訓練好的 NeSVoR 模型重建 3D 體積

    模型內部已儲存 ROI 範圍，可自動正規化座標。
    傳入世界座標即可，無須手動正規化。

    Args:
        model: 訓練好的 NeSVoR 模型
        output_spacing: 輸出體素間距 (mm)
        roi_min: (3,) ROI 最小座標 (None = 使用 model 內建值)
        roi_max: (3,) ROI 最大座標 (None = 使用 model 內建值)
        device: 計算裝置 (None = 使用 model 所在裝置)
    Returns:
        volume: 3D numpy 陣列
        affine: 4x4 仿射矩陣
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    if roi_min is None:
        roi_min = model.roi_min.cpu()
    if roi_max is None:
        roi_max = model.roi_max.cpu()

    roi_min_np = roi_min.numpy() if isinstance(roi_min, torch.Tensor) else np.array(roi_min)
    roi_max_np = roi_max.numpy() if isinstance(roi_max, torch.Tensor) else np.array(roi_max)

    # 建立輸出網格
    extent = roi_max_np - roi_min_np
    nx = max(1, int(np.ceil(extent[0] / output_spacing)))
    ny = max(1, int(np.ceil(extent[1] / output_spacing)))
    nz = max(1, int(np.ceil(extent[2] / output_spacing)))

    print(f"重建體積: {nx} x {ny} x {nz} = {nx*ny*nz:,} voxels, "
          f"spacing={output_spacing:.2f} mm")

    # 產生世界座標網格
    xs = torch.linspace(roi_min_np[0], roi_max_np[0], nx)
    ys = torch.linspace(roi_min_np[1], roi_max_np[1], ny)
    zs = torch.linspace(roi_min_np[2], roi_max_np[2], nz)
    grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
    coords = torch.stack(
        [grid_x, grid_y, grid_z], dim=-1
    ).reshape(-1, 3)

    # 分批取樣（避免 GPU 記憶體不足）
    batch_size = 65536
    volumes = []
    for i in range(0, coords.shape[0], batch_size):
        batch = coords[i:i + batch_size].to(device)
        # sample_volume 會在內部正規化到 [0,1]
        v = model.sample_volume(batch)
        volumes.append(v.cpu())

    volume = torch.cat(volumes).reshape(nx, ny, nz).numpy()

    # 各向同性高斯 PSF 平滑（避免混疊與雜訊）
    # σ = r / 2.3548 (FWHM = r)，轉換為體素單位: σ_voxel = 1 / 2.3548
    sigma_voxel = output_spacing / 2.3548 / output_spacing  # = 1/2.3548
    volume = gaussian_filter(volume, sigma=sigma_voxel)
    print(f"已套用各向同性高斯 PSF: σ={output_spacing/2.3548:.4f} mm "
          f"(σ_voxel={sigma_voxel:.4f})")

    # NIfTI 仿射矩陣：對角 spacing + ROI 原點
    affine = np.eye(4)
    affine[0, 0] = output_spacing
    affine[1, 1] = output_spacing
    affine[2, 2] = output_spacing
    affine[:3, 3] = roi_min_np

    return volume, affine


def save_nifti(volume, affine, filename):
    """將重建體積儲存為 NIfTI 檔案"""
    img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(img, str(filename))
    print(f"已儲存: {filename}, "
          f"形狀: {volume.shape}, "
          f"體素間距: {affine[0,0]:.2f} mm")


# ---- 直接執行入口 ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeSVoR 3D 體積重建")
    parser.add_argument("--model", type=str, default="model.pt",
                        help="訓練好的模型檔案路徑")
    parser.add_argument("--output", type=str,
                        default="reconstructed.nii.gz",
                        help="輸出 NIfTI 檔案路徑")
    parser.add_argument("--spacing", type=float, default=0.8,
                        help="輸出體素間距 (mm)")
    args = parser.parse_args()

    # 載入模型
    checkpoint = torch.load(args.model, map_location="cpu",
                            weights_only=False)
    config = checkpoint["config"]
    n_slices = checkpoint["n_slices"]
    roi_min = checkpoint["roi_min"]
    roi_max = checkpoint["roi_max"]

    model = NeSVoR(n_slices, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.device)

    # 重建
    volume, affine = reconstruct_volume(
        model,
        output_spacing=args.spacing,
        roi_min=torch.from_numpy(roi_min).float(),
        roi_max=torch.from_numpy(roi_max).float(),
        device=config.device,
    )

    save_nifti(volume, affine, args.output)
