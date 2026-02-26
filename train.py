import torch
import numpy as np
from tqdm import tqdm
from src.model import NeSVoR
from src.losses import (reconstruction_loss,
                    image_regularization,
                    bias_regularization)
from src.config import NeSVoRConfig
from src.dataset import SpinalMRIDataset


# ================================================================
#  訓練過程可視化 — 每隔 N 步擷取重建切面，最後輸出 GIF
# ================================================================

def _sample_slice(model, roi_min, roi_max, device,
                  axis=2, frac=0.5, resolution=128):
    """
    從目前模型擷取一張 2D 切面（世界座標）。

    Args:
        axis: 切面法線方向 (0=sagittal, 1=coronal, 2=axial)
        frac: 在該軸上的比例位置 [0,1]
        resolution: 輸出影像邊長 (px)
    Returns:
        (resolution, resolution) numpy float32 陣列
    """
    model.eval()
    roi_min_np = roi_min.cpu().numpy()
    roi_max_np = roi_max.cpu().numpy()

    # 決定切面位置
    axes = [i for i in range(3) if i != axis]
    fixed_val = roi_min_np[axis] + frac * (roi_max_np[axis] - roi_min_np[axis])

    u = np.linspace(roi_min_np[axes[0]], roi_max_np[axes[0]], resolution)
    v = np.linspace(roi_min_np[axes[1]], roi_max_np[axes[1]], resolution)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    coords = np.zeros((resolution * resolution, 3), dtype=np.float32)
    coords[:, axes[0]] = uu.ravel()
    coords[:, axes[1]] = vv.ravel()
    coords[:, axis] = fixed_val

    coords_t = torch.from_numpy(coords).to(device)
    with torch.no_grad():
        vals = model.sample_volume(coords_t)
    img = vals.cpu().numpy().reshape(resolution, resolution)
    model.train()
    return img


def _make_frame(img, step, loss, vmin, vmax):
    """
    將 2D 陣列渲染為帶標注的 RGB 圖片 (numpy uint8)。
    使用 matplotlib 離屏渲染，不彈出視窗。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.imshow(img.T, origin="lower", cmap="gray",
              vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(f"Step {step}  |  Loss {loss:.4f}", fontsize=11)
    ax.axis("off")
    fig.tight_layout(pad=0.3)

    # 渲染到 numpy RGB
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    frame = buf[:, :, :3].copy()  # RGBA → RGB
    plt.close(fig)
    return frame


def save_training_gif(frames, path="training.gif", fps=5):
    """將 frame list 儲存為 GIF。"""
    import imageio.v2 as imageio
    imageio.mimsave(str(path), frames, fps=fps, loop=0)
    print(f"訓練過程 GIF 已儲存: {path}  ({len(frames)} 幀, {fps} fps)")


# ================================================================
#  主訓練函式
# ================================================================

def train(dataset_or_paths, config=None, downsample=1,
          gif_path="training.gif", gif_interval=None,
          gif_axis=2, gif_frac=0.5, gif_res=128):
    """
    NeSVoR 脊椎 MRI 訓練流程

    Args:
        dataset_or_paths: SpinalMRIDataset 或 dict (name→path)
        config: NeSVoRConfig（None = 預設）
        downsample: 面內降採樣倍率
        gif_path: GIF 輸出路徑（None = 不產生 GIF）
        gif_interval: 每幾步擷取一幀（None = 自動, ~30 幀）
        gif_axis: 切面法線 (0=sagittal, 1=coronal, 2=axial)
        gif_frac: 切面位置比例 [0,1]
        gif_res: 切面解析度 (px)
    Returns:
        model, dataset
    """
    if config is None:
        config = NeSVoRConfig()

    # --- 1. 準備資料集 ---
    if isinstance(dataset_or_paths, SpinalMRIDataset):
        dataset = dataset_or_paths
    else:
        dataset = SpinalMRIDataset(
            dataset_or_paths, downsample=downsample
        )

    data = dataset.get_training_data()
    device = config.device

    slices_data = data["intensities"].to(device)
    pixel_positions = data["pixel_positions"].to(device)
    slice_indices = data["slice_indices"].to(device)
    psf_sigmas = data["psf_sigmas"].to(device)       # (n_slices, 3)
    roi_min = data["roi_min"].to(device)
    roi_max = data["roi_max"].to(device)
    n_slices = data["n_slices"]

    # --- 2. 建立模型 ---
    model = NeSVoR(n_slices, config).to(device)
    model.roi_min.copy_(roi_min)
    model.roi_max.copy_(roi_max)

    # 以 Affine Matrix 初始化切片剛體變換
    initial_transforms = data["initial_transforms"].to(device)
    with torch.no_grad():
        model.slice_transforms.copy_(initial_transforms)
    print(f"已從 Affine Matrix 初始化 {n_slices} 個切片的剛體變換")

    # --- 3. 優化器 ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr
    )
    milestones = [
        config.n_iterations // 2,
        config.n_iterations * 3 // 4,
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones,
        gamma=config.lr_decay
    )

    N_total = slices_data.shape[0]
    print(f"\n開始訓練: {config.n_iterations} iterations, "
          f"batch_size={config.batch_size}, device={device}")
    print(f"像素總數: {N_total:,}, 切片數: {n_slices}")

    # --- GIF 設定 ---
    make_gif = gif_path is not None
    frames = []
    if make_gif:
        if gif_interval is None:
            gif_interval = max(1, config.n_iterations // 30)
        print(f"GIF: 每 {gif_interval} 步擷取切面 "
              f"(axis={gif_axis}, frac={gif_frac:.1f}, "
              f"res={gif_res})")

    losses_for_gif = []
    current_loss = 0.0

    # --- 4. 訓練迴圈 ---
    for step in tqdm(range(config.n_iterations)):
        # 隨機取樣 batch
        idx = torch.randint(0, N_total, (config.batch_size,),
                            device=device)
        batch_I = slices_data[idx]
        batch_pos = pixel_positions[idx]
        batch_slice = slice_indices[idx]

        # 前向傳播
        mean_I, total_var, reg_data = model(
            batch_pos, batch_slice, psf_sigmas
        )

        # 損失: L = L_I + λ_B R_B + λ_V R_V
        loss_recon = reconstruction_loss(
            mean_I, total_var, batch_I
        )
        loss_tv = image_regularization(
            reg_data['volume_values'],
            reg_data['sample_coords'],
            K=config.n_samples,
        )
        loss_bias = bias_regularization(reg_data['log_bias'])

        loss = (loss_recon
                + config.lambda_v * loss_tv
                + config.lambda_b * loss_bias)

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = loss.item()

        if step % 500 == 0:
            print(f"Step {step}, Loss: {current_loss:.4f} "
                  f"(recon={loss_recon.item():.4f}, "
                  f"TV={loss_tv.item():.4f}, "
                  f"bias={loss_bias.item():.4f}), "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # --- 擷取 GIF 幀 ---
        if make_gif and (step % gif_interval == 0
                         or step == config.n_iterations - 1):
            img = _sample_slice(model, roi_min, roi_max, device,
                                axis=gif_axis, frac=gif_frac,
                                resolution=gif_res)
            losses_for_gif.append(current_loss)
            frames.append((img, step, current_loss))

    # --- 5. 產生 GIF ---
    if make_gif and frames:
        # 統一灰度範圍
        all_imgs = [f[0] for f in frames]
        vmin = min(im.min() for im in all_imgs)
        vmax = max(im.max() for im in all_imgs)
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0

        rendered = [_make_frame(img, s, l, vmin, vmax)
                    for img, s, l in frames]
        save_training_gif(rendered, path=gif_path, fps=5)

    return model, dataset


# ================================================================
#  CLI 入口
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NeSVoR 脊椎 MRI 訓練")
    parser.add_argument("--data_dir", type=str,
                        default="102@19-OCT-23@NIFTI",
                        help="含 coronal.nii.gz / sagittal.nii.gz 的資料夾")
    parser.add_argument("--output", type=str, default="model.pt",
                        help="模型儲存路徑")
    parser.add_argument("--downsample", type=int, default=1,
                        help="面內降採樣倍率 (1=原始解析度)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="覆蓋預設迭代次數")
    # GIF 相關參數
    parser.add_argument("--gif", type=str, default="training.gif",
                        help="訓練過程 GIF 路徑 (設為 'none' 可關閉)")
    parser.add_argument("--gif_interval", type=int, default=None,
                        help="每幾步擷取一幀 (None=自動)")
    parser.add_argument("--gif_axis", type=int, default=2,
                        choices=[0, 1, 2],
                        help="切面法線 (0=sagittal, 1=coronal, 2=axial)")
    parser.add_argument("--gif_frac", type=float, default=0.5,
                        help="切面位置 [0,1]")
    parser.add_argument("--gif_res", type=int, default=128,
                        help="切面解析度 (px)")
    args = parser.parse_args()

    from pathlib import Path
    data_dir = Path(args.data_dir)
    nifti_paths = {
        "coronal": str(data_dir / "coronal.nii.gz"),
        "sagittal": str(data_dir / "sagittal.nii.gz"),
    }

    config = NeSVoRConfig()
    if args.iterations is not None:
        config.n_iterations = args.iterations

    gif_path = None if args.gif.lower() == "none" else args.gif

    model, dataset = train(
        nifti_paths, config=config,
        downsample=args.downsample,
        gif_path=gif_path,
        gif_interval=args.gif_interval,
        gif_axis=args.gif_axis,
        gif_frac=args.gif_frac,
        gif_res=args.gif_res,
    )

    # 儲存模型
    torch.save({
        "model_state_dict": model.state_dict(),
        "roi_min": dataset.roi_min,
        "roi_max": dataset.roi_max,
        "n_slices": dataset.n_slices,
        "config": config,
    }, args.output)
    print(f"\n模型已儲存至: {args.output}")
