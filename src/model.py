import torch
import torch.nn as nn
import numpy as np
from .hash_encoding import HashGridEncoding
from .networks import VolumeNetwork, BiasFieldNetwork, VarianceNetwork
from .config import NeSVoRConfig
 
 
class NeSVoR(nn.Module):
    """
    NeSVoR 主模型
 
    將切片獲取過程建模為：
      I_ij = C_i * ∫ M_ij(x) B_i(x) [V(x) + ε_i(x)] dx
 
    通過蒙特卡羅取樣近似積分，並用神經網路參數化
    V(x), B_i(x), σ²_i(x)
    """
 
    def __init__(self, n_slices, config=None):
        """
        Args:
            n_slices: 輸入切片總數
            config: NeSVoRConfig 實例
        """
        super().__init__()
        if config is None:
            config = NeSVoRConfig()
        self.config = config
        self.n_slices = n_slices
 
        # === 1. 雜湊網格編碼器 ===
        self.encoding = HashGridEncoding(
            n_levels=config.n_levels,
            n_features=config.n_features,
            log2_hashmap_size=config.log2_hashmap_size,
            base_resolution=config.base_resolution,
            growth_factor=config.growth_factor,
        )
 
        full_enc_dim = self.encoding.output_dim
        low_enc_dim = config.bias_levels * config.n_features
 
        # === 2. 三個 MLP 分支 ===
        self.volume_net = VolumeNetwork(
            encoding_dim=full_enc_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden_layers,
            feature_dim=config.embedding_dim,
        )
        self.bias_net = BiasFieldNetwork(
            low_encoding_dim=low_enc_dim,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden_layers,
        )
        self.variance_net = VarianceNetwork(
            feature_dim=config.embedding_dim,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            n_hidden=config.n_hidden_layers,
        )
 
        # === 3. 切片特定參數 ===
        # 切片嵌入向量: 編碼切片特定資訊
        self.slice_embeddings = nn.Parameter(
            torch.randn(n_slices, config.embedding_dim)
        )
 
        # 切片縮放因子: softmax 重參數化確保平均為 1
        self.slice_scaling_logits = nn.Parameter(
            torch.zeros(n_slices)
        )
 
        # 切片級方差: 對數形式儲存確保正值
        self.log_slice_variance = nn.Parameter(
            torch.full((n_slices,), -7.0)
        )
 
        # 切片剛體變換: 6D (3 旋轉 + 3 平移)
        self.slice_transforms = nn.Parameter(
            torch.zeros(n_slices, 6)
        )

        # ROI 範圍 (用於座標正規化到 [0,1])
        self.register_buffer('roi_min', torch.zeros(3))
        self.register_buffer('roi_max', torch.ones(3))
 
    def get_scaling_factors(self):
        """
        計算切片縮放因子，使用 softmax 重參數化
        確保平均值為 1: C_i = N_s * softmax(c)_i
        """
        return self.n_slices * torch.softmax(
            self.slice_scaling_logits, dim=0
        )
 
    def axis_angle_to_matrix(self, axis_angle):
        """
        軸角表示轉旋轉矩陣 (Rodrigues 公式)
 
        Args:
            axis_angle: (N, 3) 旋轉向量
                        方向 = 旋轉軸, 範數 = 旋轉角度
        Returns:
            (N, 3, 3) 旋轉矩陣
        """
        theta = torch.norm(axis_angle, dim=-1, keepdim=True)
        # 避免零角度的數值問題
        theta = theta.clamp(min=1e-8)
        k = axis_angle / theta  # 單位旋轉軸
        K = torch.zeros(*k.shape[:-1], 3, 3,
                        device=k.device, dtype=k.dtype)
        K[..., 0, 1] = -k[..., 2]
        K[..., 0, 2] = k[..., 1]
        K[..., 1, 0] = k[..., 2]
        K[..., 1, 2] = -k[..., 0]
        K[..., 2, 0] = -k[..., 1]
        K[..., 2, 1] = k[..., 0]
 
        I = torch.eye(3, device=k.device).expand_as(K)
        theta = theta.unsqueeze(-1)
        R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        return R
 
    def get_transform(self, slice_idx):
        """
        取得指定切片的剛體變換 (R, t)
 
        Args:
            slice_idx: 切片索引 (可以是單個整數或 tensor)
        Returns:
            R: (3, 3) 旋轉矩陣
            t: (3,) 平移向量
        """
        params = self.slice_transforms[slice_idx]  # (6,)
        R = self.axis_angle_to_matrix(
            params[:3].unsqueeze(0)
        ).squeeze(0)
        t = params[3:]
        return R, t
 
    def sample_psf_points(self, pixel_coords, slice_idx,
                          psf_sigma, K):
        """
        從 PSF 定義的高斯分佈中生成蒙特卡羅取樣點
 
        Args:
            pixel_coords: (N, 3) 切片座標系中的像素位置 p_ij
            slice_idx: (N,) 每個像素對應的切片索引
            psf_sigma: (3,) 或 (n_slices, 3) PSF 各軸標準差
            K: 取樣數
        Returns:
            x_samples: (N, K, 3) 3D 空間中的取樣點
        """
        N = pixel_coords.shape[0]
        device = pixel_coords.device
        per_slice = (psf_sigma.dim() == 2)

        x_samples = torch.zeros(N, K, 3, device=device)

        unique_slices = torch.unique(slice_idx)
        for s in unique_slices:
            mask = (slice_idx == s)
            n_masked = mask.sum()
            R, t = self.get_transform(s)

            # 選取此切片的 PSF sigma
            sigma_s = psf_sigma[s] if per_slice else psf_sigma

            # u ~ N(0, diag(sigma^2))
            u = torch.randn(n_masked, K, 3, device=device)
            u = u * sigma_s

            # 切片座標系中的取樣點
            local = u + pixel_coords[mask].unsqueeze(1)

            # 應用剛體變換到 3D 空間
            x_samples[mask] = (local @ R.T) + t.unsqueeze(0)

        return x_samples
 
    def forward(self, pixel_coords, slice_idx, psf_sigma):
        """
        前向傳播：從像素座標計算模擬像素強度及方差
 
        實現論文公式 (7) 和 (8)：
          E[I_ij] = (C_i/K) Σ_k B_i(x_ijk) V(x_ijk)
          var(I_ij) = (C_i²/K) Σ_k M(x_ijk) B²(x_ijk) σ²(x_ijk)
 
        Args:
            pixel_coords: (N, 3) 像素在切片座標系中的位置
            slice_idx: (N,) 切片索引
            psf_sigma: (3,) PSF 標準差
 
        Returns:
            mean_I: (N,) 模擬像素強度的期望值
            total_var: (N,) 像素強度的總方差
        """
        K = self.config.n_samples
        N = pixel_coords.shape[0]
 
        # Step 1: PSF 取樣
        x_samples = self.sample_psf_points(
            pixel_coords, slice_idx, psf_sigma, K
        )  # (N, K, 3)
 
        # Step 2: 將取樣點歸一化到 [0, 1]（基於 ROI 範圍）
        x_flat = x_samples.reshape(-1, 3)  # (N*K, 3)
        roi_extent = (self.roi_max - self.roi_min).clamp(min=1e-8)
        x_norm = ((x_flat - self.roi_min) / roi_extent).clamp(0.0, 1.0)

        # Step 3: 雜湊網格編碼
        full_enc = self.encoding(x_norm)  # (N*K, enc_dim)
        low_enc = self.encoding.get_low_level_encoding(
            x_norm, self.config.bias_levels
        )  # (N*K, low_dim)
 
        # Step 4: 體積網路
        volume, features = self.volume_net(full_enc)
        # volume: (N*K, 1), features: (N*K, feat_dim)
 
        # Step 5: 準備切片嵌入
        # 將每個像素的切片嵌入擴展到 K 個取樣點
        emb = self.slice_embeddings[slice_idx]  # (N, emb_dim)
        emb_expanded = emb.unsqueeze(1).expand(
            -1, K, -1
        ).reshape(-1, emb.shape[-1])  # (N*K, emb_dim)
 
        # Step 6: 偏場網路
        bias = self.bias_net(low_enc, emb_expanded)  # (N*K, 1)
 
        # Step 7: 方差網路
        pixel_var = self.variance_net(
            features, emb_expanded
        )  # (N*K, 1)
 
        # Step 8: 計算像素強度的期望值
        # E[I_ij] = (C_i/K) Σ_k B_i(x_ijk) * V(x_ijk)
        C = self.get_scaling_factors()  # (n_slices,)
        C_pixels = C[slice_idx]  # (N,)
 
        bv = (bias * volume).reshape(N, K)  # (N, K)
        mean_I = C_pixels * bv.mean(dim=1)  # (N,)
 
        # Step 9: 計算像素級方差
        b2_sigma2 = (bias**2 * pixel_var).reshape(N, K)
        pixel_level_var = C_pixels**2 * b2_sigma2.mean(dim=1)
 
        # Step 10: 加上切片級方差
        slice_var = torch.exp(
            self.log_slice_variance[slice_idx]
        )
        total_var = pixel_level_var + slice_var  # (N,)

        # 正則化所需的中間量
        reg_data = {
            'volume_values': volume.squeeze(-1).reshape(N, K),  # (N, K)
            'sample_coords': x_samples,                          # (N, K, 3)
            'log_bias': torch.log(bias.clamp(min=1e-8)).squeeze(-1),  # (N*K,)
        }

        return mean_I, total_var, reg_data
 
    @torch.no_grad()
    def sample_volume(self, grid_coords):
        """
        推論時取樣重建體積

        Args:
            grid_coords: (M, 3) 世界座標（自動正規化至 [0,1]）
        Returns:
            (M,) 體積強度
        """
        roi_extent = (self.roi_max - self.roi_min).clamp(min=1e-8)
        coords_norm = ((grid_coords - self.roi_min) / roi_extent).clamp(0.0, 1.0)
        enc = self.encoding(coords_norm)
        volume, _ = self.volume_net(enc)
        return volume.squeeze(-1)
