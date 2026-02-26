import torch
 
 
def reconstruction_loss(predicted_mean, predicted_var,
                        target_intensity):
    """
    切片重建損失 (負對數似然)
 
    L_ij = (I_ij - Ī_ij)² / (2σ²_ij) + ½ log(σ²_ij)
 
    第一項：重建誤差，方差作為權重（方差大 = 權重小 = 忽略離群值）
    第二項：防止方差無限增大的正則化
 
    Args:
        predicted_mean: (N,) 模擬像素強度 Ī_ij
        predicted_var: (N,) 預測方差 σ²_ij
        target_intensity: (N,) 實際像素強度 I_ij
    Returns:
        標量損失值
    """
    # 固定方差下限避免數值不穩定
    var_clamped = predicted_var.clamp(min=1e-6)
 
    residual = target_intensity - predicted_mean
    loss = residual**2 / (2 * var_clamped) \
         + 0.5 * torch.log(var_clamped)
    return loss.mean()
 
 
def image_regularization(volume_values, sample_coords, K):
    """
    影像正則化 (各向同性全變差 Total Variation)
 
    利用蒙特卡羅取樣點配對近似方向導數：
      R_V ≈ (2/K|B|) Σ Σ |V(x_k) - V(x_l)| / ||x_k - x_l||
 
    技巧：不需要額外的前向/反向傳播，計算開銷極小
 
    Args:
        volume_values: (N, K) 取樣點上的體積強度
        sample_coords: (N, K, 3) 取樣點座標
        K: 取樣數
    Returns:
        正則化損失
    """
    half_K = K // 2
    v1 = volume_values[:, :half_K]  # 前半
    v2 = volume_values[:, half_K:2*half_K]  # 後半
    x1 = sample_coords[:, :half_K]
    x2 = sample_coords[:, half_K:2*half_K]
 
    # 方向導數近似
    diff_v = torch.abs(v1 - v2)
    diff_x = torch.norm(x1 - x2, dim=-1).clamp(min=1e-8)
    return (diff_v / diff_x).mean()
 
 
def bias_regularization(log_bias_values):
    """
    偏場正則化：約束對數偏場的均值為零
 
    R_B = (mean(log B_i(x)))^2
 
    Args:
        log_bias_values: (M,) 所有取樣點的偏場對數值
    Returns:
        正則化損失
    """
    return log_bias_values.mean() ** 2
