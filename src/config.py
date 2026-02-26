from dataclasses import dataclass
import torch
 
@dataclass
class NeSVoRConfig:
    """NeSVoR 全域超參數配置"""
 
    # === 雜湊網格編碼 (Hash Grid Encoding) ===
    n_levels: int = 12           # 網格層級數 L
    n_features: int = 2          # 每個頂點的特徵維度 F
    log2_hashmap_size: int = 19  # 雜湊表大小 = 2^19
    base_resolution: int = 16    # 最粗網格大小 N1
    growth_factor: float = 1.38  # 網格縮放因子 s
 
    # === MLP 網路 ===
    hidden_dim: int = 64         # MLP 隱藏層神經元數
    n_hidden_layers: int = 1     # MLP 隱藏層數
    embedding_dim: int = 16      # 切片嵌入向量維度
    bias_levels: int = 4         # 偏場網路使用的編碼層級數 b
 
    # === 訓練 ===
    n_iterations: int = 6000     # 總迭代次數
    batch_size: int = 4096       # 每次迭代的像素數
    n_samples: int = 128         # 蒙特卡羅取樣數 K
    lr: float = 5e-3             # 初始學習率
    lr_decay: float = 1/3        # 學習率衰減因子
 
    # === 正則化 ===
    lambda_v: float = 0.5        # 影像正則化權重
    lambda_b: float = 1.0        # 偏場正則化權重
 
    # === 裝置 ===
    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
