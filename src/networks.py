import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class SimpleMLP(nn.Module):
    """通用的多層感知機，用於建構三個分支網路"""
 
    def __init__(self, in_dim, hidden_dim, out_dim,
                 n_hidden=1, out_activation=None):
        """
        Args:
            in_dim: 輸入維度
            hidden_dim: 隱藏層神經元數 (論文使用 64)
            out_dim: 輸出維度
            n_hidden: 隱藏層數 (論文使用 1)
            out_activation: 輸出激活函數
                           "softplus" / "exp" / None
        """
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_activation = out_activation
 
    def forward(self, x):
        out = self.net(x)
        if self.out_activation == "softplus":
            return F.softplus(out)
        elif self.out_activation == "exp":
            return torch.exp(out)
        return out
 
 
class VolumeNetwork(nn.Module):
    """
    體積強度網路 MLP_V
 
    輸入：全層級雜湊網格編碼 φ(x)
    輸出：V(x) 體積強度 (1D) + z(x) 特徵向量
 
    重要：不接收切片嵌入，因為我們希望 V(x) 只學習
    切片無關的底層體積強度。
    """
 
    def __init__(self, encoding_dim, hidden_dim=64,
                 n_hidden=1, feature_dim=16):
        super().__init__()
        self.feature_dim = feature_dim
        # 輸出 = 1 (V) + feature_dim (z)
        self.mlp = SimpleMLP(
            in_dim=encoding_dim,
            hidden_dim=hidden_dim,
            out_dim=1 + feature_dim,
            n_hidden=n_hidden
        )
 
    def forward(self, encoding):
        """
        Args:
            encoding: (N, encoding_dim) 雜湊網格編碼
        Returns:
            volume: (N, 1) 體積強度 (softplus 確保非負)
            features: (N, feature_dim) 特徵向量
        """
        out = self.mlp(encoding)
        volume = F.softplus(out[:, :1])  # 強度必須 >= 0
        features = out[:, 1:]
        return volume, features
 
 
class BiasFieldNetwork(nn.Module):
    """
    偏場網路 MLP_B
 
    輸入：低層級編碼 φ_{1:b}(x) + 切片嵌入 e_i
    輸出：B_i(x) 偏場值 (通過 exp 確保正值)
 
    設計要點：
      - 只用低頻編碼，防止偏場學到高頻細節
      - 切片嵌入提供切片特定資訊
    """
 
    def __init__(self, low_encoding_dim, embedding_dim,
                 hidden_dim=64, n_hidden=1):
        super().__init__()
        self.mlp = SimpleMLP(
            in_dim=low_encoding_dim + embedding_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            n_hidden=n_hidden,
            out_activation="exp"  # B > 0
        )
    def forward(self, low_encoding, slice_embedding):
        """
        Args:
            low_encoding: (N, low_dim) 低層級編碼
            slice_embedding: (N, emb_dim) 切片嵌入
        Returns:
            (N, 1) 偏場值
        """
        x = torch.cat([low_encoding, slice_embedding], dim=-1)
        return self.mlp(x)
 
 
class VarianceNetwork(nn.Module):
    """
    方差網路 MLP_σ
 
    輸入：體積網路的特徵向量 z(x) + 切片嵌入 e_i
    輸出：σ²_i(x) 像素級方差 (通過 exp 確保正值)
    """
 
    def __init__(self, feature_dim, embedding_dim,
                 hidden_dim=64, n_hidden=1):
        super().__init__()
        self.mlp = SimpleMLP(
            in_dim=feature_dim + embedding_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            n_hidden=n_hidden,
            out_activation="exp"  # σ² > 0
        )
 
    def forward(self, features, slice_embedding):
        """
        Args:
            features: (N, feat_dim) 體積網路輸出的特徵
            slice_embedding: (N, emb_dim) 切片嵌入
        Returns:
            (N, 1) 像素級方差
        """
        x = torch.cat([features, slice_embedding], dim=-1)
        return self.mlp(x)

 
