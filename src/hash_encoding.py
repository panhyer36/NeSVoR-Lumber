import torch
import torch.nn as nn
import numpy as np
 
 
class HashGridEncoding(nn.Module):
    """
    多解析度雜湊網格編碼 (Multi-Resolution Hash Grid Encoding)
 
    原理：
      1. 將 3D 空間劃分為 L 個層級的網格，從粗到細
      2. 在每個層級上，找到包圍查詢點的 8 個頂點
      3. 將頂點索引通過雜湊函數映射到雜湊表
      4. 使用三線性插值得到特徵向量
      5. 拼接所有層級的特徵作為最終編碼
    """
 
    def __init__(self, n_levels=12, n_features=2,
                 log2_hashmap_size=19, base_resolution=16,
                 growth_factor=1.38):
        super().__init__()
        self.n_levels = n_levels
        self.n_features = n_features
        self.hashmap_size = 2 ** log2_hashmap_size
 
        # 計算每個層級的網格解析度
        self.resolutions = []
        for l in range(n_levels):
            res = int(np.floor(base_resolution * (growth_factor ** l)))
            self.resolutions.append(res)
 
        # 為每個層級創建雜湊表
        # 每個雜湊表儲存 hashmap_size 個特徵向量
        self.hash_tables = nn.ParameterList([
            nn.Parameter(
                torch.empty(self.hashmap_size, n_features)
                     .uniform_(-1e-4, 1e-4)  # 論文推薦的初始化
            )
            for _ in range(n_levels)
        ])
 
        # 雜湊函數中使用的大質數
        self.primes = torch.tensor([1, 2654435761, 805459861],
                                   dtype=torch.long)

    @property
    def output_dim(self):
        """編碼輸出的總維度 = n_levels * n_features"""
        return self.n_levels * self.n_features
 
    def _hash(self, coords_grid, level):
        """
        雜湊函數：將 3D 網格索引映射到雜湊表索引
 
        公式： h(i,j,k) = (i XOR j*π1 XOR k*π2) mod T
        其中 π1, π2 是大質數，T 是雜湊表大小
 
        Args:
            coords_grid: (N, 3) 整數網格座標
            level: 層級索引
        Returns:
            (N,) 雜湊表索引
        """
        primes = self.primes.to(coords_grid.device)
        # XOR 運算實現雜湊
        hashed = coords_grid[:, 0] * primes[0]
        hashed = hashed ^ (coords_grid[:, 1] * primes[1])
        hashed = hashed ^ (coords_grid[:, 2] * primes[2])
        return hashed % self.hashmap_size
 
    def _trilinear_interp(self, x_normalized, level):
        """
        在指定層級的網格上執行三線性插值
 
        步驟：
          1. 將歸一化座標 [0,1] 縮放到網格解析度
          2. 找到包圍的 8 個頂點的網格索引
          3. 查詢雜湊表得到特徵
          4. 計算三線性插值權重並加權平均
 
        Args:
            x_normalized: (N, 3) 歸一化到 [0,1] 的座標
            level: 層級索引
        Returns:
            (N, n_features) 插值後的特徵向量
        """
        res = self.resolutions[level]
        table = self.hash_tables[level]
 
        # 縮放到網格座標
        x_scaled = x_normalized * res  # (N, 3)
 
        # 下界網格索引 (floor)
        x_floor = torch.floor(x_scaled).long()  # (N, 3)
 
        # 插值權重 (小數部分)
        w = x_scaled - x_floor.float()  # (N, 3)
 
        # 建構 8 個頂點的偏移: (8, 3)
        offsets = torch.tensor([
            [0,0,0], [1,0,0], [0,1,0], [1,1,0],
            [0,0,1], [1,0,1], [0,1,1], [1,1,1]
        ], device=x_normalized.device, dtype=torch.long)
 
        # 8 個頂點的網格座標: (N, 8, 3)
        corners = x_floor.unsqueeze(1) + offsets.unsqueeze(0)
 
        # 雜湊查表: (N, 8) -> (N, 8, n_features)
        N_pts = x_normalized.shape[0]
        corner_flat = corners.reshape(-1, 3)  # (N*8, 3)
        hash_idx = self._hash(corner_flat, level)  # (N*8,)
        features = table[hash_idx].reshape(N_pts, 8, -1)
 
        # 三線性插值權重計算
        # c000*(1-wx)(1-wy)(1-wz) + c100*wx*(1-wy)(1-wz) + ...
        wx, wy, wz = w[:, 0:1], w[:, 1:2], w[:, 2:3]  # (N,1)
 
        # 8 個頂點的權重
        weights = torch.stack([
            (1-wx)*(1-wy)*(1-wz),  # c000
            wx*(1-wy)*(1-wz),      # c100
            (1-wx)*wy*(1-wz),      # c010
            wx*wy*(1-wz),          # c110
            (1-wx)*(1-wy)*wz,      # c001
            wx*(1-wy)*wz,          # c101
            (1-wx)*wy*wz,          # c011
            wx*wy*wz,              # c111
        ], dim=1)  # (N, 8, 1)
 
        # 加權求和
        return (weights * features).sum(dim=1)  # (N, n_features)
 
    def forward(self, x):
        """
        前向傳播：將 3D 座標編碼為高維特徵向量
 
        Args:
            x: (N, 3) 歸一化座標 [0, 1]
        Returns:
            (N, n_levels * n_features) 編碼後的特徵
        """
        features = []
        for l in range(self.n_levels):
            feat = self._trilinear_interp(x, l)
            features.append(feat)
        return torch.cat(features, dim=-1)
 
    def get_low_level_encoding(self, x, n_levels):
        """
        取得前 n_levels 個層級的低頻編碼
        用於偏場網路，僅提供低頻資訊
 
        Args:
            x: (N, 3) 歸一化座標
            n_levels: 使用的層級數
        Returns:
            (N, n_levels * n_features)
        """
        features = []
        for l in range(min(n_levels, self.n_levels)):
            feat = self._trilinear_interp(x, l)
            features.append(feat)
        return torch.cat(features, dim=-1)
