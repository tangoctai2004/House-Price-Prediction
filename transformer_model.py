"""Tabular Transformer model for house-price regression.

The model receives two inputs:
- numeric_values: normalized numeric columns, shape (batch, num_numeric)
- categorical_ids: integer ids for text/category columns, shape (batch, num_cat)
"""

from __future__ import annotations

import torch
from torch import nn


class HousePriceTransformer(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        categorical_cardinalities: list[int],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.num_numeric = num_numeric
        self.num_categorical = len(categorical_cardinalities)
        self.d_model = d_model

        self.numeric_weight = nn.Parameter(torch.randn(num_numeric, d_model) * 0.02)
        self.numeric_bias = nn.Parameter(torch.zeros(num_numeric, d_model))

        # BƯỚC 5: Đưa các mã số của cột chữ qua lớp Embedding để chuyển đổi thành vector dense
        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in categorical_cardinalities]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regression_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, numeric_values: torch.Tensor, categorical_ids: torch.Tensor) -> torch.Tensor:
        # BƯỚC 7: Đưa dữ liệu đã xử lý vào Transformer.
        # - Chiếu các cột số lên không gian d_model làm numeric tokens.
        numeric_tokens = numeric_values.unsqueeze(-1) * self.numeric_weight + self.numeric_bias

        # - Chuyển các chỉ mục phân loại thành embedding tokens.
        categorical_tokens = []
        for idx, embedding in enumerate(self.categorical_embeddings):
            categorical_tokens.append(embedding(categorical_ids[:, idx]))
        categorical_tokens = torch.stack(categorical_tokens, dim=1)

        # - Gộp chung hai loại tokens lại để làm đầu vào cho mô hình Transformer.
        tokens = torch.cat([numeric_tokens, categorical_tokens], dim=1)
        
        # - Transformer tự tạo Q, K, V và cơ chế Attention để học quan hệ chéo giữa các thông tin.
        encoded = self.transformer(tokens)

        # - Tính trung bình các tokens (pooling) để đưa qua regression head dự đoán giá tạm thời.
        pooled = encoded.mean(dim=1)
        return self.regression_head(pooled).squeeze(-1)

