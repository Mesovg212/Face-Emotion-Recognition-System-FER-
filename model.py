from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        feature_dim: int = 512,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
        pretrained_cnn: bool = False,
        freeze_cnn: bool = False,
    ):
        super().__init__()

        # CNN backbone (ResNet-18)
        # Use weights=None to avoid internet downloads by default
        if pretrained_cnn:
            try:
                self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                self.cnn = models.resnet18(weights=None)
        else:
            self.cnn = models.resnet18(weights=None)

        self.cnn.fc = nn.Identity()
        cnn_out_dim = feature_dim  # resnet18 -> 512

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # LSTM head
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)  # [B*T, 512]
        feats = feats.view(B, T, -1)  # [B, T, 512]

        out, (h_n, c_n) = self.lstm(feats)  # out: [B, T, H]
        # take last layer's hidden state at final time step
        last_hidden = out[:, -1, :]  # [B, H]
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)  # [B, num_classes]
        return logits