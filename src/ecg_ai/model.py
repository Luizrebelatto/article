from __future__ import annotations


def build_model(n_leads: int = 12):
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 9, stride=stride, padding=4, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, 9, padding=4, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels and stride == 1
                else nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels),
                )
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.body(x) + self.skip(x))

    class FutureAFCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_leads, 32, 15, stride=2, padding=7, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                ResidualBlock(32, 64, 2),
                ResidualBlock(64, 128, 2),
                ResidualBlock(128, 256, 2),
                nn.AdaptiveAvgPool1d(1),
            )
            self.dropout = nn.Dropout(0.3)
            self.head = nn.Linear(256, 1)

        def forward(self, x):
            return self.head(self.dropout(self.features(x).squeeze(-1))).squeeze(-1)

    return FutureAFCNN()
