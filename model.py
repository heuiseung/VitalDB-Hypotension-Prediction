"""
저혈압 조기 예측 모델 정의.

- HypotensionModel: 1D-CNN + LSTM 기반 시계열 이진 분류 (메인 모델).
- HypoNet: MLP 기반 이진 분류 (기존 호환/대조용).
"""
import torch
import torch.nn as nn


class HypotensionModel(nn.Module):
    """1D-CNN + LSTM 기반 저혈압 이진 분류 모델.

    시계열 특성(또는 단일 시점 특성)을 1D-CNN으로 국소 패턴 추출 후,
    LSTM으로 시간 의존성을 반영하고, 마지막 hidden state로 이진 logits 출력.
    입력 shape: (batch, num_features) 또는 (batch, seq_len, num_features).

    Attributes:
        in_dim: 입력 특성 수 (feature dimension).
    """

    def __init__(
        self,
        in_dim: int,
        conv_channels: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """HypotensionModel 초기화.

        Args:
            in_dim: 입력 특성 차원 (feature 개수).
            conv_channels: 1D-CNN 출력 채널 수. 국소 패턴 수 증가 시 용량 up.
            lstm_hidden: LSTM hidden size. 시계열 표현력과 과적합 트레이드오프.
            lstm_layers: LSTM 층 수. 2 이상이면 layer 간 dropout 적용.
            dropout: Conv/LSTM 후 dropout 비율. 과적합 완화용.
        """
        super().__init__()
        self.in_dim = in_dim
        # 1D-CNN: (B, 1, F) → (B, conv_channels, F). 채널 방향으로 패턴 추출.
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # LSTM 입력: (B, F, C). seq_len=F, input_size=conv_channels로 시간 축 따라 처리.
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파: 특성 → CNN → LSTM → FC → logits (배치당 스칼라).

        Args:
            x: (B, F) 또는 (B, T, F). 2D면 단일 시점으로 간주하고 channel 차원만 추가.

        Returns:
            (B,) shape logits. BCEWithLogitsLoss 등과 함께 사용.
        """
        # 2D 입력이면 (B, F) → (B, 1, F)로 channel 차원 추가 (CNN 입력 형식)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # (B, 1, F) → conv → (B, C, F)
        x = self.conv(x)
        # LSTM은 (seq_len, batch, input_size): (B, F, C)로 permute
        x = x.permute(0, 2, 1)
        out, (h_n, _) = self.lstm(x)
        # 마지막 층의 마지막 시점 hidden만 사용 (시계열 요약)
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


class HypotensionModelV2(nn.Module):
    """개선된 1D-CNN + BiLSTM: 2단 CNN, 양방향 LSTM, 2층 FC 헤드.

    전체 데이터 학습 시 표현력·일반화 향상을 위해 채널 확장(32→64), BiLSTM, FC 128→32→1 적용.
    """

    def __init__(
        self,
        in_dim: int,
        conv_channels: tuple = (32, 64),
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        fc_hidden: int = 32,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        # 2단 1D-CNN: (B,1,F) -> (B,32,F) -> (B,64,F). BatchNorm으로 학습 안정화.
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # BiLSTM: forward+backward 결합으로 시계열 양방향 맥락 활용
        self.lstm = nn.LSTM(
            input_size=conv_channels[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )
        lstm_out = lstm_hidden * 2  # bidirectional
        self.fc = nn.Sequential(
            nn.Linear(lstm_out, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, (h_n, _) = self.lstm(x)
        # 양방향이므로 마지막 층의 forward/backward hidden 결합
        last_forward = h_n[-2]
        last_backward = h_n[-1]
        last_hidden = torch.cat([last_forward, last_backward], dim=1)
        logits = self.fc(last_hidden).squeeze(-1)
        return logits


class HypoNet(nn.Module):
    """저혈압 이진 분류용 MLP (기존 실험/대조용).

    단순 fully-connected 층으로만 구성. 시계열 구조 없이 시점별 특성만 사용할 때 활용.
    """

    def __init__(self, in_dim: int) -> None:
        """HypoNet 초기화.

        Args:
            in_dim: 입력 특성 차원.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파: x → MLP → logits (배치당 스칼라).

        Args:
            x: (B, in_dim) 특성 텐서.

        Returns:
            (B,) shape logits.
        """
        return self.net(x).squeeze(-1)
