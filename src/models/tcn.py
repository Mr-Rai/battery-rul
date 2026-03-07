"""
Temporal Convolutional Network (TCN) for battery RUL estimation.

Why TCN over LSTM:
- Parallelizable across time steps (LSTMs are sequential)
- No vanishing gradient problem — residual connections throughout
- Receptive field is explicit and controllable via dilation factors
- On degradation sequences (100-500 cycles), TCN trains 3-5x faster
  than comparable LSTM with similar accuracy

Architecture:
- Input: sequence of cycle feature vectors, shape (batch, seq_len, n_features)
- Dilated causal convolutions with exponentially increasing dilation (1,2,4,8,...)
- Residual blocks with weight normalization
- Output: single scalar RUL prediction per sequence

Reference: Bai et al. (2018) "An Empirical Evaluation of Generic Convolutional
and Recurrent Networks for Sequence Modeling" — arXiv:1803.01271
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import mlflow
import logging

logger = logging.getLogger(__name__)


class CausalConv1d(nn.Module):
    """
    Causal convolution: output at time t depends only on inputs up to time t.
    Achieved by left-padding the input by (kernel_size - 1) * dilation.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        # TODO: implement
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        raise NotImplementedError


class TCNResidualBlock(nn.Module):
    """
    Residual block with two dilated causal convolutions.
    Weight normalization applied to both conv layers.
    Dropout applied after each activation.
    Residual connection: if in_channels != out_channels, use 1x1 conv to match.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        # TODO: implement
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        raise NotImplementedError


class TCN(nn.Module):
    """
    Full TCN: stack of residual blocks with exponentially increasing dilation.

    Receptive field = 1 + 2 * (kernel_size - 1) * sum(dilations)
    Must cover the full sequence length to capture long-range degradation trends.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        kernel_size: int = 3,
        num_layers: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        # TODO: implement — dilation doubles each layer: [1, 2, 4, 8, 16, 32]
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        raise NotImplementedError


class BatterySequenceDataset(Dataset):
    """
    PyTorch Dataset for battery degradation sequences.

    Each sample: (sequence of last `seq_len` cycle feature vectors, RUL target)
    Sequences are constructed with a sliding window over each cell's history.
    Cells are kept separate — no sequence crosses cell boundaries.
    """

    def __init__(self, features: pd.DataFrame, targets: pd.Series, seq_len: int = 30):
        # TODO: implement
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO: implement
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: implement
        raise NotImplementedError


def train(
    model: TCN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
) -> TCN:
    """
    Train TCN with early stopping and MLflow logging.

    Args:
        model: Initialized TCN.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training config (lr, epochs, patience, etc.)

    Returns:
        Best model checkpoint (lowest val RMSE).
    """
    # TODO: implement
    raise NotImplementedError
