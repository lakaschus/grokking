import pytest
import torch
import types
from torch.nn.utils.rnn import pad_sequence

from training import (
    train_classification,
    get_eq_positions,
    create_mask_from_positions,
    generate_flat_tensors,
    mask_tensor,
    get_criterion,
)
from model import Transformer
from data import BINARY_TOKENS


# Test that train and val data are completely distinct
# TODO
