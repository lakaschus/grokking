import pytest
import torch
import types
from torch.nn.utils.rnn import pad_sequence

from training import (
    get_eq_positions,
    create_mask_from_positions,
    generate_flat_tensors,
    mask_tensor,
    get_criterion,
)
from data import BINARY_TOKENS


@pytest.fixture
def config_sequence():
    return types.SimpleNamespace(
        task_type="sequence",
        num_steps=100,
        learning_rate=1e-3,
        weight_decay=1e-5,
        num_layers=2,
        dim_model=128,
        num_heads=4,
        operation="x+y_binary",
        training_fraction=0.8,
        batch_size=32,
        curriculum="random",
    )


@pytest.fixture(scope="session")
def two_dim_tensor():
    eq = BINARY_TOKENS["="]
    pl = BINARY_TOKENS["+"]
    eos = BINARY_TOKENS["<EOS>"]
    two_dim_tensor = torch.tensor(
        [[1, 0, pl, 1, eq, 1, 1, eos], [0, pl, 0, eq, 0, eos, eos, eos]],
        device=torch.device("cuda"),
    )  # Batch Size * Sequence Length
    return two_dim_tensor


@pytest.fixture(scope="session")
def three_dim_tensor():
    # 2 * 3 * 4
    three_dim_tensor = torch.tensor(
        [
            [[-1, 0, 0, 10], [10, 0, 0, -1], [-1, 1, 1, 10]],
            [[-1, 1, 10, 1], [-1, 1, 0, 10], [-1, 10, -1, 1]],
        ],
        device=torch.device("cuda"),
    )  # Batch Size * Sequence Length * Tokens
    # convert to float
    three_dim_tensor = three_dim_tensor.float()
    return three_dim_tensor


@pytest.fixture(scope="session")
def two_dim_labels():
    # 2 * 3
    two_dim_labels = torch.tensor(
        [
            [3, 0, 3],
            [2, 3, 1],
        ],
        device=torch.device("cuda"),
    )  # Batch Size * Sequence Length
    return two_dim_labels


def test_get_eq_positions(two_dim_tensor):
    eq_pos = get_eq_positions(two_dim_tensor)
    assert eq_pos.tolist() == [3, 2]


def test_create_mask_from_positions():
    # corresponds to mask at token 5
    eq_pos = torch.tensor([4, 3], device=torch.device("cuda"))
    tens = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9]],
        device=torch.device("cuda"),
    )
    mask = create_mask_from_positions(
        tens.shape[1], tens.shape[0], eq_pos, torch.device("cuda")
    )
    assert tens[mask].tolist() == [6, 7, 8, 6, 7, 8, 9]


def test_generate_flat_tensors(two_dim_tensor, three_dim_tensor):
    flat_2d_tensor = generate_flat_tensors(two_dim_tensor)
    assert flat_2d_tensor.tolist() == [1, 0, 2, 1, 3, 1, 1, 4, 0, 2, 0, 3, 0, 4, 4, 4]
    # Now with three dimensional tensor
    # (batch, seq, token) -> (batch * seq, token)
    flat_3d_tensor = generate_flat_tensors(three_dim_tensor)
    assert flat_3d_tensor.tolist() == [
        [-1, 0, 0, 10],
        [10, 0, 0, -1],
        [-1, 1, 1, 10],
        [-1, 1, 10, 1],
        [-1, 1, 0, 10],
        [-1, 10, -1, 1],
    ]


def test_mask_tensor(two_dim_tensor, three_dim_tensor):
    eq_pos = torch.tensor([4, 3], device=torch.device("cuda"))
    mask = create_mask_from_positions(8, 2, eq_pos, torch.device("cuda"))
    masked_tensor = mask_tensor(two_dim_tensor, mask)
    assert masked_tensor.tolist() == [1, 1, 4, 0, 4, 4, 4]
    eq_pos2 = torch.tensor([0, 1], device=torch.device("cuda"))
    mask2 = create_mask_from_positions(3, 2, eq_pos2, torch.device("cuda"))
    masked_tensor_3d = mask_tensor(three_dim_tensor, mask2)
    assert masked_tensor_3d.int().tolist() == [
        [10, 0, 0, -1],
        [-1, 1, 1, 10],
        [-1, 10, -1, 1],
    ]


def test_get_criterion(config_sequence, three_dim_tensor, two_dim_labels):
    criterion = get_criterion(config_sequence)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    loss1 = criterion(three_dim_tensor.transpose(1, 2), two_dim_labels)
    assert loss1 < 0.1
    loss2 = criterion(three_dim_tensor.transpose(1, 2), two_dim_labels * 0)
    assert loss2 > 5
    loss1_flat = criterion(
        generate_flat_tensors(three_dim_tensor), generate_flat_tensors(two_dim_labels)
    )
    loss2_flat = criterion(
        generate_flat_tensors(three_dim_tensor),
        generate_flat_tensors(two_dim_labels * 0),
    )
    # assert abs(loss1_flat - loss2_flat) < 0.1
    assert abs(loss1 - loss1_flat) < 0.01
    assert abs(loss2 - loss2_flat) < 0.01


def test_pad_sequence():
    a = torch.ones(3, 2)
    b = torch.ones(2, 2)
    c = torch.ones(1, 2)
    padded_seq = pad_sequence([a, b, c], padding_value=0)
    assert padded_seq.size() == (3, 3, 2)
    assert padded_seq[0, 0, 0] == 1
    assert padded_seq[1, -1, 0] == 0
