# tests/test_training.py

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock
from typing import Dict, Any, Tuple, List

from training import (
    train_classification,
    train_sequence,
    compute_sequence_loss_and_accuracy,
    evaluate_classification,
    evaluate_sequence,
    get_criterion,
    save_best_model,
    create_checkpoint,
)
from model import Transformer
from data import BINARY_TOKENS


@pytest.fixture
def mock_config_classification():
    return {
        "task_type": "classification",
        "num_steps": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1,
        "num_layers": 2,
        "dim_model": 128,
        "num_heads": 4,
        "operation": "x+y",
        "prime": 10,
        "training_fraction": 0.8,
        "batch_size": 32,
        "curriculum": "random",
    }


@pytest.fixture
def mock_config_sequence():
    return {
        "task_type": "sequence",
        "num_steps": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1,
        "num_layers": 2,
        "dim_model": 128,
        "num_heads": 4,
        "operation": "x+y_binary",
        "prime": 4,
        "training_fraction": 0.8,
        "batch_size": 32,
        "curriculum": "random",
    }


@pytest.fixture
def mock_model():
    return MagicMock(spec=Transformer)


@pytest.fixture
def mock_optimizer():
    return MagicMock()


@pytest.fixture
def mock_scheduler():
    return MagicMock()


@pytest.fixture
def classification_batch():
    # Mock inputs and labels for classification
    inputs = torch.randint(0, 10, (32, 5))  # (batch_size, seq_len)
    labels = torch.randint(0, 10, (32,))  # (batch_size,)
    return inputs, labels


@pytest.fixture
def sequence_batch():
    # Mock inputs and labels for sequence
    inputs = torch.randint(0, 5, (32, 10))  # (batch_size, seq_len)
    labels = torch.randint(
        0, 5, (32,)
    )  # Not used in sequence, but included for completeness
    return inputs, labels


def test_train_classification(mock_model, classification_batch):
    inputs, labels = classification_batch
    criterion = torch.nn.CrossEntropyLoss()

    # Mock model output
    mock_output = torch.randn(5, 32, 10)  # (seq_len, batch_size, num_tokens)
    mock_model.return_value = mock_output

    output, loss, acc = train_classification(mock_model, inputs, labels, criterion)

    # Assertions
    mock_model.assert_called_once_with(inputs)
    assert loss.shape == torch.Size([])
    assert isinstance(acc, torch.Tensor)
    assert 0.0 <= acc.item() <= 1.0


def test_train_sequence(mock_model, sequence_batch, mock_config_sequence):
    inputs, labels = sequence_batch
    criterion = torch.nn.CrossEntropyLoss(ignore_index=BINARY_TOKENS["<EOS>"])

    # Mock model output
    mock_output = torch.randn(32, 9, 5)  # [batch_size, seq_len -1, num_tokens]
    mock_model.return_value = mock_output

    loss, acc = compute_sequence_loss_and_accuracy(
        mock_output,
        inputs[:, 1:],
        (inputs == BINARY_TOKENS["="]).nonzero(as_tuple=True)[1],
        criterion,
        mock_config_sequence,
    )

    # Assertions
    assert loss.shape == torch.Size([])
    assert isinstance(acc, torch.Tensor)
    assert 0.0 <= acc.item() <= 1.0


def test_evaluate_classification(mock_model, classification_batch):
    inputs, labels = classification_batch
    criterion = torch.nn.CrossEntropyLoss()

    # Mock model output
    mock_output = torch.randn(5, 32, 10)
    mock_model.return_value = mock_output

    loss, correct, samples = evaluate_classification(
        mock_model, inputs, labels, criterion
    )

    # Assertions
    mock_model.assert_called_once_with(inputs)
    assert isinstance(loss, float)
    assert isinstance(correct, float)
    assert samples == 32


def test_evaluate_sequence(mock_model, sequence_batch, mock_config_sequence):
    inputs, labels = sequence_batch
    criterion = torch.nn.CrossEntropyLoss(ignore_index=BINARY_TOKENS["<EOS>"])

    # Mock model output
    mock_output = torch.randn(32, 9, 5)  # [batch_size, seq_len -1, num_tokens]
    mock_model.return_value = mock_output

    loss, correct, samples = evaluate_sequence(
        mock_model, inputs, labels, criterion, mock_config_sequence
    )

    # Assertions
    mock_model.assert_called_once_with(inputs[:, :-1])
    assert isinstance(loss, float)
    assert isinstance(correct, float)
    assert samples == (inputs[:, 1:] != BINARY_TOKENS["<EOS>"]).sum().item()


def test_compute_sequence_loss_and_accuracy():
    # Create mock outputs and targets
    output = torch.tensor(
        [[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.2, 0.8], [0.9, 0.1]]]
    )  # [batch_size=2, seq_len=3, num_tokens=2]

    target = torch.tensor([[1, 0, 1], [0, 1, 0]])  # [batch_size=2, seq_len=3]

    eq_positions = torch.tensor([1, 1])  # Both sequences have '=' at position 1

    criterion = torch.nn.CrossEntropyLoss(ignore_index=1)  # Assume '<EOS>' token is 1

    config = {"task_type": "sequence"}

    loss, acc = compute_sequence_loss_and_accuracy(
        output, target, eq_positions, criterion, config
    )

    # Since we have masking beyond '=' at position 1, only positions 2 are considered
    # For the first sequence: target[0,2] = 1
    # For the second sequence: target[1,2] = 0
    # Predicted classes are argmax of output[:,2] for each sequence
    expected_loss = criterion(output[:, 2], target[:, 2]).item()
    expected_acc = (
        (torch.argmax(output[:, 2], dim=1) == target[:, 2]).float().mean()
    ).item()

    assert loss.item() == pytest.approx(expected_loss)
    assert acc.item() == pytest.approx(expected_acc)


def test_get_criterion_classification(mock_config_classification):
    config = mock_config_classification
    criterion = get_criterion(config)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)


def test_get_criterion_sequence(mock_config_sequence):
    config = mock_config_sequence
    criterion = get_criterion(config)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.ignore_index == BINARY_TOKENS["<EOS>"]
