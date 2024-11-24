# test_data.py

import pytest
from data import (
    get_data,
    get_next_prime,
    operation_mod_p_data,
    select_out_of_domain_examples,
    ALL_OPERATIONS,
)
import torch
from torch.utils.data import TensorDataset


def collect_all_samples(data_loader):
    """
    Helper function to collect all samples from a DataLoader.

    Args:
        data_loader (DataLoader): The DataLoader to iterate through.

    Returns:
        Set of tuples where each tuple is (input_sequence, label_sequence).
    """
    samples = set()
    for batch in data_loader:
        inputs, labels = batch
        # Convert tensors to tuples for set operations
        for input_seq, label_seq in zip(inputs, labels):
            samples.add((tuple(input_seq.tolist()), tuple(label_seq.tolist())))
    return samples


@pytest.mark.parametrize(
    "max_bit_length, training_fraction, curriculum",
    [(4, 0.8, "random"), (4, 0.8, "ordered")],
)
def test_train_val_mutually_exclusive(max_bit_length, training_fraction, curriculum):
    """
    Test that the training and validation datasets are mutually exclusive.

    This test generates a dataset with specified parameters, splits it into training and validation,
    and ensures that no sample appears in both subsets.
    """
    # Generate binary addition data
    train_loader, val_loader, _, op_token, eq_token, num_unique_tokens, encoder = (
        get_data(
            operation="x+y",
            task_type="sequence",
            max_bit_length_train=max_bit_length,
            max_bit_length_val_out=max_bit_length + 1,
            training_fraction=training_fraction,
            batch_size=1024,  # Large batch size to collect all samples at once
            curriculum=curriculum,  # Test different curriculum strategies
        )
    )

    # Collect all samples from training and validation datasets
    samples_train = collect_all_samples(train_loader)
    samples_val = collect_all_samples(val_loader)

    # Check that the intersection is empty
    intersection = samples_train.intersection(samples_val)
    assert (
        len(intersection) == 0
    ), f"Training and validation datasets are not mutually exclusive. Overlapping samples: {intersection}"


def test_get_next_prime():
    # Test basic cases: exact primes and numbers between primes
    assert get_next_prime(90) == 97
    assert get_next_prime(97) == 97
    assert get_next_prime(96) == 97

    # Test small numbers and edge cases
    assert get_next_prime(2) == 2
    assert get_next_prime(1) == 2
    assert get_next_prime(0) == 2


def test_operation_mod_p_data_simple():
    """Test basic operations with small modulo value to verify correctness"""
    p = 7

    # Test x+y_mod
    inputs, labels, _, _ = operation_mod_p_data("x+y_mod", p)
    for i in range(len(inputs)):
        x, op, y, _ = inputs[i]
        result = labels[i]
        assert (
            result == (x + y) % p
        ), f"x+y_mod failed: {x} + {y} mod {p} should be {(x + y) % p}, got {result}"

    # Test x-y_mod
    inputs, labels, _, _ = operation_mod_p_data("x-y_mod", p)
    for i in range(len(inputs)):
        x, op, y, _ = inputs[i]
        result = labels[i]
        assert (
            result == (x - y) % p
        ), f"x-y_mod failed: {x} - {y} mod {p} should be {(x - y) % p}, got {result}"

    # Test x^2+y^2_mod
    inputs, labels, _, _ = operation_mod_p_data("x^2+y^2_mod", p)
    for i in range(len(inputs)):
        x, op, y, _ = inputs[i]
        result = labels[i]
        assert (
            result == (x**2 + y**2) % p
        ), f"x^2+y^2_mod failed: {x}^2 + {y}^2 mod {p} should be {(x**2 + y**2) % p}, got {result}"

    # Test x^3+xy+y^2_mod
    inputs, labels, _, _ = operation_mod_p_data("x^3+xy+y^2_mod", p)
    for i in range(len(inputs)):
        x, op, y, _ = inputs[i]
        result = labels[i]
        expected = (x**3 + x * y + y**2) % p
        assert (
            result == expected
        ), f"x^3+xy+y^2_mod failed: {x}^3 + {x}*{y} + {y}^2 mod {p} should be {expected}, got {result}"

    # Test x^3+xy+y^2+x_mod
    inputs, labels, _, _ = operation_mod_p_data("x^3+xy+y^2+x_mod", p)
    for i in range(len(inputs)):
        x, op, y, _ = inputs[i]
        result = labels[i]
        expected = (x**3 + x * y + y**2 + x) % p
        assert (
            result == expected
        ), f"x^3+xy+y^2+x_mod failed: {x}^3 + {x}*{y} + {y}^2 + {x} mod {p} should be {expected}, got {result}"


def test_regular_addition():
    """Test regular addition without modulo"""
    p = 5  # Value doesn't matter for regular addition
    inputs, labels, _, _ = operation_mod_p_data("x+y", p)
    for i in range(len(inputs)):
        x, op, y, _ = inputs[i]
        result = labels[i]
        assert result == x + y, f"x+y failed: {x} + {y} should be {x + y}, got {result}"


def test_dataset_sizes():
    """Test that datasets have expected sizes based on modulo value"""
    p = 5
    expected_size = p * p  # Cartesian product of numbers 0 to p-1

    for operation in [
        "x+y_mod",
        "x-y_mod",
        "x^2+y^2_mod",
        "x^3+xy+y^2_mod",
        "x^3+xy+y^2+x_mod",
        "x+y",
    ]:
        inputs, labels, _, _ = operation_mod_p_data(operation, p)
        assert (
            len(inputs) == expected_size
        ), f"{operation} dataset size should be {expected_size}, got {len(inputs)}"
        assert (
            len(labels) == expected_size
        ), f"{operation} labels size should be {expected_size}, got {len(labels)}"


@pytest.mark.parametrize(
    "operation",
    ["x+y_mod", "x-y_mod", "x^2+y^2_mod", "x^3+xy+y^2_mod", "x^3+xy+y^2+x_mod", "x+y"],
)
def test_dataloader_operations(operation):
    """Test that the data in the DataLoader matches expected calculations"""
    # Use small numbers for easy verification
    max_bit_length_train = 6
    max_bit_length_val_out = 7
    batch_size = 4

    # Get the actual modulo (next prime number) for modulo operations
    modulo = (
        get_next_prime(max_bit_length_train)
        if "mod" in operation
        else max_bit_length_train
    )

    # Get dataloaders
    (
        train_loader,
        val_in_loader,
        val_out_loader,
        op_token,
        eq_token,
        num_tokens,
        encoder,
    ) = get_data(
        operation=operation,
        task_type="classification",
        max_bit_length_train=max_bit_length_train,
        max_bit_length_val_out=max_bit_length_val_out,
        batch_size=batch_size,
        training_fraction=0.8,
        curriculum="random",
    )

    # Check a batch from each loader
    for loader in [train_loader, val_in_loader]:
        batch_inputs, batch_labels = next(iter(loader))

        # Check each sample in the batch
        for i in range(len(batch_inputs)):
            x = batch_inputs[i][0].item()  # First element is x
            y = batch_inputs[i][2].item()  # Third element is y
            result = batch_labels[i].item()

            # Calculate expected result based on operation
            if operation == "x+y_mod":
                expected = (x + y) % modulo
            elif operation == "x-y_mod":
                expected = (x - y) % modulo
            elif operation == "x^2+y^2_mod":
                expected = (x**2 + y**2) % modulo
            elif operation == "x^3+xy+y^2_mod":
                expected = (x**3 + x * y + y**2) % modulo
            elif operation == "x^3+xy+y^2+x_mod":
                expected = (x**3 + x * y + y**2 + x) % modulo
            elif operation == "x+y":
                expected = x + y

            assert result == expected, (
                f"{operation} failed in DataLoader:\n"
                f"x={x}, y={y}\n"
                f"modulo={modulo}\n"
                f"Expected {expected}, got {result}"
            )

        # Basic checks for tensor shapes and token values
        assert (
            batch_inputs.dim() == 2
        ), f"Expected 2D tensor for inputs, got {batch_inputs.dim()}D"
        assert (
            batch_labels.dim() == 1
        ), f"Expected 1D tensor for labels, got {batch_labels.dim()}D"
        assert (
            batch_inputs.size(0) <= batch_size
        ), f"Batch size exceeded: {batch_inputs.size(0)} > {batch_size}"

        # Check that operation and equals tokens are in correct positions
        assert (
            batch_inputs[:, 1] == op_token
        ).all(), "Operation token not in correct position"
        assert (
            batch_inputs[:, 3] == eq_token
        ).all(), "Equals token not in correct position"


def test_select_out_of_domain_examples():
    # Create a simple training dataset
    train_inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    train_labels = torch.tensor([100, 200, 300])
    train_dataset = TensorDataset(train_inputs, train_labels)

    # Create out-of-domain examples including both overlapping and new examples
    inputs_out = torch.tensor(
        [
            [1, 2, 3, 4],  # Should be filtered (exists in training)
            [5, 6, 7, 8],  # Should be filtered (exists in training)
            [13, 14, 15, 16],  # Should remain (new example)
            [17, 18, 19, 20],  # Should remain (new example)
        ]
    )
    labels_out = torch.tensor([100, 200, 400, 500])

    # Get filtered examples
    filtered_inputs, filtered_labels = select_out_of_domain_examples(
        inputs_out, labels_out, train_dataset
    )

    # Verify the shape of filtered tensors
    assert filtered_inputs.shape[0] == 2, "Should have 2 examples after filtering"
    assert filtered_labels.shape[0] == 2, "Should have 2 labels after filtering"

    # Verify that the correct examples were kept
    expected_inputs = torch.tensor([[13, 14, 15, 16], [17, 18, 19, 20]])
    expected_labels = torch.tensor([400, 500])

    assert torch.equal(
        filtered_inputs, expected_inputs
    ), "Filtered inputs don't match expected"
    assert torch.equal(
        filtered_labels, expected_labels
    ), "Filtered labels don't match expected"
