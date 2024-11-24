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

from data_encoder import Encoder  # assuming this is how your Encoder class is imported


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
    "base, a, b, c, flipped, expected_input, expected_label",
    [
        # Test Case 1: Base 2 (Binary)
        (
            2,  # base
            1,  # a
            1,  # b
            2,  # c (should be [1, 0] in base 2)
            False,  # flipped
            [1, "+", 1, "=", 1, 0, "<EOS>"],  # expected_input
            [1, 0, "<EOS>"],  # expected_label
        ),
        # Test Case 2: Base 10 (Decimal)
        (
            10,
            12,
            7,
            19,
            False,
            [1, 2, "+", 7, "=", 1, 9, "<EOS>"],
            [1, 9, "<EOS>"],
        ),
        # Test Case 3: Base 20
        (
            20,
            22,  # In base 20: 1*20 + 2 = [1, 2]
            3,  # [3]
            25,  # In base 20: 1*20 + 5 = [1, 5]
            False,
            [1, 2, "+", 3, "=", 1, 5, "<EOS>"],
            [1, 5, "<EOS>"],
        ),
        # Test Case 4: Base 5 with Flipped Digits
        (
            5,
            3,  # [3]
            2,  # [2]
            0,  # [0]
            True,  # flipped
            [3, "+", 2, "=", 0, "<EOS>"],
            [0, "<EOS>"],
        ),
        # Test Case 5: Base 16 (Hexadecimal-like, but using numeric digits)
        (
            16,
            255,  # In base 16: 15*16 + 15 = [15, 15]
            1,  # [1]
            256,  # In base 16: 1*16^2 + 0*16 + 0 = [1, 0, 0]
            False,
            [15, 15, "+", 1, "=", 1, 0, 0, "<EOS>"],
            [1, 0, 0, "<EOS>"],
        ),
    ],
)
def test_encoder_tokenization(
    base: int,
    a: int,
    b: int,
    c: int,
    flipped: bool,
    expected_input,
    expected_label,
):
    """
    Test the Encoder's tokenization for various bases, ensuring that numbers are
    correctly converted into digit tokens, and that operation and special tokens
    are properly included.

    Args:
        base (int): Numerical base for encoding.
        a (int): First operand.
        b (int): Second operand.
        c (int): Result operand.
        flipped (bool): Whether to flip the digit order.
        expected_input (List): Expected list of tokens for the input sequence.
        expected_label (List): Expected list of tokens for the label sequence.
    """
    # Initialize the Encoder with the specified base
    encoder = Encoder(base=base)
    token_dict = encoder.token_dict

    # Encode the operation sequence: a + b =
    operation_seq = encoder.encode_sequence(a, b, operation="+", flipped=flipped)
    # Encode the label sequence: c <EOS>
    answer_seq = encoder.encode_label_sequence(c, flipped=flipped)
    # Concatenate operation and answer for the input sequence
    input_seq = operation_seq + answer_seq
    # Labels are the answer sequence only
    label_seq = answer_seq

    # Function to map expected symbols/digits to their corresponding token indices
    def map_to_tokens(sequence):
        token_sequence = []
        for item in sequence:
            if isinstance(item, int):
                # Digit token: convert digit to string before mapping
                token = token_dict.get(str(item))
                if token is None:
                    raise KeyError(
                        f"Digit '{item}' not found in token_dict for base {base}."
                    )
            elif isinstance(item, str):
                # Operation or special token: map directly
                token = token_dict.get(item)
                if token is None:
                    raise KeyError(
                        f"Token '{item}' not found in token_dict for base {base}."
                    )
            else:
                raise ValueError(f"Unexpected item type in sequence: {item}")
            token_sequence.append(token)
        return token_sequence

    # Map expected_input and expected_label to token indices
    expected_input_tokens = map_to_tokens(expected_input)
    expected_label_tokens = map_to_tokens(expected_label)

    # Assertions to verify correctness
    assert input_seq == expected_input_tokens, (
        f"Input token sequence mismatch for base {base}.\n"
        f"Expected: {expected_input_tokens}\nGot: {input_seq}"
    )
    assert label_seq == expected_label_tokens, (
        f"Label token sequence mismatch for base {base}.\n"
        f"Expected: {expected_label_tokens}\nGot: {label_seq}"
    )


@pytest.mark.parametrize(
    "base, max_number, expected_digits",
    [
        (10, 127, 3),  # decimal: 127 needs 3 digits
        (2, 127, 7),  # binary: 1111111 needs 7 digits
        (16, 127, 2),  # hex: 7F needs 2 digits
        (8, 127, 3),  # octal: 177 needs 3 digits
        (10, 1000, 4),  # decimal: 1000 needs 4 digits
        (2, 1000, 10),  # binary: 1111101000 needs 10 digits
    ],
)
def test_theoretical_max_number_of_digits(base, max_number, expected_digits):
    """
    Test that theoretical_max_number_of_digits correctly calculates the number of digits
    needed to represent max_number in the given base.

    Args:
        base (int): The base of the number system
        max_number (int): The number to calculate digits for
        expected_digits (int): Expected number of digits
    """
    encoder = Encoder(base=base, max_number=max_number)
    result = encoder.theoretical_max_number_of_digits()
    assert result == expected_digits, (
        f"For base {base} and number {max_number}, "
        f"expected {expected_digits} digits but got {result}. "
        f"In base {base}, {max_number} is "
        f"{encoder.encode_number(max_number)}"
    )


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
