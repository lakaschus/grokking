# test_data.py

import pytest
from data import get_data


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
    "max_bit_length", [3]
)  # You can adjust the bit length as needed
def test_flipped_dataset_is_correctly_flipped(max_bit_length):
    """
    Test that the "x+y_binary_flipped" dataset is correctly flipped compared to "x+y_binary".

    This test generates both datasets with the same parameters and ensures that for each sample,
    the flipped dataset's inputs and labels are the reversed versions of the standard dataset's
    inputs and labels.
    """
    # Generate standard binary addition data
    (
        train_loader,
        val_in_loader,
        val_out_loader,
        op_token,
        eq_token,
        num_unique_tokens,
    ) = get_data(
        operation="x+y_binary",
        max_bit_length_train=max_bit_length,
        max_bit_length_val_out=max_bit_length + 1,
        training_fraction=0.9,
        batch_size=1024,  # Large batch size to collect all samples at once
        curriculum="ordered",  # Ensure the order is consistent
    )

    # Generate flipped binary addition data
    (
        train_loader_flipped,
        val_in_loader_flipped,
        val_out_loader_flipped,
        op_token_flipped,
        eq_token_flipped,
        num_unique_tokens_flipped,
    ) = get_data(
        operation="x+y_binary_flipped",
        max_bit_length_train=max_bit_length,
        max_bit_length_val_out=max_bit_length + 1,
        training_fraction=0.9,
        batch_size=1024,  # Large batch size to collect all samples at once
        curriculum="ordered",  # Ensure the order is consistent
    )

    # Collect all samples from both datasets
    samples_std = collect_all_samples(train_loader) | collect_all_samples(val_in_loader)
    samples_flipped = collect_all_samples(train_loader_flipped) | collect_all_samples(
        val_in_loader_flipped
    )

    # Create dictionaries to map input sequences to label sequences for both datasets
    dict_std = {input_seq: label_seq for input_seq, label_seq in samples_std}
    dict_flipped = {input_seq: label_seq for input_seq, label_seq in samples_flipped}

    # Iterate through standard dataset and verify flipped dataset
    for input_seq_std, label_seq_std in dict_std.items():
        # Reverse the input and label sequences for comparison
        input_seq_flipped_expected = tuple(reversed(input_seq_std))
        label_seq_flipped_expected = tuple(reversed(label_seq_std))

        # Retrieve the corresponding flipped sequences
        label_seq_flipped = dict_flipped.get(input_seq_flipped_expected, None)

        assert (
            label_seq_flipped is not None
        ), f"Flipped input sequence {input_seq_flipped_expected} not found in flipped dataset."

        assert label_seq_flipped == label_seq_flipped_expected, (
            f"Flipped label sequence does not match for input {input_seq_flipped_expected}.\n"
            f"Expected: {label_seq_flipped_expected}\n"
            f"Got: {label_seq_flipped}"
        )


@pytest.mark.parametrize(
    "max_bit_length, training_fraction, curriculum",
    [
        (4, 0.8, "random"),
        (4, 0.8, "ordered"),
        (4, 0.8, "domain_separated"),
    ],
)
def test_train_val_mutually_exclusive(max_bit_length, training_fraction, curriculum):
    """
    Test that the training and validation datasets are mutually exclusive.

    This test generates a dataset with specified parameters, splits it into training and validation,
    and ensures that no sample appears in both subsets.
    """
    # Generate binary addition data
    train_loader, val_loader, _, op_token, eq_token, num_unique_tokens = get_data(
        operation="x+y_binary",
        max_bit_length_train=max_bit_length,
        max_bit_length_val_out=max_bit_length + 1,
        training_fraction=training_fraction,
        batch_size=1024,  # Large batch size to collect all samples at once
        curriculum=curriculum,  # Test different curriculum strategies
    )

    # Collect all samples from training and validation datasets
    samples_train = collect_all_samples(train_loader)
    samples_val = collect_all_samples(val_loader)

    # Check that the intersection is empty
    intersection = samples_train.intersection(samples_val)
    assert (
        len(intersection) == 0
    ), f"Training and validation datasets are not mutually exclusive. Overlapping samples: {intersection}"
