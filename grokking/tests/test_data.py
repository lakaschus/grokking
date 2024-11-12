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
