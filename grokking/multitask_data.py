from typing import List, Tuple, Dict
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from data import get_data, operation_mod_p_data
import torch


class MultitaskDataGenerator:
    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        self.task_to_eq_token = {}

    def generate_multitask_data(
        self,
        max_bit_length_train: int,
        max_bit_length_val_out: int,
        training_fraction: float = 0.8,
        batch_size: int = 32,
        curriculum: str = "random",
    ) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], int, int]:

        train_datasets = []
        val_in_datasets = []
        val_out_datasets = []

        # Get data for each task
        for i, task in enumerate(self.tasks):
            (
                train_loader,
                val_in_loader,
                val_out_loader,
                op_token,
                eq_token,
                num_unique_tokens,
            ) = get_data(
                operation=task,
                task_type="classification",  # To be generalized
                max_bit_length_train=max_bit_length_train,
                max_bit_length_val_out=max_bit_length_val_out,
                training_fraction=training_fraction,
                batch_size=batch_size,  # Get full dataset
                curriculum=curriculum,
                increment_eq_token=i,
            )

            # Store the operation token for this task
            self.task_to_eq_token[task] = eq_token

            # Add datasets to our lists
            train_datasets.append(train_loader.dataset)
            val_in_datasets.append(val_in_loader.dataset)
            val_out_datasets.append(val_out_loader.dataset)

        # Combine datasets
        combined_train = ConcatDataset(train_datasets)
        combined_val_in = ConcatDataset(val_in_datasets)
        combined_val_out = ConcatDataset(val_out_datasets)

        # Create new dataloaders with the combined datasets
        train_loader = DataLoader(
            combined_train, batch_size=batch_size, shuffle=(curriculum == "random")
        )
        val_in_loader = DataLoader(
            combined_val_in, batch_size=batch_size, shuffle=(curriculum == "random")
        )
        val_out_loader = DataLoader(
            combined_val_out, batch_size=batch_size, shuffle=(curriculum == "random")
        )

        num_unique_tokens = max_bit_length_train + 1 + len(self.tasks)

        return (
            train_loader,
            val_in_loader,
            val_out_loader,
            op_token,
            eq_token,
            num_unique_tokens,
        )


def get_multitask_data(
    tasks: List[str],
    max_bit_length_train: int,
    max_bit_length_val_out: int,
    training_fraction: float = 0.8,
    batch_size: int = 32,
    curriculum: str = "random",
    base: int = 2,
    fixed_sequence_length: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], int, int]:
    """
    Convenience function to generate multitask datasets.

    Example usage:
        tasks = ["x+y_binary", "x/y_binary", "x+y_mod", "x-y_mod"]
        train_loader, val_in_loader, val_out_loader, op_tokens, eq_token, num_tokens = (
            get_multitask_data(
                tasks=tasks,
                max_bit_length_train=6,
                max_bit_length_val_out=7
            )
        )
    """
    generator = MultitaskDataGenerator(tasks)
    return generator.generate_multitask_data(
        max_bit_length_train=max_bit_length_train,
        max_bit_length_val_out=max_bit_length_val_out,
        training_fraction=training_fraction,
        batch_size=batch_size,
        curriculum=curriculum,
    )


if __name__ == "__main__":
    tasks = ["x+y_mod", "x-y_mod"]
    train_loader, val_in_loader, val_out_loader, op_token, eq_token, num_tokens = (
        get_multitask_data(
            tasks=tasks, max_bit_length_train=6, max_bit_length_val_out=7
        )
    )
