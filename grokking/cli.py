from argparse import ArgumentParser

import training

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--operation", type=str, default="x/y")
    parser.add_argument("--multitask", action="store_true", default=False)
    parser.add_argument("--continue_training", action="store_true", default=False)
    parser.add_argument("--fixed_sequence_length", action="store_true", default=False)
    parser.add_argument("--curriculum", type=str, default="random")
    parser.add_argument("--task_type", type=str, default="classification")
    parser.add_argument("--training_fraction", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--prime", type=int, default=97)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--num_steps", type=int, default=1e5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--base", type=int, default=2)
    parser.add_argument("--max_bit_length_train", type=int, default=6)
    parser.add_argument("--max_bit_length_val_out", type=int, default=7)
    parser.add_argument("--wandb_tracking", type=str, default="max")
    args = parser.parse_args()

    training.main(args)
