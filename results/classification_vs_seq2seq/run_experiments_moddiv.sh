python grokking/cli.py --operation "x/y" --curriculum "ordered" --training_fraction 0.5 --max_bit_length_train 6 --max_bit_length_val_out 11 --num_layers 2 --dim_model 128 --num_heads 4 --batch_size 512 --learning_rate 1e-3 --weight_decay 1 --num_steps 10000 --device cuda --dropout 0.0 --wandb_tracking minimal

python grokking/cli.py --operation "x/y_binary" --task_type sequence --max_bit_length_train 6 --max_bit_length_val_out 7 --training_fraction 0.5 --num_layers 2 --dim_model 128 --num_heads 4 --batch_size 512 --learning_rate 5e-4 --weight_decay 1 --num_steps 10000 --device cuda --dropout 0.0 --wandb_tracking minimal