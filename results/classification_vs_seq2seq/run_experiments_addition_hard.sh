lr=1e-3
num_steps=50000
num_layers=2
dim_model=128
num_heads=4
batch_size=512
dropout=0.01
weight_decay=1
operation="x+y"
wandb_tracking="minimal"
curriculum="random"
training_fraction=0.5
max_bit_length_train=7
max_bit_length_val_out=8

max_bit_length_train_sq=$((2 ** max_bit_length_train))
max_bit_length_val_out_sq=$((2 ** max_bit_length_val_out))
operation_binary="${operation}_binary_flipped"

python grokking/cli.py --operation $operation --task_type classification --max_bit_length_train $max_bit_length_train_sq --max_bit_length_val_out $max_bit_length_val_out_sq --training_fraction $training_fraction --num_layers $num_layers --dim_model $dim_model --num_heads $num_heads --batch_size $batch_size --learning_rate $lr --weight_decay $weight_decay --num_steps $num_steps --device cuda --dropout $dropout --wandb_tracking $wandb_tracking --curriculum $curriculum

python grokking/cli.py --operation $operation_binary --task_type sequence --max_bit_length_train $max_bit_length_train --max_bit_length_val_out $max_bit_length_val_out --training_fraction $training_fraction --num_layers $num_layers --dim_model $dim_model --num_heads $num_heads --batch_size $batch_size --learning_rate $lr --weight_decay $weight_decay --num_steps $num_steps --device cuda --dropout $dropout --wandb_tracking $wandb_tracking --curriculum $curriculum