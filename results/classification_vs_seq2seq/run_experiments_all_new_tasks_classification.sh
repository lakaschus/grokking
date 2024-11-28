lr=1e-3
num_steps=10000000
num_layers=4
dim_model=128
num_heads=4
batch_size=512
dropout=0.05
weight_decay=1
wandb_tracking="maximal"
curriculum="random"
training_fraction=0.5
max_bit_length_train=6
max_bit_length_val_out=7
max_bit_length_train_sq=$((2 ** max_bit_length_train))
max_bit_length_val_out_sq=$((2 ** max_bit_length_val_out))
tasks=("x,y_abs_max" "x,y_abs_min" "x,y_avg" "x,y_sqrt" "x,y_abs_diff")

for task in "${tasks[@]}"; do
    echo "Testing configuration for task: $task"
    python grokking/cli.py \
        --operation "$task" \
        --task_type classification \
        --max_bit_length_train $max_bit_length_train_sq \
        --max_bit_length_val_out $max_bit_length_val_out_sq \
        --training_fraction $training_fraction \
        --num_layers $num_layers \
        --dim_model $dim_model \
        --num_heads $num_heads \
        --batch_size $batch_size \
        --learning_rate $lr \
        --weight_decay $weight_decay \
        --num_steps 1 \
        --device cuda \
        --dropout $dropout \
        --wandb_tracking disabled \
        --curriculum $curriculum
done

# Main experiment loop
for task in "${tasks[@]}"; do
    echo "Running full experiment for task: $task"
    python grokking/cli.py \
        --operation "$task" \
        --task_type classification \
        --max_bit_length_train $max_bit_length_train_sq \
        --max_bit_length_val_out $max_bit_length_val_out_sq \
        --training_fraction $training_fraction \
        --num_layers $num_layers \
        --dim_model $dim_model \
        --num_heads $num_heads \
        --batch_size $batch_size \
        --learning_rate $lr \
        --weight_decay $weight_decay \
        --num_steps $num_steps \
        --device cuda \
        --dropout $dropout \
        --wandb_tracking $wandb_tracking \
        --curriculum $curriculum
done