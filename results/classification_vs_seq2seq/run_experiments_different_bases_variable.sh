lr=1e-3
num_steps=2000000
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
task="xy_abs_diff"
bases=(64 40 20 10 5 2)

for base in "${bases[@]}"; do
    echo "Testing configuration for base: $base"
    python grokking/cli.py \
        --operation $task \
        --base "$base" \
        --task_type sequence \
        --max_bit_length_train $max_bit_length_train \
        --max_bit_length_val_out $max_bit_length_val_out \
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
for base in "${bases[@]}"; do
    echo "Running full experiment for base: $base"
    python grokking/cli.py \
        --operation "$task" \
        --base "$base" \
        --task_type sequence \
        --max_bit_length_train $max_bit_length_train \
        --max_bit_length_val_out $max_bit_length_val_out \
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