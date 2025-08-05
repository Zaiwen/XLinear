model_name=XLinear
root_path_name=./dataset/crop
data_path_name=crop.csv
model_id_name=crop
data_name=custom
random_seed=2025

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id crop_96_12 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 12 \
    --enc_in 8 \
    --d_model 128\
    --c_ff 8\
    --t_ff 32\
    --c_dropout 0.8\
    --t_dropout 0.5\
    --head_dropout 0.5\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0001

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id crop_96_24 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 24 \
    --enc_in 8 \
    --d_model 24\
    --c_ff 8\
    --t_ff 32\
    --c_dropout 0\
    --t_dropout 0.8\
    --head_dropout 0.4\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 10 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id crop_96_36 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 36 \
    --enc_in 8 \
    --d_model 32\
    --c_ff 8\
    --t_ff 48\
    --c_dropout 0\
    --t_dropout 0.8\
    --head_dropout 0.2\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 10 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id crop_96_48 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 48 \
    --enc_in 8 \
    --d_model 32\
    --c_ff 8\
    --t_ff 48\
    --c_dropout 0\
    --t_dropout 0.8\
    --head_dropout 0.4\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001
