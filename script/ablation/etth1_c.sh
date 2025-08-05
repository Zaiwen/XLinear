model_name=XLinear-GT
root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
random_seed=2025

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh1_96_96 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 7 \
    --d_model 128\
    --t_ff 512\
    --c_ff 7 \
    --t_dropout 0\
    --c_dropout 0\
    --head_dropout 0.2\
    --embed_dropout 0\
    --patience 5\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 128 \
    --itr 1 \
    --learning_rate 0.0005


python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh1_96_192 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --enc_in 7 \
    --d_model 128\
    --t_ff 1024\
    --c_ff 7\
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.3\
    --embed_dropout 0\
    --patience 5\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 128 \
    --itr 1 \
    --learning_rate 0.0005

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh1_96_336 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --enc_in 7 \
    --d_model 128\
    --t_ff 1024\
    --c_ff 7 \
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.4\
    --embed_dropout 0\
    --patience 5\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 128 \
    --itr 1 \
    --learning_rate 0.0005

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh1_96_720 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --d_model 64\
    --c_ff 7\
    --t_ff 128\
    --c_dropout 0\
    --t_dropout 0.1\
    --head_dropout 0.2\
    --embed_dropout 0\
    --patience 10\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 128 \
    --itr 1 \
    --learning_rate 0.0005