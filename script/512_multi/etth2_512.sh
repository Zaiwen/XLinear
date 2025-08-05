model_name=XLinear
root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
random_seed=2025

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh2_512_96 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 512 \
    --pred_len 96 \
    --enc_in 7 \
    --d_model 1024\
    --c_ff 4\
    --t_ff 128\
    --c_dropout 0\
    --t_dropout 0.4\
    --head_dropout 0.4\
    --embed_dropout 0.4\
    --patience 10\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 256 \
    --itr 1 \
    --learning_rate 0.0001


python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh2_512_192 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 512 \
    --pred_len 192 \
    --enc_in 7 \
    --d_model 1024\
    --c_ff 4\
    --t_ff 128\
    --c_dropout 0\
    --t_dropout 0.4\
    --head_dropout 0.4\
    --embed_dropout 0.4\
    --patience 10\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 256 \
    --itr 1 \
    --learning_rate 0.0001


python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh2_512_336 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 512 \
    --pred_len 336 \
    --enc_in 7 \
    --d_model 256\
    --c_ff 7\
    --t_ff 32\
    --c_dropout 0\
    --t_dropout 0.4\
    --head_dropout 0.4\
    --embed_dropout 0.4\
    --patience 10\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 256 \
    --itr 1 \
    --learning_rate 0.0001

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh2_512_720 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 512 \
    --pred_len 720 \
    --enc_in 7 \
    --d_model 1024\
    --c_ff 7\
    --t_ff 128\
    --c_dropout 0\
    --t_dropout 0.4\
    --head_dropout 0.4\
    --embed_dropout 0.4\
    --patience 10\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 256 \
    --itr 1 \
    --learning_rate 0.0001