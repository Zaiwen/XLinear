model_name=XLinear
root_path_name=./dataset/environment
data_path_name=do_409215.csv
model_id_name=do_409215
data_name=custom
random_seed=2025

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id do_409215_96_96 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 7 \
    --d_model 1024\
    --t_ff 256\
    --c_ff 7 \
    --t_dropout 0\
    --c_dropout 0\
    --head_dropout 0\
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
    --model_id do_409215_96_192 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 192 \
    --enc_in 7 \
    --d_model 1024\
    --t_ff 256\
    --c_ff 7 \
    --t_dropout 0.4\
    --c_dropout 0\
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
    --model_id do_409215_96_336 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 336 \
    --enc_in 7 \
    --d_model 1024\
    --t_ff 256\
    --c_ff 7 \
    --t_dropout 0\
    --c_dropout 0\
    --head_dropout 0\
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
    --model_id do_409215_96_720 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 720 \
    --enc_in 7 \
    --d_model 1024\
    --t_ff 256\
    --c_ff 7 \
    --t_dropout 0\
    --c_dropout 0\
    --head_dropout 0\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 10 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001