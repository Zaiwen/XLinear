model_name=XLinear
root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
random_seed=2025

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTm2_96_96 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 7 \
    --d_model 512\
    --t_ff 128\
    --c_ff 21\
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.4\
    --embed_dropout 0\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001


python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTm2_96_192 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 192 \
    --enc_in 7 \
    --d_model 256\
    --t_ff 128\
    --c_ff 32\
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.4\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 20 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTm2_96_336 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 336 \
    --enc_in 7 \
    --d_model 128\
    --t_ff 64\
    --c_ff 32\
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.4\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 20 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTm2_96_720 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 720 \
    --enc_in 7 \
    --d_model 64\
    --t_ff 32\
    --c_ff 32 \
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.4\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 5 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001