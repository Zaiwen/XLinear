model_name=XLinear
root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
random_seed=2025

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id electricity_96_96 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 321 \
    --d_model 2048\
    --t_ff 512\
    --c_ff 24\
    --c_dropout 0\
    --t_dropout 0.5\
    --head_dropout 0.7\
    --embed_dropout 0\
    --patience 3\
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
    --model_id electricity_96_192 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 192 \
    --enc_in 321 \
    --d_model 2048\
    --t_ff 512\
    --c_ff 24\
    --c_dropout 0\
    --t_dropout 0.3\
    --head_dropout 0.7\
    --embed_dropout 0\
    --patience 3\
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
    --model_id electricity_96_336 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 336 \
    --enc_in 321 \
    --d_model 2048\
    --t_ff 512\
    --c_ff 24\
    --c_dropout 0\
    --t_dropout 0.4\
    --head_dropout 0.6\
    --embed_dropout 0\
    --patience 3\
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
    --model_id electricity_96_720 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 720 \
    --enc_in 321 \
    --d_model 2048\
    --t_ff 512\
    --c_ff 24\
    --t_dropout 0\
    --c_dropout 0\
    --head_dropout 0.4\
    --embed_dropout 0\
    --des 'Exp' \
    --train_epochs 10 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001