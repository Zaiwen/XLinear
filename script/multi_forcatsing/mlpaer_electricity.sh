# 创建日志目录
if [ ! -d "./MLPAer" ]; then
    mkdir ./MLPAer
fi

if [ ! -d "./MLPAer/LongForecasting" ]; then
    mkdir ./MLPAer/LongForecasting
fi

seq_len=96
model_name=MLPAer
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
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 321 \
    --d_model 2048\
    --t_ff 512\
    --c_ff 32\
    --c_dropout 0.1\
    --t_dropout 0.2\
    --head_dropout 0.2\
    --embed_dropout 0.2\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0002 >MLPAer/LongForecasting/MLPAer_electricity_96_96.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id electricity_96_192 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --enc_in 321 \
    --d_model 2048\
    --t_ff 512\
    --c_ff 24\
    --c_dropout 0.1\
    --t_dropout 0.2\
    --head_dropout 0.3\
    --embed_dropout 0.1\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/LongForecasting/MLPAer_electricity_96_192.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id electricity_96_336 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --enc_in 321 \
    --d_model 2048\
    --t_ff 512\
    --c_ff 7\
    --c_dropout 0.1\
    --t_dropout 0.2\
    --head_dropout 0.3\
    --embed_dropout 0.1\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/LongForecasting/MLPAer_electricity_96_336.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id electricity_96_720 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --enc_in 321 \
    --d_model 2048\
    --t_ff 512\
    --c_ff 24\
    --t_dropout 0.2\
    --c_dropout 0.1\
    --head_dropout 0.6\
    --embed_dropout 0.2\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0002 >MLPAer/LongForecasting/MLPAer_electricity_96_720.log