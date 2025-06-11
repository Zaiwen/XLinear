# 创建日志目录
if [ ! -d "./MLPAer" ]; then
    mkdir ./MLPAer
fi

if [ ! -d "./MLPAer/MS" ]; then
    mkdir ./MLPAer/MS
fi

seq_len=96
model_name=MLPAer_univariate
root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
random_seed=2025

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id weather_96_96 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 21 \
    --d_model 2048\
    --t_ff 512\
    --c_ff 24\
    --c_dropout 0.1\
    --t_dropout 0.1\
    --head_dropout 0.6\
    --embed_dropout 0.2\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_weather_96_96.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id weather_96_192 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 192 \
    --enc_in 21 \
    --d_model 2048\
    --t_ff 512\
    --c_ff 21\
    --c_dropout 0.1\
    --t_dropout 0.4\
    --head_dropout 0.6\
    --embed_dropout 0.1\
    --des 'Exp' \
    --patience 3\
    --train_epochs 30 \
    --batch_size 64 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_weather_96_192.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id weather_96_336 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 336 \
    --enc_in 21 \
    --d_model 2048\
    --t_ff 512\
    --c_ff 21\
    --c_dropout 0.1\
    --t_dropout 0.2\
    --head_dropout 0.4\
    --embed_dropout 0.2\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_weather_96_336.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id weather_96_720 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 720 \
    --enc_in 21 \
    --d_model 512\
    --t_ff 512\
    --c_ff 21\
    --c_dropout 0.1\
    --t_dropout 0.1\
    --head_dropout 0.5\
    --embed_dropout 0.1\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 32\
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_weather_96_720.log