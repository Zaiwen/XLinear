# 创建日志目录
if [ ! -d "./MLPAer" ]; then
    mkdir ./MLPAer
fi

if [ ! -d "./MLPAer/LongForecasting" ]; then
    mkdir ./MLPAer/LongForecasting
fi

seq_len=144
model_name=MLPAer
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
    --model_id weather_$seq_len'_'96 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 96 \
    --enc_in 21 \
    --d_model 256\
    --t_ff 256\
    --c_ff 42\
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.2\
    --embed_dropout 0\
    --patience 10\
    --des 'Exp' \
    --train_epochs 20 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0005 >MLPAer/LongForecasting/MLPAer_weather_$seq_len'_'96.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id weather_$seq_len'_'192 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 192 \
    --enc_in 21 \
    --d_model 256\
    --t_ff 128\
    --c_ff 42\
    --c_dropout 0\
    --t_dropout 0.2\
    --head_dropout 0.3\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 10 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0005 >MLPAer/LongForecasting/MLPAer_weather_$seq_len'_'192.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id weather_$seq_len'_'336 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 336 \
    --enc_in 21 \
    --d_model 256\
    --t_ff 128\
    --c_ff 42\
    --c_dropout 0\
    --t_dropout 0.2\
    --head_dropout 0.4\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 10 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0005 >MLPAer/LongForecasting/MLPAer_weather_$seq_len'_'336.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id weather_$seq_len'_'720 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len 720 \
    --enc_in 21 \
    --d_model 512\
    --t_ff 256\
    --c_ff 48\
    --c_dropout 0.2\
    --t_dropout 0\
    --head_dropout 0.4\
    --embed_dropout 0.1\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 32\
    --itr 1 \
    --learning_rate 0.0002 >MLPAer/LongForecasting/MLPAer_weather_$seq_len'_'720.log