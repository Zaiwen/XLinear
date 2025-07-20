if [ ! -d "./MLPAer" ]; then
    mkdir ./MLPAer
fi

if [ ! -d "./MLPAer/MS" ]; then
    mkdir ./MLPAer/MS
fi

seq_len=96
model_name=MLPAer_univariate
root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
random_seed=2025

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id traffic_96_96 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 862 \
    --d_model 2048\
    --t_ff 400\
    --c_ff 32\
    --c_dropout 0.1\
    --t_dropout 0.1\
    --head_dropout 0.7\
    --embed_dropout 0.1\
    --patience 2\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_traffic_96_96.log


python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id traffic_96_192 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 192 \
    --enc_in 862 \
    --d_model 2048\
    --t_ff 400\
    --c_ff 32\
    --c_dropout 0.1\
    --t_dropout 0.3\
    --head_dropout 0.7\
    --embed_dropout 0.3\
    --patience 2\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_traffic_96_192.log


python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id traffic_96_336 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 336 \
    --enc_in 862 \
    --d_model 2048\
    --t_ff 336\
    --c_ff 32\
    --c_dropout 0.1\
    --t_dropout 0.3\
    --head_dropout 0.7\
    --embed_dropout 0.3\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_traffic_96_336.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id traffic_96_720 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 720 \
    --enc_in 862 \
    --d_model 2048\
    --t_ff 336\
    --c_ff 48\
    --c_dropout 0.1\
    --t_dropout 0.3\
    --head_dropout 0.7\
    --embed_dropout 0.3\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_traffic_96_720.log