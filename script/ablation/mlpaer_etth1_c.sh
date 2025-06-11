
# 创建日志目录
if [ ! -d "./MLPAer" ]; then
    mkdir ./MLPAer
fi

if [ ! -d "./MLPAer/only_channel" ]; then
    mkdir ./MLPAer/only_channel
fi

seq_len=96
model_name=MLPAer_only_channel
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
    --d_model 512\
    --t_ff 512\
    --c_ff 14 \
    --t_dropout 0\
    --c_dropout 0\
    --head_dropout 0.7\
    --embed_dropout 0\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 128 \
    --itr 1 \
    --learning_rate 0.0005 >MLPAer/only_channel/MLPAer_etth1_96_96.log


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
    --d_model 512\
    --t_ff 512\
    --c_ff 14\
    --c_dropout 0\
    --t_dropout 0.2\
    --head_dropout 0.6\
    --embed_dropout 0\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 128 \
    --itr 1 \
    --learning_rate 0.0002 >MLPAer/only_channel/MLPAer_etth1_96_192.log

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
    --d_model 512\
    --t_ff 512\
    --c_ff 14 \
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.7\
    --embed_dropout 0\
    --patience 5\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 128 \
    --itr 1 \
    --learning_rate 0.0002 >MLPAer/only_channel/MLPAer_etth1_96_336.log

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
    --d_model 512\
    --c_ff 14\
    --t_ff 512\
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.7\
    --embed_dropout 0\
    --patience 7\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 128 \
    --itr 1 \
    --learning_rate 0.0002 >MLPAer/only_channel/MLPAer_etth1_96_720.log