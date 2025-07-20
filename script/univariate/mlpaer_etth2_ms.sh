# 已验证

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
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
random_seed=2025

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh1_96_96 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 7 \
    --d_model 256\
    --t_ff 64\
    --c_ff 7 \
    --t_dropout 0\
    --c_dropout 0\
    --head_dropout 0.4\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 32 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_etth2_96_96.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh1_96_192 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 192 \
    --enc_in 7 \
    --d_model 512\
    --t_ff 128\
    --c_ff 7\
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.5\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 128 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_etth2_96_192.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh1_96_336 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 336 \
    --enc_in 7 \
    --d_model 512\
    --t_ff 512\
    --c_ff 7 \
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.7\
    --embed_dropout 0\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_etth2_96_336.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh1_96_720 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len 96 \
    --pred_len 720 \
    --enc_in 7 \
    --d_model 1024\
    --t_ff 512\
    --c_ff 7 \
    --c_dropout 0.1\
    --t_dropout 0.1\
    --head_dropout 0.7\
    --embed_dropout 0.3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/MS/MLPAer_etth2_96_720.log