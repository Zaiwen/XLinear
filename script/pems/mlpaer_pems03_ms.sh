# 创建日志目录
if [ ! -d "./MLPAer" ]; then
    mkdir ./MLPAer
fi

if [ ! -d "./MLPAer/MS" ]; then
    mkdir ./MLPAer/MS
fi

seq_len=96
model_name=MLPAer_univariate
root_path_name=./dataset/PEMS/
data_path_name=PEMS03.npz
model_id_name=PEMS03
data_name=PEMS
random_seed=2025

# python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id pems03_$seq_len'_'12 \
#     --model $model_name \
#     --data $data_name \
#     --features MS \
#     --seq_len $seq_len \
#     --pred_len 12 \
#     --enc_in 358 \
#     --d_model 1024\
#     --t_ff 128\
#     --c_ff 256\
#     --c_dropout 0\
#     --t_dropout 0\
#     --head_dropout 0.2\
#     --embed_dropout 0\
#     --patience 3\
#     --des 'Exp' \
#     --train_epochs 20 \
#     --batch_size 32 \
#     --itr 1 \
#     --learning_rate 0.001 >MLPAer/MS/MLPAer_pems03_$seq_len'_'12.log

# python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id pems03_$seq_len'_'24 \
#     --model $model_name \
#     --data $data_name \
#     --features MS \
#     --seq_len $seq_len \
#     --pred_len 24 \
#     --enc_in 358 \
#     --d_model 1024\
#     --t_ff 256\
#     --c_ff 256\
#     --c_dropout 0\
#     --t_dropout 0\
#     --head_dropout 0.2\
#     --embed_dropout 0\
#     --patience 3\
#     --des 'Exp' \
#     --train_epochs 20 \
#     --batch_size 32 \
#     --itr 1 \
#     --learning_rate 0.001 >MLPAer/MS/MLPAer_pems03_$seq_len'_'24.log

# python -u run_longExp.py \
#     --random_seed $random_seed \
#     --is_training 1 \
#     --root_path $root_path_name \
#     --data_path $data_path_name \
#     --model_id pems03_$seq_len'_'48 \
#     --model $model_name \
#     --data $data_name \
#     --features MS \
#     --seq_len $seq_len \
#     --pred_len 48 \
#     --enc_in 358 \
#     --d_model 512\
#     --t_ff 128\
#     --c_ff 256\
#     --c_dropout 0.4\
#     --t_dropout 0\
#     --head_dropout 0.3\
#     --embed_dropout 0.2\
#     --patience 3\
#     --des 'Exp' \
#     --train_epochs 20 \
#     --batch_size 128 \
#     --itr 1 \
#     --learning_rate 0.002 >MLPAer/MS/MLPAer_pems03_$seq_len'_'48.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 0 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id pems03_$seq_len'_'96 \
    --model $model_name \
    --data $data_name \
    --features MS \
    --seq_len $seq_len \
    --pred_len 96 \
    --enc_in 358 \
    --d_model 512\
    --t_ff 128\
    --c_ff 358\
    --c_dropout 0.5\
    --t_dropout 0\
    --head_dropout 0.3\
    --embed_dropout 0.2\
    --patience 10\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 4 \
    --itr 1 \
    --learning_rate 0.001 >MLPAer/MS/MLPAer_pems03_$seq_len'_'96_test.log