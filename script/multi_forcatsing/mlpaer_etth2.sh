if [ ! -d "./MLPAer" ]; then
    mkdir ./MLPAer
fi

if [ ! -d "./MLPAer/LongForecasting" ]; then
    mkdir ./MLPAer/LongForecasting
fi

model_name=MLPAer
root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
random_seed=2025
# input = 96

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh2_96_96 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --enc_in 7 \
    --d_model 1024\
    --c_ff 7\
    --t_ff 128\
    --c_dropout 0\
    --t_dropout 0.6\
    --head_dropout 0.5\
    --embed_dropout 0.2\
    --patience 5\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/LongForecasting/MLPAer_etth2_96_96.log


python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh2_96_192 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --enc_in 7 \
    --d_model 1024\
    --c_ff 7\
    --t_ff 128\
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.6\
    --embed_dropout 0.2\
    --patience 10\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 16 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/LongForecasting/MLPAer_etth2_96_192.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh2_96_336 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --enc_in 7 \
    --d_model 336\
    --c_ff 7\
    --t_ff 256\
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.4\
    --embed_dropout 0\
    --patience 5\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 128 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/LongForecasting/MLPAer_etth2_96_336.log

python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ETTh2_96_720 \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --enc_in 7 \
    --d_model 512\
    --c_ff 7\
    --t_ff 128\
    --c_dropout 0\
    --t_dropout 0\
    --head_dropout 0.3\
    --embed_dropout 0\
    --patience 3\
    --des 'Exp' \
    --train_epochs 30 \
    --batch_size 128 \
    --itr 1 \
    --learning_rate 0.0001 >MLPAer/LongForecasting/MLPAer_etth2_96_720.log