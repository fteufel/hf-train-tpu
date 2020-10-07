export WANDB_PROJECT=finetune-bert
wandb login "3850816f07ddd291761dba5da24453c2350b898a"
export TPU_IP_ADDRESS=10.116.37.234
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"

# Full UniRef100: 216 580 000
# This dataset  :   7 502 898
# Dataset is 30x larger --> train 30x shorter.

python3 train_mlm.py \
 --output_dir=run_one \
 --model_type=bert \
 --model_name_or_path="Rostlab/prot_bert" \
 --do_train \
 --train_data_file==debugset.txt \
 --mlm \
 --mlm_probability=0.15 \
 --block_size=512 \
 --per_device_train_batch_size=30 \
 --learning_rate=0.002 \
 --weight_decay=0.01 \
 --max_steps=10000 \
 --line_by_line \
 --save_steps=2500 \
 --warmup_steps=1333 \
 --debug True


#original
# --max_steps=300000
# --save_steps=5000
# --warmup_steps=40000
