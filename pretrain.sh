export WANDB_PROJECT=finetune-bert

# Full UniRef100: 216 580 000
# This dataset  :   7 502 898
# Dataset is 30x larger --> train 30x shorter.

python train_mlm.py \
 --output_dir=run_one \
 --model_type=bert \
 --model_name_or_path="Rostlab/prot_bert" \
 --do_train \
 --train_data_file=preprossed_all.txt \
 --mlm \
 --mlm_probability=0.15 \
 --block_size=512 \
 --per_device_train_batch_size=30 \
 --learning_rate=0.002 \
 --weight_decay=0.01 \
 --max_steps=10000 \
 --save_steps=2500 \
 --warmup_steps=1333


#original
# --max_steps=300000
# --save_steps=5000
# --warmup_steps=40000
