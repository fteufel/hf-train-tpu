
export WANDB_PROJECT=finetune-bert
export TPU_IP_ADDRESS=10.116.37.234
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"



export TRAIN_FILE="../preprocessed_all_short.txt"

python3 xla_spawn.py \
 --num_cores 8\
 train_mlm.py
 --output_dir=output3\
 --model_type=bert\
 --model_name_or_path="Rostlab/prot_bert"\
 --mlm\
 --do_train\
 --train_data_file=$TRAIN_FILE \
 --run_name=debug_train_script \
 --mlm_probability=0.15 \
 --block_size=512 \
 --per_device_train_batch_size=30 \
 --learning_rate=0.002 \
 --weight_decay=0.01 \
 --max_steps=10000 \
 --line_by_line \
 --save_steps=1000 \
 --warmup_steps=1333 \
 --logging_steps=10\
 --logging_dir=firstrun\
