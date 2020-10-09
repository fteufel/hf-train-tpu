# Pretraining BERT on TPUs

Own repo to make working on google cloud easy.  
Just contains training scripts adapted from huggingface, optimized datasets and script with the run parameters.  
huggingface training script takes care of distributed training and tpus, just need to run this in a torch_xla instance on gcloud.

`train_mlm.py` is huggingface's `run_language_modeling.py` with `get_dataset` replaced by a custom function to use huggingface datasets to load a line-by-line `.txt` file, activated by `--out_of_core`. The default implementation would try to load the full file into memory at once.  
`--line_by_line` was modified to truncate and pad to `block_size`, so that all batches are the same size. Otherwise xla graph needs to recompile all the time and using TPUs is pointless.



# Setup 


Make virtual machine
```
gcloud compute instances create first-try \
--zone=europe-west4-a \
--machine-type=n1-standard-16  \
--image-family=torch-xla \
--image-project=ml-images  \
--boot-disk-size=200GB \
--scopes=https://www.googleapis.com/auth/cloud-platform
 --service-account=service-account-manual@bert-archaea-fine-tuning.iam.gserviceaccount.com
```
Login
```
gcloud compute ssh first-try --zone=europe-west4-a

```
Make TPU

```
gcloud compute tpus create train-bert-2 \
--zone=europe-west4-a \
--network=default \
--version=pytorch-nightly  \
--accelerator-type=v3-8
```

Set environment variables

```
 export TPU_IP_ADDRESS=10.116.37.234
 export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
```

# Run stuff

`conda activate torch-xla-nightly`, then install transformers from source and `pip install datasets`.  

## quick check

Now `python3 train_mlm.py` should work on TPU.  

Hyperparameters can be modified in `run.sh`. Reference for arguments https://huggingface.co/transformers/main_classes/trainer.html. TrainingArguments parameters are exposed as CLI.

## Using screen

1. after `screen` run `. /anaconda3/etc/profile.d/conda.sh` to make sure anaconda is available.  
2. Now `conda activate torch-xla-nightly` works.
3. `sh run_distributed.sh`



From CLI:  

```
. /anaconda3/etc/profile.d/conda.sh
 export TPU_IP_ADDRESS=10.235.157.10
 export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
 export TRAIN_FILE=~/preprocessed_allorgs_alllengths.txt
conda activate torch-xla-1.6

python3 xla_spawn.py --num_cores 8 train_mlm.py --output_dir=output/9     --model_type=bert     --model_name_or_path=Rostlab/prot_bert     --do_train     --train_data_file=$TRAIN_FILE --mlm --line_by_line --block_size 512 --max_steps 200 --out_of_core --run_name 200stepslowlr --logging_steps 10 --learning_rate 0.000001 --per_device_train_batch_size 25
```

Sometimes necessary to ctrl+c x2, then `pkill -9 python`.  



## Notes for prot_bert

- batch size 30 fails with block size 512 - assume torch-xla is more memory hungry than same model in TF
- batch size 20 seems to be maximum that works