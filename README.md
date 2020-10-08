# Pretraining BERT on TPUs

Own repo to make working on google cloud easy.  
Just contains training script adapted from huggingface, lamb optimizer and script with the run parameters.  
huggingface training script takes care of distributed training and tpus, just need to run this in a torch_xla instance on gcloud.

`train_mlm.py` is huggingface's `run_language_modeling.py` with `get_dataset` replaced by a custom function to use huggingface datasets to load a line-by-line `.txt` file. The default implementation would try to load the full file into memory at once.

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
gcloud compute tpus create train-bert \
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

Now `python3 train_mlm.py` should work on TPU.
