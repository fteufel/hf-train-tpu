# Pretraining BERT on TPUs

Own repo to make working on google cloud easy.  
Just contains training script adapted from huggingface, lamb optimizer and script with the run parameters.  
huggingface training script takes care of distributed training and tpus, just need to run this in a torch_xla instance on gcloud.