from transformers import TFBertForSequenceClassification, BertTokenizer, TFTrainer, TFTrainingArguments




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
ds = load_dataset('text', data_files=['hf-train-tpu/preprossed_merged.txt'])
dataset = ds['train'].map(lambda examples: tokenizer(examples['text']), batched=True)
dataset.save_to_disk("path/of/my/dataset/directory")

reloaded_encoded_dataset = load_from_disk("path/of/my/dataset/directory")

dataset.set_format(type='tensorflow', columns=['text'])
features = {x: dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.max_len]) for x in ['input_ids', 'token_type_ids', 'attention_mask']}
tfdataset = tf.data.Dataset.from_tensor_slices(features).batch(32)


# set up model and tokenizer

model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15) #this does not tokenize, only gets tokenizer to know pad_token


training_args = TFTrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = TFTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)





import tensorflow_addons as tfa
import os
from transformers import TFBertForMaskedLM, BertTokenizer, TFTrainer, TFTrainingArguments, HfArgumentParser, set_seed, DataCollatorForLanguageModeling




def main():
    logger.info('entered main')
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()



    #no need on cloud?
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])

    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    training_args.strategy= strategy

    tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir, do_lower_case=False)


    if os.path.exists(os.path.splitext(args.train_data_file)[0]):
        dataset = load_from_disks(os.path.splitext(args.train_data_file)[0])
    else:
        ds = load_dataset('text', data_files=[args.train_data_file])
        dataset = ds['train'].map(lambda examples: tokenizer(examples['text']), batched=True)
        dataset.save_to_disk(os.path.splitext(args.train_data_file)[0])

    dataset.set_format(type='tensorflow', columns=['text'])
    features = {x: dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.max_len]) for x in ['input_ids', 'token_type_ids', 'attention_mask']}
    tfdataset = tf.data.Dataset.from_tensor_slices(features).batch(32)

    # Set seed
    set_seed(training_args.seed)


    trainer = TFTrainer(model, training_args, train_dataset, optimizer = (tfa.optimizers.LAMB, None))


    model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )

        logger.info('Beginning training')
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)