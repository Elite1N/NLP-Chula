import os
import argparse
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
import numpy as np

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate exact match and character error rate (CER)
    cer_metric = evaluate.load("cer")
    cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    exact_matches = sum(p.strip() == l.strip() for p, l in zip(decoded_preds, decoded_labels))
    em_score = exact_matches / len(decoded_labels)

    return {"cer": cer, "exact_match": em_score}

def preprocess_function(examples):
    # Insert spaces between characters to ensure char-level tokenization works perfectly
    source_texts = [" ".join(list(str(s))) for s in examples["source"]]
    target_texts = [" ".join(list(str(t))) for t in examples["target"]]
    
    model_inputs = tokenizer(source_texts, max_length=64, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_texts, max_length=64, truncation=True)
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_t5(model_name, dataset_dict, output_dir, learning_rate=5e-4, num_epochs=10):
    # Base minimal architecture for character-to-character baseline
    config = T5Config(
        vocab_size=len(tokenizer),
        d_model=128,          # Small hidden layer
        d_kv=32,
        d_ff=512,
        num_layers=4,         # Small number of layers
        num_decoder_layers=4,
        num_heads=4,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )
    
    model = T5ForConditionalGeneration(config)

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False, # Lower CER is better
        # report_to="wandb"  # Uncomment this to use wandb as requested in the pdf
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print(f"Starting training for {model_name}...")
    trainer.train()
    trainer.save_model(f"{output_dir}/best_model")
    print(f"Model saved to {output_dir}/best_model")
    
    return trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="t5_small_char")
    parser.add_argument("--use_silver", action="store_true", help="Whether to include silver data")
    args = parser.parse_args()
    
    # Load tokenizer
    global tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("../misc/char_tokenizer")
    # Make sure bos_token_id is not none for generation
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id

    # Load Data
    gold_df = pd.read_csv('../data/processed/processed-train-gold.csv')
    
    if args.use_silver and os.path.exists('../data/processed/processed-train-silver.csv'):
        silver_df = pd.read_csv('../data/processed/processed-train-silver.csv')
        train_df = pd.concat([gold_df, silver_df]).sample(frac=1).reset_index(drop=True)
    else:
        train_df = gold_df

    # Basic split: 90% train, 10% test (for internal validation)
    # The gold standard test set provided doesn't have answers, so we validate internally
    split_index = int(len(train_df) * 0.9)
    train_split = train_df.iloc[:split_index]
    test_split = train_df.iloc[split_index:]

    datasets = DatasetDict({
        "train": Dataset.from_pandas(train_split),
        "test": Dataset.from_pandas(test_split),
    })

    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    
    output_dir = f"../experiments/{args.model_name}"
    
    train_t5(args.model_name, tokenized_datasets, output_dir)
