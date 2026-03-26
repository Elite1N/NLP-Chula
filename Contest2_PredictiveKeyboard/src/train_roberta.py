import argparse
import os
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    RobertaConfig,
)
from torch.utils.data import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='distilroberta-base')
    parser.add_argument('--train_file', type=str, default='data/train.src.tok')
    parser.add_argument('--output_dir', type=str, default='experiments/roberta_baseline')
    parser.add_argument('--limit', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=128)
    
    args = parser.parse_args()

    # Handle paths
    if not os.path.exists(args.train_file):
        # try ../
        if os.path.exists('../' + args.train_file):
            args.train_file = '../' + args.train_file
            args.output_dir = '../' + args.output_dir
        else:
            raise FileNotFoundError(f"Could not find {args.train_file}")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading model: {args.model_name}")
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    print(f"Loading dataset from {args.train_file} (limit={args.limit})...")
    
    # We create a temporary file for LineByLineTextDataset if limit is set
    # because LineByLineTextDataset doesn't support limit natively well without reading all?
    # Actually it reads lines in memory.
    
    # Let's read lines manually and limit
    lines = []
    with open(args.train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            if line.strip():
                lines.append(line.strip())
                
    # Create a dummy file from lines
    temp_train_file = os.path.join(args.output_dir, "temp_train.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(temp_train_file, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
            
    print(f"Tokenizing {len(lines)} lines...")
    

    class SimpleTextDataset(Dataset):
        def __init__(self, tokenizer, file_path, block_size):
            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            
            self.examples = tokenizer(lines, truncation=True, padding='max_length', max_length=block_size)["input_ids"]
            
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, i):
            return torch.tensor(self.examples[i])

    dataset = SimpleTextDataset(
        tokenizer=tokenizer,
        file_path=temp_train_file,
        block_size=args.block_size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=5000, 
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Cleanup
    if os.path.exists(temp_train_file):
        os.remove(temp_train_file)

if __name__ == "__main__":
    main()
