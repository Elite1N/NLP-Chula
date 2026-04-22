import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
import pandas as pd

def train_character_tokenizer(dataset_files, save_path):
    """
    Trains a character-level Tokenizer from scratch for a Seq2Seq model.
    A simple approach is to train BPE with initial alphabet containing all characters and vocab size equal to the character set.
    """
    # 1. Initialize empty tokenizer with BPE
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # We don't pre-tokenize by words because we want strictly character level
    # If we skip pre-tokenizer, BPE will find frequent pairs. 
    # To restrict it to char-level, we just set vocab_size exactly equal to the initial alphabet size.
    # Alternatively, Hugging Face recommends training BPE with small vocab size
    # Let's collect all characters
    chars = set()
    characters_data = []
    
    # Read the data and split by characters explicitly
    for f in dataset_files:
        df = pd.read_csv(f)
        for val in df['source'].tolist() + df['target'].tolist():
            val = str(val)
            for c in list(val):
                chars.add(c)
                
    initial_alphabet = list(chars)
    # The special tokens that T5 uses are unk, pad, bos, eos
    vocab_size = len(initial_alphabet) + 5
    
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[MASK]"],
        initial_alphabet=initial_alphabet,
        vocab_size=vocab_size # Just big enough for the characters, no subwords formed
    )
    
    # For training we generate text lines where characters are separated by whitespaces
    with open("temp_corpus.txt", "w", encoding="utf-8") as f:
        for val_f in dataset_files:
            df = pd.read_csv(val_f)
            for val in df['source'].tolist() + df['target'].tolist():
                f.write(" ".join(list(str(val))) + "\n")
                
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.train(["temp_corpus.txt"], trainer)
    os.remove("temp_corpus.txt")
    
    # Save and wrap in PreTrainedTokenizerFast
    tokenizer.save("char_tokenizer.json")
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="char_tokenizer.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        mask_token="[MASK]"
    )
    
    os.makedirs(save_path, exist_ok=True)
    hf_tokenizer.save_pretrained(save_path)
    print(f"Tokenizer saved to {save_path} with vocab length: {len(hf_tokenizer)}")
    return hf_tokenizer

if __name__ == "__main__":
    train_files = ['../data/processed/processed-train-gold.csv']
    if os.path.exists('../data/processed/processed-train-silver.csv'):
        train_files.append('../data/processed/processed-train-silver.csv')
    
    train_character_tokenizer(train_files, "../misc/char_tokenizer")
