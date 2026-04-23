import argparse
import pandas as pd
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast
import torch

def generate_predictions(model_path, test_csv, output_txt, max_length=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path} into {device}...")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained("../misc/char_tokenizer")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    df = pd.read_csv(test_csv)
    
    predictions = []
    
    for idx, row in df.iterrows():
        # Space-separated characters for character-level tokenization
        text = " ".join(list(str(row['name']))) + " " + tokenizer.eos_token
        
        inputs = tokenizer(text, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
            
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the spaces we inserted for tokenization
        decoded = decoded.replace(" ", "")
        predictions.append(decoded)
        
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)} samples...")
            
    with open(output_txt, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")
            
    print(f"Saved {len(predictions)} predictions to {output_txt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--test_csv", type=str, default="../data/raw/romanization-test-no-answer.csv")
    parser.add_argument("--output_txt", type=str, default="../romanization-test-pred.txt")
    args = parser.parse_args()
    
    generate_predictions(args.model_path, args.test_csv, args.output_txt)
