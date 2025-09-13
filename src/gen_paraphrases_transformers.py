import pandas as pd
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
import os
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Qwen3Paraphraser:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        """
        Initialize the Qwen3 paraphraser.
        
        Args:
            model_name: The Qwen3 model to use
        """
        self.model_name = model_name
        
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        logger.info("Model loaded successfully!")
    
    def create_paraphrase_prompt(self, text: str) -> str:
        """Create a prompt for paraphrasing text in plain language."""
        prompt = f"""Reescreva o seguinte texto em uma linguagem simples e clara. Use frases mais curtas, palavras comuns e explicações diretas. Mantenha o mesmo significado, mas torne a leitura mais fácil. Retorne só texto simplificado, sem comentários adicionais.

Texto a simplificar: {text}

"""
        return prompt
    
    def generate_paraphrase(self, text: str) -> Tuple[str, str]:
        """
        Generate a paraphrase for a single text.
        
        Returns:
            Tuple of (thinking_content, paraphrased_content)
        """
        prompt = self.create_paraphrase_prompt(text)
        
        # Prepare messages in chat format
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Enable thinking mode
        )
        
        # Tokenize input
        model_inputs = self.tokenizer([formatted_text], return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=4096,
                temperature=0.7,
                top_p = 0.8,
                top_k = 20,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract only the generated tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content (similar to your original code)
        try:
            # Find the last occurrence of 151668 (</think> token)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        # Decode thinking and actual content
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        return thinking_content, content
    
    def process_batch(self, texts: List[str], batch_size: int = 4, save_thinking: bool = False) -> List[dict]:
        """
        Process a batch of texts and return paraphrases.
        
        Args:
            texts: List of texts to paraphrase
            batch_size: Number of texts to process at once
            save_thinking: Whether to save the thinking content
            
        Returns:
            List of dictionaries containing results
        """
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                try:
                    thinking, paraphrase = self.generate_paraphrase(text)
                    
                    result = {
                        'original_text': text,
                        'simple_text': paraphrase.strip()
                    }
                    
                    if save_thinking:
                        result['thinking_content'] = thinking.strip()
                    
                    results.append(result)
                    
                    # Log progress for long texts
                    if len(text) > 200:
                        logger.info(f"Processed long text ({len(text)} chars)")
                        
                except Exception as e:
                    logger.warning(f"Error processing text: {e}")
                    result = {
                        'original_text': text,
                        'simple_text': f"Error: Could not paraphrase - {str(e)}"
                    }
                    if save_thinking:
                        result['thinking_content'] = ""
                    results.append(result)
        
        return results

def load_parquet(file_path: str, text_column: str = 'text') -> pd.DataFrame:
    """Load parquet file and validate text column exists."""
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found. Available columns: {df.columns.tolist()}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading parquet file: {e}")
        raise

def save_results(results_df: pd.DataFrame, output_path: str):
    """Save the dataframe with original and paraphrased texts."""
    try:
        results_df.to_parquet(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Saved {len(results_df)} rows with columns: {results_df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Paraphrase text using Qwen3 model")
    parser.add_argument("--dataset", "-d", default="eduagarcia/LegalPT_dedup", help="dataset")
    parser.add_argument("--corpus", "-c", default="acordaos_tcu", help="corpus")
    parser.add_argument("--output", "-o", required=True, help="Output parquet file path")
    parser.add_argument("--text_column", "-t", default="text", help="Name of the text column")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-4B", help="Qwen3 model to use")
    parser.add_argument("--batch_size", "-b", type=int, default=2, help="Batch size for processing")
    parser.add_argument("--max_samples", default=None, type=int, help="Maximum number of rows to process (for testing)")
    parser.add_argument("--save_thinking", action="store_true", help="Save the thinking content as well")
    
    args = parser.parse_args()
    
     # Load dataset
    print(f"Loading dataset: {args.dataset}, config: {args.corpus}")
    try:
        ds = load_dataset(args.dataset, args.corpus)
        
        # Use train split if available, otherwise use the first available split
        if 'train' in ds:
            df = ds['train']
        else:
            split_name = list(ds.keys())[0]
            df = ds[split_name]
            print(f"Using split: {split_name}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Limit samples if specified
    if args.max_samples and len(df) > args.max_samples:
        df = df.select(range(args.max_samples))
        print(f"Limited to {args.max_samples} samples")
    
    print(f"Processing {len(df)} samples")
    
    # Initialize the paraphraser
    paraphraser = Qwen3Paraphraser(model_name=args.model)
    
    
    # Remove any null or empty texts
    original_length = len(df[args.text_column])
    valid_indices = []
    clean_texts = []
    
    for idx, sample in enumerate(df):
        text = sample[args.text_column]
        if pd.notna(text) and str(text).strip():
            valid_indices.append(idx)
            clean_texts.append(str(text).strip())
    
    if len(clean_texts) < original_length:
        logger.warning(f"Removed {original_length - len(clean_texts)} null/empty texts")
    
    # Process texts in batches
    logger.info(f"Processing {len(clean_texts)} texts in batches of {args.batch_size}")
    results = paraphraser.process_batch(clean_texts, batch_size=args.batch_size, save_thinking=args.save_thinking)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # If we filtered out some rows, we need to align with the original dataframe
    if len(valid_indices) < len(df):
        # Create a full results dataframe with all original rows
        full_results = []
        result_idx = 0
        
        for i in range(len(df)):
            if i in valid_indices:
                full_results.append(results[result_idx])
                result_idx += 1
            else:
                row = {
                    'original_text': df[args.text_column].iloc[i],
                    'simple_text': "Error: Empty or null text"
                }
                if args.save_thinking:
                    row['thinking_content'] = ""
                full_results.append(row)
        
        results_df = pd.DataFrame(full_results)
    
    # Add any additional columns from original dataframe
    for col in df.columns:
        if col != args.text_column and col not in results_df.columns:
            results_df[col] = df[col].values
    
    # Save results
    save_results(results_df, args.output)
    
    logger.info("Processing completed successfully!")
    
    # Print some sample results
    print("\n" + "="*80)
    print("SAMPLE RESULTS")
    print("="*80)
    
    for i in range(min(3, len(results_df))):
        print(f"\nSample {i+1}:")
        print(f"Original: {str(results_df['original_text'].iloc[i])[:150]}...")
        print(f"Simple:   {str(results_df['simple_text'].iloc[i])[:150]}...")
        
        if args.save_thinking and 'thinking_content' in results_df.columns:
            thinking = str(results_df['thinking_content'].iloc[i])
            if thinking and thinking != "":
                print(f"Thinking: {thinking[:100]}...")

if __name__ == "__main__":
    main()