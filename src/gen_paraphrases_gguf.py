import pandas as pd
from datasets import load_dataset
from llama_cpp import Llama
import argparse
from tqdm import tqdm
import os
from typing import List, Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GGUFParaphraser:
    def __init__(self, model_path: str, n_ctx: int = 6144, n_threads: int = 8, n_gpu_layers: int = -1):
        """
        Initialize the GGUF paraphraser using llama-cpp-python.
        
        Args:
            model_path: Path to the .gguf model file
            n_ctx: Context window size
            n_threads: Number of CPU threads
        """
        self.model_path = model_path
        
        logger.info(f"Loading GGUF model: {model_path}")
        self.llm = Llama.from_pretrained(
            repo_id="Qwen/Qwen3-4B-GGUF",
            filename=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        logger.info("Model loaded successfully!")

    def create_paraphrase_prompt(self, text: str) -> str:
        """Create a prompt for paraphrasing text in plain language."""
        return f"""<|system|>
Você é um assistente simplificador de textos.
<|user|>
Simplifique o texto a seguir, mas mantenha o sentido original. Retorne só o texto simplificado.

Texto original: {text}

Texto simplificado:"""

    def generate_paraphrase(self, text: str) -> Tuple[str, str]:
        """
        Generate a paraphrase for a single text.
        
        Returns:
            Tuple of (thinking_content, paraphrased_content)
        """
        #prompt = self.create_paraphrase_prompt(text)
        #print('test')

        try:
            output = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Você é um assistente simplificador de textos."},
                    {"role": "user", "content": f"Simplifique o texto a seguir, mas mantenha o sentido original. Retorne só o texto simplificado.\n\nTexto original: {text}\n\nTexto simplificado: /no_think"}
                ],
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                max_tokens=4096,
                repeat_penalty=1.1,
            )
            paraphrased = output["choices"][0]["message"]["content"].strip()
            
            # Remove <think> tag if present
            if paraphrased.startswith("<think>\n\n</think>\n"):
                paraphrased = paraphrased[18:].strip()

        except Exception as e:
            logger.warning(f"Error generating paraphrase: {e}")
            paraphrased = f"Error: Could not paraphrase - {str(e)}"

        # llama-cpp doesn’t expose "thinking" vs. "content", so we return paraphrase only
        return "", paraphrased

    def process_batch(self, texts: List[str], batch_size: int = 4, output_file: str = None, save_thinking: bool = False) -> List[dict]:
        """Process a batch of texts and return paraphrases."""
        results = []
        processed_count = 0

        for i, text in tqdm(enumerate(texts), desc="Processing docs"):
            #batch = texts[i:i + batch_size]
            #for text in batch:
            _ , paraphrase = self.generate_paraphrase(text)
            if paraphrase:
                results.append({
                    'original_text': text,
                    'simple_text': paraphrase,
                    'sample_id': i
                })
                processed_count += 1
            else:
                print(f"Failed to generate paraphrase for sample {i}")
                
            # Save intermediate results
            if len(results) % batch_size == 0 and results:
                save_intermediate_results(results, output_file, processed_count)
        return results

def load_parquet(file_path: str, text_column: str = "text") -> pd.DataFrame:
    """Load parquet file and validate text column exists."""
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} rows from {file_path}")
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available columns: {df.columns.tolist()}")
    return df

def save_intermediate_results(results: List[Dict], output_file: str, processed_count: int):
    """Save intermediate results to avoid losing progress"""
    intermediate_file = output_file.replace('.parquet', f'_intermediate_{processed_count}.parquet')
    df = pd.DataFrame(results)
    df.to_parquet(intermediate_file, index=False)
    print(f"\nSaved intermediate results: {len(results)} pairs to {intermediate_file}")

def save_results(results_df: pd.DataFrame, output_path: str):
    """Save the dataframe with original and paraphrased texts."""
    results_df.to_parquet(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Saved {len(results_df)} rows with columns: {results_df.columns.tolist()}")

def main():
    parser = argparse.ArgumentParser(description="Paraphrase text using GGUF model with llama-cpp-python")
    parser.add_argument("--dataset", "-d", default="eduagarcia/LegalPT_dedup", help="dataset")
    parser.add_argument("--corpus", "-c", default="acordaos_tcu", help="corpus")
    parser.add_argument("--output", "-o", required=True, help="Output parquet file path")
    parser.add_argument("--text_column", "-t", default="text", help="Name of the text column")
    parser.add_argument("--model_path", "-m", required=True, help="Path to GGUF model file")
    parser.add_argument("--batch_size", "-b", type=int, default=10000, help="Batch size for saving")
    parser.add_argument("--max_samples", default=None, type=int, help="Maximum number of rows to process (for testing)")
    parser.add_argument("--save_thinking", action="store_true", help="Save the thinking content as well (dummy for GGUF)")
    
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}, config: {args.corpus}")
    ds = load_dataset(args.dataset, args.corpus)
    df = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    if args.max_samples and len(df) > args.max_samples:
        df = df.select(range(args.max_samples))
        print(f"Limited to {args.max_samples} samples")

    # Clean texts
    clean_texts = [str(x).strip() for x in df[args.text_column] if pd.notna(x) and str(x).strip()]
    logger.info(f"Saving {len(clean_texts)} texts in batches of {args.batch_size}")

    paraphraser = GGUFParaphraser(model_path=args.model_path)
    results = paraphraser.process_batch(clean_texts, batch_size=args.batch_size, output_file=args.output, save_thinking=args.save_thinking)

    results_df = pd.DataFrame(results)
    save_results(results_df, args.output)

if __name__ == "__main__":
    main()
