import os
import torch
import pandas as pd
from tqdm import tqdm
import argparse
import re
from unsloth import FastLanguageModel

def load_model(model_path, use_trained=True):
    """
    Carrega o modelo treinado ou base
    
    Args:
        model_path: Caminho para o modelo treinado LoRA
        use_trained: Se True, usa modelo treinado. Se False, usa base
    """
    print(f"🚀 Carregando modelo...")
    
    max_seq_length = 4096
    dtype = None
    load_in_4bit = True
    
    if use_trained:
        # Carrega modelo base + adaptadores LoRA treinados
        print(f"📂 Carregando modelo treinado de: {model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
    else:
        # Carrega apenas modelo base
        print("📂 Carregando modelo base")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Qwen3-1.7B-bnb-4bit",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
    
    # Ativa modo inferência
    FastLanguageModel.for_inference(model)
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("✅ Modelo carregado!")
    return model, tokenizer


def generate_simplification_batch(model, tokenizer, texts, system_msg, max_new_tokens=4096, temperature=0.7, top_p=0.8):
    """
    Gera simplificações para um batch de textos
    
    Args:
        texts: Lista de textos a simplificar
    
    Returns:
        Lista de simplificações
    """
    # Prepara todos os prompts
    all_messages = []
    for text in texts:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Simplifique o texto a seguir, mas mantenha o sentido original. Retorne só o texto simplificado.\n\nTexto original: {text} \n\nTexto simplificado:\n\n"}
        ]
        all_messages.append(messages)
    
    # Tokeniza todos os prompts
    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        for messages in all_messages
    ]
    
    # Tokeniza em batch com padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3072  # Reserva espaço para geração
    ).to(model.device)
    
    # multi-EOS (pare no fim do turno do assistant)
    extra_eos = []
    for tok in ("<|im_end|>", "</s>", "<|end|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid != tokenizer.unk_token_id and tid is not None:
                extra_eos.append(tid)
        except Exception:
            pass
    EOS_IDS = list({tid for tid in ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []) + extra_eos})
    
    # Guarda tamanhos dos prompts para extrair apenas o texto gerado
    #prompt_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    input_lengths = inputs.attention_mask.sum(dim=1).tolist() #right padding, mas funciona pra left??(gpt)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            #temperature=0.3,
            #top_p=0.85,
            #top_k=40,
            # contrastive search leve:
            #penalty_alpha=0.6,        # (se disponível na sua versão HF)
            # anti-repetição:
            repetition_penalty=1.15,
            no_repeat_ngram_size=5,
            # outras proteções:
            length_penalty=0.8,       # favorece respostas um pouco mais curtas
            renormalize_logits=True,
            # parada
            eos_token_id=EOS_IDS,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            #min_new_tokens=20,
        )
    
    # Decodifica todas as saídas
    results = []
    for i, (output, input_len) in enumerate(zip(outputs, input_lengths)):
        #print(f"Prompt len = {input_len}, Output len = {len(output)}")
        #prompt_ids = inputs.input_ids[i] #left pad
        #print(len(prompt_ids))# maior valor do input len
        #if output.shape[0] >= prompt_ids.shape[0]:
        #    generated_tokens = output[len(prompt_ids):] #left pad
        #else:
        #    generated_tokens = output
        # Extrai apenas os tokens gerados (remove o prompt completo)
        #generated_tokens = output[len(inputs.input_ids[i]):]
        #prompt_len = int(inputs.attention_mask[i].sum().item())
        #generated_tokens = output[prompt_len:]
        #result_with_special = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        generated_tokens = outputs[i][inputs.input_ids[i].shape[0]:]
        
        # TRUNCAR NA PRIMEIRA OCORRÊNCIA DE QUALQUER EOS_ID (antes de decodificar):
        eos_pos = min((j for j,t in enumerate(generated_tokens.tolist()) if t in EOS_IDS), default=None)
        if eos_pos is not None:
            generated_tokens = generated_tokens[:eos_pos]
            
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        #if len(result.strip()) < 5:
        #    print(f"\n⚠️ RESPOSTA CURTA DETECTADA:")
        #    print(f"  Input len (real): {input_len}")
        #    print(f"  Input len (com padding): {len(prompt_ids)}")
        #    print(f"  Output len: {len(output)}")
        #    print(f"  Generated tokens: {len(generated_tokens)}")
        #    print(f"  Result with special: '{result_with_special[:100]}'")
        #    print(f"  Result: '{result}'")
        
        # Remove possíveis artefatos do chat template
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL|re.IGNORECASE)
        result = re.sub(r"([^\n]{0,30})\1{3,}", r"\1", result)                    # repetições curtinhas
        result = re.sub(r"(?:\b\w+\b\s+){0,3}(\b\w+\b)(?:\s+\1){2,}", r"\1", result)

        result = result.strip()
        
        # Remove tags de thinking se aparecerem
        if "</think>" in result:
            result = result.split("</think>")[-1].strip()
        
        # Remove possíveis prefixos "assistant" que podem vazar
        if result.startswith("assistant"):
            result = result[len("assistant"):].strip()
        
        results.append(result)
    
    return results


def process_parquet(input_file, output_file, model_path="./qwen3-1.7b-lora-final", 
                   use_trained=True, batch_size=4, max_new_tokens=4096,
                   temperature=0.7, top_p=0.8):
    """
    Processa arquivo parquet e adiciona coluna com simplificações
    
    Args:
        input_file: Caminho do arquivo parquet de entrada
        output_file: Caminho do arquivo parquet de saída
        model_path: Caminho do modelo treinado
        use_trained: Se True, usa modelo treinado. Se False, usa base
        batch_size: Número de textos a processar por vez
        max_new_tokens: Máximo de tokens a gerar
        temperature: Temperatura para sampling
        top_p: Top-p para nucleus sampling
    """
    # Carrega modelo
    model, tokenizer = load_model(model_path, use_trained=use_trained)
    
    # Carrega dados
    print(f"📂 Carregando arquivo: {input_file}")
    df = pd.read_parquet(input_file)
    #df = df.iloc[:16]
    
    if "original_text" not in df.columns:
        raise ValueError("❌ Arquivo não contém coluna 'original_text'")
    
    print(f"📊 Total de textos a processar: {len(df)}")
    print(f"📦 Batch size: {batch_size}")
    
    # System message (mesmo do treinamento)
    SYSTEM_MSG = "Você é um assistente simplificador de textos."
    
    # Gera simplificações em batches
    simplifications = []
    
    print("\n🔄 Gerando simplificações...")
    last_checkpoint = -1
    # Processa em batches
    for i in tqdm(range(0, len(df), batch_size)):
        try:
            # Pega batch de textos
            batch_end = min(i + batch_size, len(df))
            batch_texts = df["original_text"].iloc[i:batch_end].tolist()
            
            # Filtra textos vazios e guarda índices válidos
            valid_texts = []
            valid_indices = []
            for idx, text in enumerate(batch_texts):
                if pd.isna(text) or str(text).strip() == "":
                    continue
                valid_texts.append(str(text))
                valid_indices.append(idx)
            
            # Se não há textos válidos no batch, adiciona vazios
            if len(valid_texts) == 0:
                simplifications.extend([""] * len(batch_texts))
                continue
            
            # Gera simplificações para textos válidos
            batch_results = generate_simplification_batch(
                model, 
                tokenizer, 
                valid_texts, 
                SYSTEM_MSG,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            # Reconstrói lista completa com vazios nos lugares certos
            batch_simplifications = [""] * len(batch_texts)
            for valid_idx, result in zip(valid_indices, batch_results):
                batch_simplifications[valid_idx] = result
            
            simplifications.extend(batch_simplifications)
            
            # Limpa cache CUDA periodicamente
            if (i // batch_size + 1) % 50 == 0:
                torch.cuda.empty_cache()
                
            # Salvar progresso a cada 10%
            progress = (i + batch_size) / len(df)
            current_checkpoint = int(progress * 10)  # 0–10

            if current_checkpoint > last_checkpoint:
                test_df_partial = df.iloc[:len(simplifications)].copy()
                test_df_partial["qwen3"] = simplifications
                test_df_partial.to_csv(f"gov_bs1_partial_{current_checkpoint*10}_base.csv", index=False)
                print(f"💾 Progresso salvo em {current_checkpoint*10}% ({len(simplifications)} registros)")
                last_checkpoint = current_checkpoint
                
        except Exception as e:
            print(f"\n⚠️ Erro no batch {i}-{batch_end}: {e}")
            # Em caso de erro, adiciona vazios para o batch inteiro
            simplifications.extend([""] * (batch_end - i))
            torch.cuda.empty_cache()
            continue
    
    # Adiciona coluna ao dataframe
    df["qwen3"] = simplifications
    
    # Salva resultado
    print(f"\n💾 Salvando resultado em: {output_file}")
    df.to_parquet(output_file, index=False)
    
    # Estatísticas
    valid_simplifications = sum(1 for s in simplifications if s.strip() != "")
    print("\n" + "="*50)
    print("✅ Processamento completo!")
    print("="*50)
    print(f"📊 Total de textos: {len(df)}")
    print(f"✅ Simplificações geradas: {valid_simplifications}")
    print(f"❌ Erros/vazios: {len(df) - valid_simplifications}")
    print(f"📁 Arquivo salvo: {output_file}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera simplificações usando Qwen3")
    parser.add_argument("--input", type=str, required=True, help="Arquivo parquet de entrada")
    parser.add_argument("--output", type=str, required=True, help="Arquivo parquet de saída")
    parser.add_argument("--model_path", type=str, default="./qwen3-1.7b-unsloth-output/checkpoint-9000", 
                       help="Caminho do modelo treinado")
    parser.add_argument("--use_base", action="store_true", 
                       help="Usar modelo base ao invés do treinado")
    parser.add_argument("--batch_size", type=int, default=1, 
                       help="Tamanho do batch para inferência")
    parser.add_argument("--max_tokens", type=int, default=2048, 
                       help="Máximo de tokens a gerar")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Temperatura para sampling")
    parser.add_argument("--top_p", type=float, default=0.8, 
                       help="Top-p para nucleus sampling")
    
    args = parser.parse_args()
    
    process_parquet(
        input_file=args.input,
        output_file=args.output,
        model_path=args.model_path,
        use_trained=not args.use_base,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )