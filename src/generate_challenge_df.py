import evaluate
import pandas as pd
from easse.sari import corpus_sari
csv_files_good = ['challenge_good_qwen2_5_results_batched.csv',
                  'challenge_good_bode_results_v2.csv']

csv_files_hard = ['challenge_hard_qwen2_5_results_batched_new_p2.csv', 
                  'challenge_hard_bode_results_v2.csv']
dfs =[]
for i, file in enumerate(csv_files_good):
    df_aux = pd.read_csv(file)
    if i==0:
        df = df_aux[['original_text' , 'paraphrase' ]].copy()
    if 'qwen' in file:
        df['qwen2.5'] = df_aux['qwen2.5_output']
    else:
        df['bode'] = df_aux['bode_output']
df['tipo'] = 'good'
print(df.columns)
dfs.append(df)

for i, file in enumerate(csv_files_hard):
    df_aux = pd.read_csv(file)
    if i==0:
        df = df_aux[['original_text' , 'paraphrase' ]].copy()
    if 'qwen' in file:
        df['qwen2.5'] = df_aux['qwen2.5_output']
    else:
        df['bode'] = df_aux['bode_output']
df['tipo'] = 'hard'
dfs.append(df)

df_final = pd.concat(dfs, ignore_index=True)
print(df_final.columns)
#df_final.to_parquet("llm_as_a_judge.parquet")
print("  - Calculando D-SARI...")

sari_result = corpus_sari(
    df_final[df_final['tipo']=='hard']["original_text"].tolist(),
    df_final[df_final['tipo']=='hard']["qwen2.5"].tolist(),
    [df_final[df_final['tipo']=='hard']["paraphrase"].tolist()]
)
print("Qwen hard:")
print(sari_result)

sari_result = corpus_sari(
    df_final[df_final['tipo']=='good']["original_text"].tolist(),
    df_final[df_final['tipo']=='good']["qwen2.5"].tolist(),
    [df_final[df_final['tipo']=='good']["paraphrase"].tolist()]
)
print("Qwen good:")
print(sari_result)

sari_result = corpus_sari(
    df_final[df_final['tipo']=='hard']["original_text"].tolist(),
    df_final[df_final['tipo']=='hard']["bode"].tolist(),
    [df_final[df_final['tipo']=='hard']["paraphrase"].tolist()]
)
print("bode hard:")
print(sari_result)

sari_result = corpus_sari(
    df_final[df_final['tipo']=='good']["original_text"].tolist(),
    df_final[df_final['tipo']=='good']["bode"].tolist(),
    [df_final[df_final['tipo']=='good']["paraphrase"].tolist()]
)
print("bode good:")
print(sari_result)

# To get individual components, you can use:
from easse.sari import get_corpus_sari_operation_scores

add_score, keep_score, del_score = get_corpus_sari_operation_scores(
    df_final[df_final['tipo']=='good']["original_text"].tolist(), df_final[df_final['tipo']=='good']["bode"].tolist(), [df_final[df_final['tipo']=='good']["paraphrase"].tolist()]
)

print(f"F1_add: {add_score}")
print(f"F1_keep: {keep_score}")
print(f"P_del: {del_score}")
