import pandas as pd
from utils import filtrar_por_flesch_diff, filtrar_por_similaridade, filtrar_por_diversidade

parquet_path = "/home/arthur/nlp/repo/simplification/legal-doc-simplification-data/datastf_paraphrases_v2.parquet"
save_file = parquet_path + ".final"
df=pd.read_parquet(parquet_path)
print(len(df))
df = filtrar_por_flesch_diff(df)
print(len(df))
df = filtrar_por_similaridade(df)
print(len(df))
df = filtrar_por_diversidade(df)
print(len(df))
df.to_parquet(save_file)
