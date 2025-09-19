import pandas as pd
from utils import calcular_similaridade_sliding, calcular_delta_flesch

parquet_path = "/home/arthur/nlp/repo/simplification/legal-doc-simplification-data/data/museum/museum.parquet"
df=pd.read_parquet(parquet_path)
similarity=calcular_similaridade_sliding(df)
deltas_flesch=calcular_delta_flesch(df)
print(min(similarity))
print(sum(deltas_flesch)/len(deltas_flesch))

