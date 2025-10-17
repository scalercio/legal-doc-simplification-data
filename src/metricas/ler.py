import pandas as pd

# Carrega o arquivo .parquet
df = pd.read_parquet("/home/camila/legal-doc-simplification-data/dataset_gov/gov_lang_br.parquet")

# Exibe as colunas
print(df.columns)
