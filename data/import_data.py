from google.cloud import bigquery
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, classification_report, roc_curve)
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

# ===========================================================================
# ETAPA 1: CARREGAR DADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"MODELO V5 - LIGHTGBM (GRADIENT BOOSTING OTIMIZADO)")
print(f"{'='*70}")
print(f"ETAPA 1: CARREGANDO DATASET COM AMOSTRAGEM ALEATÓRIA")
print(f"{'='*70}")

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` 
    LIMIT 200000
"""

print("Carregando 200,000 amostras...")
df = client.query(query).to_dataframe()

df.to_csv('sampled_dataset.csv', index=False)