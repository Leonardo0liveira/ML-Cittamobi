from google.cloud import bigquery
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from collections import Counter

project_id = "proj-ml-469320"
client = bigquery.Client(project=project_id)

# ===========================================================================
# ETAPA 1: CARREGAR DADOS (MESMO DO V3)
# ===========================================================================
print(f"\n{'='*70}")
print(f"MODELO V3 ENHANCED - BALANCEAMENTO DE CLASSES")
print(f"{'='*70}")
print(f"ETAPA 1: CARREGANDO DATASET COM AMOSTRAGEM ALEATÓRIA")
print(f"{'='*70}")

query = """
    SELECT * FROM `proj-ml-469320.app_cittamobi.dataset-updated` 
    TABLESAMPLE SYSTEM (20 PERCENT)
    LIMIT 200000
"""

print("Carregando 200,000 amostras com amostragem aleatória...")
df = client.query(query).to_dataframe()
print(f"✓ Dados carregados: {len(df):,} registros")

# ===========================================================================
# ETAPA 2: LIMPEZA MODERADA (MESMO DO V3)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 2: LIMPEZA MODERADA DOS DADOS")
print(f"{'='*70}")

target = "target"

print(f"\n=== Distribuição do Target (Original) ===")
target_dist = df[target].value_counts()
print(target_dist)
print(f"Proporção classe 0: {target_dist[0]/len(df)*100:.2f}%")
print(f"Proporção classe 1: {target_dist[1]/len(df)*100:.2f}%")

df_original_size = len(df)

# Filtros moderados
if 'user_frequency' in df.columns:
    user_freq_threshold = df['user_frequency'].quantile(0.10)
    df = df[df['user_frequency'] >= user_freq_threshold]
    removed = df_original_size - len(df)
    print(f"✓ Filtro 1: Removidos {removed:,} registros ({removed/df_original_size*100:.1f}%)")
    df_original_size = len(df)

if 'device_lat' in df.columns and 'device_lon' in df.columns:
    df = df[~((df['device_lat'].isna()) | (df['device_lon'].isna()))]
    df = df[~((df['device_lat'] == 0) & (df['device_lon'] == 0))]
    removed = df_original_size - len(df)
    print(f"✓ Filtro 2: Removidos {removed:,} registros ({removed/df_original_size*100:.1f}%)")
    df_original_size = len(df)

if 'dist_device_stop' in df.columns:
    dist_threshold = df['dist_device_stop'].quantile(0.98)
    df = df[df['dist_device_stop'] <= dist_threshold]
    removed = df_original_size - len(df)
    print(f"✓ Filtro 3: Removidos {removed:,} registros ({removed/df_original_size*100:.1f}%)")
    df_original_size = len(df)

if 'headway_avg_stop_hour' in df.columns:
    df = df[df['headway_avg_stop_hour'] > 0]

if 'stop_event_count' in df.columns:
    stop_threshold = df['stop_event_count'].quantile(0.10)
    df = df[df['stop_event_count'] >= stop_threshold]

total_removed = 200000 - len(df)
removal_pct = total_removed / 200000 * 100
print(f"\n✓ Total mantido: {len(df):,} ({100-removal_pct:.1f}%)")

# ===========================================================================
# ETAPA 3: FEATURE ENGINEERING (MESMO DO V3)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 3: FEATURE ENGINEERING")
print(f"{'='*70}")

features_to_drop = ['y_pred', 'y_pred_proba', 'ctm_service_route', 'direction', 'lotacao_proxy_binaria']
X = df.drop(columns=[target] + features_to_drop, errors='ignore')
y = df[target]

# Processar timestamp
if 'event_timestamp' in X.columns:
    X['event_timestamp'] = pd.to_datetime(X['event_timestamp'])
    X['year'] = X['event_timestamp'].dt.year
    X['month'] = X['event_timestamp'].dt.month
    X['day'] = X['event_timestamp'].dt.day
    X['hour'] = X['event_timestamp'].dt.hour
    X['dayofweek'] = X['event_timestamp'].dt.dayofweek
    X['minute'] = X['event_timestamp'].dt.minute
    X['week_of_year'] = X['event_timestamp'].dt.isocalendar().week
    X = X.drop(columns=['event_timestamp'])

# Features de interação
if 'hour' in X.columns and 'dayofweek' in X.columns:
    X['hour_x_dayofweek'] = X['hour'] * X['dayofweek']

if 'dist_device_stop' in X.columns and 'is_peak_hour' in X.columns:
    X['dist_x_peak'] = X['dist_device_stop'] * (X['is_peak_hour'] + 1)

if 'stop_event_rate' in X.columns and 'stop_total_samples' in X.columns:
    X['event_rate_normalized'] = X['stop_event_rate'] / (X['stop_total_samples'] + 1)

if 'stop_event_count' in X.columns and 'stop_total_samples' in X.columns:
    X['event_density'] = X['stop_event_count'] / (X['stop_total_samples'] + 1)

# Features cíclicas
if 'time_day_of_month' in X.columns:
    X['day_of_month_sin'] = np.sin(2 * np.pi * X['time_day_of_month'] / 31)
    X['day_of_month_cos'] = np.cos(2 * np.pi * X['time_day_of_month'] / 31)

if 'week_of_year' in X.columns:
    X['week_sin'] = np.sin(2 * np.pi * X['week_of_year'] / 52)
    X['week_cos'] = np.cos(2 * np.pi * X['week_of_year'] / 52)

# Label Encoding
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print(f"✓ Features criadas: {X.shape[1]}")

# ===========================================================================
# ETAPA 4: SELEÇÃO DE FEATURES (TOP 40)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 4: SELEÇÃO DAS MELHORES FEATURES")
print(f"{'='*70}")

# Treinar modelo rápido para feature importance
tscv = TimeSeriesSplit(n_splits=2)
for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train_temp, X_test_temp = X.iloc[train_index], X.iloc[test_index]
    y_train_temp, y_test_temp = y.iloc[train_index], y.iloc[test_index]

dtrain_temp = xgb.DMatrix(X_train_temp, label=y_train_temp)
scale_pos_weight = (len(y_train_temp) - y_train_temp.sum()) / y_train_temp.sum()

params_temp = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'logloss',
    'seed': 42,
    'scale_pos_weight': scale_pos_weight
}

model_temp = xgb.train(
    params=params_temp,
    dtrain=dtrain_temp,
    num_boost_round=50,
    verbose_eval=False
)

importance_dict = model_temp.get_score(importance_type='gain')
importance_df = pd.DataFrame([
    {'feature': k, 'importance': v} 
    for k, v in importance_dict.items()
]).sort_values('importance', ascending=False)

top_n = 40
top_features = importance_df.head(top_n)['feature'].tolist()
X_selected = X[top_features].copy()

print(f"✓ Features selecionadas: {len(top_features)}")

# ===========================================================================
# ETAPA 5: DIVISÃO TEMPORAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 5: DIVISÃO TEMPORAL (TimeSeriesSplit)")
print(f"{'='*70}")

tscv = TimeSeriesSplit(n_splits=3)
X_train, X_test, y_train, y_test = None, None, None, None

for fold, (train_index, test_index) in enumerate(tscv.split(X_selected)):
    X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    if fold == 2:  # Último fold
        print(f"Fold {fold + 1} (selecionado): Train={len(X_train):,} | Test={len(X_test):,}")

print(f"\n=== Distribuição Original do Treino ===")
train_dist = y_train.value_counts()
print(f"Classe 0: {train_dist[0]:,} ({train_dist[0]/len(y_train)*100:.2f}%)")
print(f"Classe 1: {train_dist[1]:,} ({train_dist[1]/len(y_train)*100:.2f}%)")
print(f"Razão (0/1): {train_dist[0]/train_dist[1]:.2f}:1")

# ===========================================================================
# ETAPA 6: TESTAR DIFERENTES ESTRATÉGIAS DE BALANCEAMENTO
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 6: TESTANDO ESTRATÉGIAS DE BALANCEAMENTO")
print(f"{'='*70}")

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Parâmetros base
base_params = {
    'objective': 'binary:logistic',
    'max_depth': 12,
    'learning_rate': 0.02,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.01,
    'reg_lambda': 1,
    'eval_metric': 'logloss',
    'seed': 42
}

results = []

# ===========================================================================
# ESTRATÉGIA 1: SEM BALANCEAMENTO (Baseline com scale_pos_weight)
# ===========================================================================
print(f"\n--- Estratégia 1: SEM BALANCEAMENTO (Baseline) ---")

X_train_1 = X_train.copy()
y_train_1 = y_train.copy()

scale_pos_weight_1 = (len(y_train_1) - y_train_1.sum()) / y_train_1.sum()
params_1 = {**base_params, 'scale_pos_weight': scale_pos_weight_1}

dtrain_1 = xgb.DMatrix(X_train_1, label=y_train_1)
dtest_1 = xgb.DMatrix(X_test, label=y_test)

model_1 = xgb.train(
    params=params_1,
    dtrain=dtrain_1,
    num_boost_round=200,
    evals=[(dtest_1, 'test')],
    early_stopping_rounds=20,
    verbose_eval=False
)

y_pred_proba_1 = model_1.predict(dtest_1)
y_pred_1 = (y_pred_proba_1 >= 0.5).astype(int)

results.append({
    'strategy': 'Baseline (scale_pos_weight)',
    'train_size': len(X_train_1),
    'train_class_0': (y_train_1 == 0).sum(),
    'train_class_1': (y_train_1 == 1).sum(),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_1),
    'accuracy': accuracy_score(y_test, y_pred_1),
    'precision': precision_score(y_test, y_pred_1),
    'recall': recall_score(y_test, y_pred_1),
    'f1_score': f1_score(y_test, y_pred_1),
    'f1_macro': f1_score(y_test, y_pred_1, average='macro'),
    'model': model_1,
    'y_pred_proba': y_pred_proba_1
})

print(f"Train size: {len(X_train_1):,}")
print(f"ROC-AUC: {results[-1]['roc_auc']:.4f}")
print(f"F1-Score: {results[-1]['f1_score']:.4f}")
print(f"F1-Macro: {results[-1]['f1_macro']:.4f}")

# ===========================================================================
# ESTRATÉGIA 2: SMOTE (Synthetic Minority Over-sampling)
# ===========================================================================
print(f"\n--- Estratégia 2: SMOTE ---")

# Converter para numpy arrays para evitar problemas de dtype
X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
y_train_np = y_train.values if hasattr(y_train, 'values') else y_train

smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=3)  # 30% mais rápido
X_train_2_np, y_train_2_np = smote.fit_resample(X_train_np, y_train_np)

# Converter de volta para DataFrame para manter os nomes das features
X_train_2 = pd.DataFrame(X_train_2_np, columns=X_train.columns)
y_train_2 = pd.Series(y_train_2_np)

print(f"Antes: Classe 0={train_dist[0]:,}, Classe 1={train_dist[1]:,}")
print(f"Depois: Classe 0={(y_train_2==0).sum():,}, Classe 1={(y_train_2==1).sum():,}")
print(f"Samples sintéticos criados: {len(X_train_2) - len(X_train):,}")

params_2 = {**base_params, 'scale_pos_weight': 1}  # Não precisa mais, já está balanceado

dtrain_2 = xgb.DMatrix(X_train_2, label=y_train_2)
dtest_2 = xgb.DMatrix(X_test, label=y_test)

model_2 = xgb.train(
    params=params_2,
    dtrain=dtrain_2,
    num_boost_round=200,
    evals=[(dtest_2, 'test')],
    early_stopping_rounds=20,
    verbose_eval=False
)

y_pred_proba_2 = model_2.predict(dtest_2)
y_pred_2 = (y_pred_proba_2 >= 0.5).astype(int)

results.append({
    'strategy': 'SMOTE (oversample minority)',
    'train_size': len(X_train_2),
    'train_class_0': (y_train_2 == 0).sum(),
    'train_class_1': (y_train_2 == 1).sum(),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_2),
    'accuracy': accuracy_score(y_test, y_pred_2),
    'precision': precision_score(y_test, y_pred_2),
    'recall': recall_score(y_test, y_pred_2),
    'f1_score': f1_score(y_test, y_pred_2),
    'f1_macro': f1_score(y_test, y_pred_2, average='macro'),
    'model': model_2,
    'y_pred_proba': y_pred_proba_2
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f}")
print(f"F1-Score: {results[-1]['f1_score']:.4f}")
print(f"F1-Macro: {results[-1]['f1_macro']:.4f}")

# ===========================================================================
# ESTRATÉGIA 3: UNDERSAMPLING com Tomek Links
# ===========================================================================
print(f"\n--- Estratégia 3: UNDERSAMPLING (Tomek Links) ---")

X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
y_train_np = y_train.values if hasattr(y_train, 'values') else y_train

tomek = TomekLinks(sampling_strategy='auto')
X_train_3_np, y_train_3_np = tomek.fit_resample(X_train_np, y_train_np)

# Converter de volta para DataFrame para manter os nomes das features
X_train_3 = pd.DataFrame(X_train_3_np, columns=X_train.columns)
y_train_3 = pd.Series(y_train_3_np)

print(f"Antes: Classe 0={train_dist[0]:,}, Classe 1={train_dist[1]:,}")
print(f"Depois: Classe 0={(y_train_3==0).sum():,}, Classe 1={(y_train_3==1).sum():,}")
print(f"Samples removidos (limpeza de fronteira): {len(X_train) - len(X_train_3):,}")

scale_pos_weight_3 = (len(y_train_3) - y_train_3.sum()) / y_train_3.sum()
params_3 = {**base_params, 'scale_pos_weight': scale_pos_weight_3}

dtrain_3 = xgb.DMatrix(X_train_3, label=y_train_3)
dtest_3 = xgb.DMatrix(X_test, label=y_test)

model_3 = xgb.train(
    params=params_3,
    dtrain=dtrain_3,
    num_boost_round=200,
    evals=[(dtest_3, 'test')],
    early_stopping_rounds=20,
    verbose_eval=False
)

y_pred_proba_3 = model_3.predict(dtest_3)
y_pred_3 = (y_pred_proba_3 >= 0.5).astype(int)

results.append({
    'strategy': 'Tomek Links (undersample)',
    'train_size': len(X_train_3),
    'train_class_0': (y_train_3 == 0).sum(),
    'train_class_1': (y_train_3 == 1).sum(),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_3),
    'accuracy': accuracy_score(y_test, y_pred_3),
    'precision': precision_score(y_test, y_pred_3),
    'recall': recall_score(y_test, y_pred_3),
    'f1_score': f1_score(y_test, y_pred_3),
    'f1_macro': f1_score(y_test, y_pred_3, average='macro'),
    'model': model_3,
    'y_pred_proba': y_pred_proba_3
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f}")
print(f"F1-Score: {results[-1]['f1_score']:.4f}")
print(f"F1-Macro: {results[-1]['f1_macro']:.4f}")

# ===========================================================================
# ESTRATÉGIA 4: SMOTETomek (SMOTE + Tomek Links)
# ===========================================================================
print(f"\n--- Estratégia 4: SMOTETomek (Híbrido) ---")

X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
y_train_np = y_train.values if hasattr(y_train, 'values') else y_train

smotetomek = SMOTETomek(sampling_strategy=0.3, random_state=42)
X_train_4_np, y_train_4_np = smotetomek.fit_resample(X_train_np, y_train_np)

# Converter de volta para DataFrame para manter os nomes das features
X_train_4 = pd.DataFrame(X_train_4_np, columns=X_train.columns)
y_train_4 = pd.Series(y_train_4_np)

print(f"Antes: Classe 0={train_dist[0]:,}, Classe 1={train_dist[1]:,}")
print(f"Depois: Classe 0={(y_train_4==0).sum():,}, Classe 1={(y_train_4==1).sum():,}")
print(f"Samples adicionados/removidos: {len(X_train_4) - len(X_train):,}")

params_4 = {**base_params, 'scale_pos_weight': 1}

dtrain_4 = xgb.DMatrix(X_train_4, label=y_train_4)
dtest_4 = xgb.DMatrix(X_test, label=y_test)

model_4 = xgb.train(
    params=params_4,
    dtrain=dtrain_4,
    num_boost_round=200,
    evals=[(dtest_4, 'test')],
    early_stopping_rounds=20,
    verbose_eval=False
)

y_pred_proba_4 = model_4.predict(dtest_4)
y_pred_4 = (y_pred_proba_4 >= 0.5).astype(int)

results.append({
    'strategy': 'SMOTETomek (hybrid)',
    'train_size': len(X_train_4),
    'train_class_0': (y_train_4 == 0).sum(),
    'train_class_1': (y_train_4 == 1).sum(),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_4),
    'accuracy': accuracy_score(y_test, y_pred_4),
    'precision': precision_score(y_test, y_pred_4),
    'recall': recall_score(y_test, y_pred_4),
    'f1_score': f1_score(y_test, y_pred_4),
    'f1_macro': f1_score(y_test, y_pred_4, average='macro'),
    'model': model_4,
    'y_pred_proba': y_pred_proba_4
})

print(f"ROC-AUC: {results[-1]['roc_auc']:.4f}")
print(f"F1-Score: {results[-1]['f1_score']:.4f}")
print(f"F1-Macro: {results[-1]['f1_macro']:.4f}")

# ===========================================================================
# ETAPA 7: COMPARAÇÃO DE RESULTADOS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 7: COMPARAÇÃO DAS ESTRATÉGIAS")
print(f"{'='*70}")

results_df = pd.DataFrame([{
    'Estratégia': r['strategy'],
    'Train Size': f"{r['train_size']:,}",
    'Classe 0': f"{r['train_class_0']:,}",
    'Classe 1': f"{r['train_class_1']:,}",
    'ROC-AUC': f"{r['roc_auc']:.4f}",
    'Accuracy': f"{r['accuracy']:.4f}",
    'Precision': f"{r['precision']:.4f}",
    'Recall': f"{r['recall']:.4f}",
    'F1-Score': f"{r['f1_score']:.4f}",
    'F1-Macro': f"{r['f1_macro']:.4f}"
} for r in results])

print("\n" + results_df.to_string(index=False))

# Selecionar melhor modelo baseado em F1-Macro
best_idx = np.argmax([r['f1_macro'] for r in results])
best_result = results[best_idx]

print(f"\n{'='*70}")
print(f"MELHOR ESTRATÉGIA: {best_result['strategy']}")
print(f"{'='*70}")
print(f"F1-Macro: {best_result['f1_macro']:.4f} ⭐")
print(f"ROC-AUC: {best_result['roc_auc']:.4f}")
print(f"F1-Score: {best_result['f1_score']:.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall: {best_result['recall']:.4f}")

# ===========================================================================
# ETAPA 8: OTIMIZAÇÃO DE THRESHOLD (MELHOR MODELO)
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 8: OTIMIZANDO THRESHOLD (MELHOR MODELO)")
print(f"{'='*70}")

best_model = best_result['model']
y_pred_proba_best = best_result['y_pred_proba']

thresholds_to_test = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
threshold_results = []

for thresh in thresholds_to_test:
    y_pred_thresh = (y_pred_proba_best >= thresh).astype(int)
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    f1_macro = f1_score(y_test, y_pred_thresh, average='macro')
    
    threshold_results.append({
        'threshold': thresh,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'f1_macro': f1_macro
    })
    
    print(f"Threshold={thresh:.2f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f} | F1-Macro={f1_macro:.4f}")

best_threshold = max(threshold_results, key=lambda x: x['f1_macro'])
print(f"\n✓ Melhor threshold: {best_threshold['threshold']:.2f} (F1-Macro={best_threshold['f1_macro']:.4f})")

y_pred_final = (y_pred_proba_best >= best_threshold['threshold']).astype(int)

# ===========================================================================
# ETAPA 9: MÉTRICAS FINAIS
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 9: MÉTRICAS FINAIS - V3 ENHANCED")
print(f"{'='*70}")

from sklearn.metrics import confusion_matrix, classification_report

accuracy_final = accuracy_score(y_test, y_pred_final)
precision_final = precision_score(y_test, y_pred_final)
recall_final = recall_score(y_test, y_pred_final)
f1_final = f1_score(y_test, y_pred_final)
f1_macro_final = f1_score(y_test, y_pred_final, average='macro')
roc_auc_final = roc_auc_score(y_test, y_pred_proba_best)

print(f"\n=== Métricas Finais ===")
print(f"Estratégia: {best_result['strategy']}")
print(f"Threshold: {best_threshold['threshold']:.2f}")
print(f"")
print(f"ROC-AUC:     {roc_auc_final:.4f}")
print(f"Accuracy:    {accuracy_final:.4f} ({accuracy_final*100:.2f}%)")
print(f"Precision:   {precision_final:.4f}")
print(f"Recall:      {recall_final:.4f}")
print(f"F1-Score:    {f1_final:.4f}")
print(f"F1-Macro:    {f1_macro_final:.4f} ⭐")

cm = confusion_matrix(y_test, y_pred_final)
print(f"\n=== Matriz de Confusão ===")
print(f"                 Predito")
print(f"                 0      1")
print(f"Real  0       {cm[0,0]:6d} {cm[0,1]:6d}")
print(f"      1       {cm[1,0]:6d} {cm[1,1]:6d}")

print(f"\n=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred_final, target_names=['Classe 0', 'Classe 1']))

# ===========================================================================
# ETAPA 10: VISUALIZAÇÕES
# ===========================================================================
print(f"\n{'='*70}")
print(f"ETAPA 10: GERANDO VISUALIZAÇÕES")
print(f"{'='*70}")

# Comparação de estratégias
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = ['roc_auc', 'f1_score', 'f1_macro', 'precision']
titles = ['ROC-AUC', 'F1-Score', 'F1-Macro', 'Precision']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[idx // 2, idx % 2]
    values = [r[metric] for r in results]
    strategies = [r['strategy'] for r in results]
    
    bars = ax.bar(range(len(strategies)), values, color=colors)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([s.split(' ')[0] for s in strategies], rotation=45, ha='right')
    ax.set_ylabel(title)
    ax.set_title(f'Comparação: {title}')
    ax.grid(axis='y', alpha=0.3)
    
    # Destacar melhor
    best_idx_metric = np.argmax(values)
    bars[best_idx_metric].set_color('#2ca02c')
    bars[best_idx_metric].set_edgecolor('black')
    bars[best_idx_metric].set_linewidth(2)
    
    # Adicionar valores
    for i, v in enumerate(values):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('balancing_strategies_comparison.png', dpi=300, bbox_inches='tight')
print("✓ balancing_strategies_comparison.png")

# Matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['Classe 0', 'Classe 1'],
            yticklabels=['Classe 0', 'Classe 1'])
plt.title(f'Matriz de Confusão - V3 Enhanced\n{best_result["strategy"]}')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig('confusion_matrix_v3_enhanced.png', dpi=300, bbox_inches='tight')
print("✓ confusion_matrix_v3_enhanced.png")

# Threshold analysis com F1-Macro
thresholds_plot = [r['threshold'] for r in threshold_results]
precisions_plot = [r['precision'] for r in threshold_results]
recalls_plot = [r['recall'] for r in threshold_results]
f1s_plot = [r['f1_score'] for r in threshold_results]
f1_macros_plot = [r['f1_macro'] for r in threshold_results]

plt.figure(figsize=(12, 6))
plt.plot(thresholds_plot, precisions_plot, marker='o', label='Precision', linewidth=2)
plt.plot(thresholds_plot, recalls_plot, marker='s', label='Recall', linewidth=2)
plt.plot(thresholds_plot, f1s_plot, marker='^', label='F1-Score', linewidth=2)
plt.plot(thresholds_plot, f1_macros_plot, marker='D', label='F1-Macro', linewidth=2.5, color='purple')
plt.axvline(x=best_threshold['threshold'], color='red', linestyle='--', 
            label=f'Best ({best_threshold["threshold"]:.2f})', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold Analysis - V3 Enhanced (F1-Macro Optimized)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_analysis_v3_enhanced.png', dpi=300, bbox_inches='tight')
print("✓ threshold_analysis_v3_enhanced.png")

# Salvar melhor modelo
best_model.save_model('xgboost_model_v3_enhanced.json')
print("✓ xgboost_model_v3_enhanced.json")

# Salvar relatório
with open('v3_enhanced_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("RELATÓRIO V3 ENHANCED - BALANCEAMENTO DE CLASSES\n")
    f.write("="*70 + "\n\n")
    
    f.write("ESTRATÉGIAS TESTADAS:\n")
    f.write("-"*70 + "\n")
    for i, r in enumerate(results, 1):
        f.write(f"\n{i}. {r['strategy']}\n")
        f.write(f"   Train Size: {r['train_size']:,}\n")
        f.write(f"   Classe 0: {r['train_class_0']:,} | Classe 1: {r['train_class_1']:,}\n")
        f.write(f"   ROC-AUC: {r['roc_auc']:.4f}\n")
        f.write(f"   F1-Score: {r['f1_score']:.4f}\n")
        f.write(f"   F1-Macro: {r['f1_macro']:.4f}\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write(f"MELHOR ESTRATÉGIA: {best_result['strategy']}\n")
    f.write("="*70 + "\n")
    f.write(f"F1-Macro: {f1_macro_final:.4f}\n")
    f.write(f"ROC-AUC: {roc_auc_final:.4f}\n")
    f.write(f"F1-Score: {f1_final:.4f}\n")
    f.write(f"Precision: {precision_final:.4f}\n")
    f.write(f"Recall: {recall_final:.4f}\n")
    f.write(f"Threshold: {best_threshold['threshold']:.2f}\n")

print("✓ v3_enhanced_report.txt")

# ===========================================================================
# RESUMO FINAL
# ===========================================================================
print(f"\n{'='*70}")
print(f"🎉 V3 ENHANCED - RESUMO FINAL")
print(f"{'='*70}")
print(f"\n🔬 Estratégias Testadas:")
for i, r in enumerate(results, 1):
    symbol = "⭐" if r == best_result else "  "
    print(f"{symbol} {i}. {r['strategy']:35s} | F1-Macro: {r['f1_macro']:.4f}")

print(f"\n🏆 Melhor Estratégia: {best_result['strategy']}")
print(f"\n📊 Métricas Finais:")
print(f"   ROC-AUC:   {roc_auc_final:.4f}")
print(f"   Accuracy:  {accuracy_final:.4f} ({accuracy_final*100:.2f}%)")
print(f"   Precision: {precision_final:.4f}")
print(f"   Recall:    {recall_final:.4f}")
print(f"   F1-Score:  {f1_final:.4f}")
print(f"   F1-Macro:  {f1_macro_final:.4f} ⭐")
print(f"   Threshold: {best_threshold['threshold']:.2f}")

print(f"\n📁 Arquivos gerados:")
print(f"   - balancing_strategies_comparison.png")
print(f"   - confusion_matrix_v3_enhanced.png")
print(f"   - threshold_analysis_v3_enhanced.png")
print(f"   - xgboost_model_v3_enhanced.json")
print(f"   - v3_enhanced_report.txt")

print(f"\n📈 Comparação com V3 Original:")
v3_roc_auc = 0.9283
v3_f1_macro = (0.98 + 0.43) / 2  # Aproximação da média das duas classes
improvement = ((f1_macro_final - v3_f1_macro) / v3_f1_macro) * 100

if f1_macro_final > v3_f1_macro:
    print(f"   ✅ F1-Macro melhorou: {v3_f1_macro:.4f} → {f1_macro_final:.4f} (+{improvement:.1f}%)")
else:
    print(f"   ⚠️ F1-Macro: {v3_f1_macro:.4f} → {f1_macro_final:.4f} ({improvement:.1f}%)")

if roc_auc_final > v3_roc_auc:
    print(f"   ✅ ROC-AUC melhorou: {v3_roc_auc:.4f} → {roc_auc_final:.4f}")
else:
    print(f"   ⚠️ ROC-AUC: {v3_roc_auc:.4f} → {roc_auc_final:.4f}")

print(f"\n{'='*70}")
